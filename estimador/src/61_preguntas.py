"""Three questions about the design, each answered by measurement.

P1 · Does taking the LAST slice of every project as the test block introduce an
     error of its own? The decisive test uses a text-blind predictor: fit the
     per-project median on the first 75% and score it on the last 25%, then fit
     and score the same predictor on a RANDOM 75/25 of the same projects. A
     median knows nothing about text, so any gap between the two is drift in the
     target itself, not model failure.

P2 · SiP rows whose task involved several developers were re-expressed to the
     main contributor's hours (E3) but kept, along with `n_dev` and
     `effort_task_total`. Does keeping them, and keeping those columns as
     features, confuse training?

P3 · Should the model see only the text -- at most the category -- instead of
     the text plus one-hot(task_kind, source, desarrollador) plus the numeric
     block? Answered by ablating the feature blocks and reading the result on
     BOTH the v3 test set and the 150 unseen projects, because the answer is
     not the same in the two regimes.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
DATA = "data/processed/pbe_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
NUM = ["n_links", "n_links_estructurales", "n_componentes", "tiene_asignado"]


def cargar():
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    emb = np.load(f"{EMB}.npy")
    ids = np.load(f"{EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fila_emb"] = [pos[t] for t in sw.task_id]
    return sw, emb


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                         learning_rate=0.06,
                                         l2_regularization=1.0, random_state=0)


# ===================================================================== P1
def p1_deriva(sw):
    print("=" * 112)
    print("P1 · ¿Introduce error propio partir siempre por el final del proyecto?")
    print("=" * 112)
    v3 = sw[sw.source != "JiraPublico"].copy()
    v3["orden"] = v3.created.fillna(pd.Timestamp("2000-01-01"))

    rng = np.random.default_rng(7)
    filas = []
    for proj, d in v3.groupby("project"):
        if len(d) < 60:
            continue
        d = d.sort_values(["orden", "task_id"])
        n_te = max(15, int(len(d) * 0.25))
        cron_tr, cron_te = d.iloc[:-n_te], d.iloc[-n_te:]
        idx = rng.permutation(len(d))
        al_te, al_tr = d.iloc[idx[:n_te]], d.iloc[idx[n_te:]]
        for etiqueta, tr, te in [("cronologica", cron_tr, cron_te),
                                 ("aleatoria", al_tr, al_te)]:
            b = float(tr.effort_hours.median())
            m = E.point_metrics(te.effort_hours.to_numpy(),
                                np.full(len(te), b))
            filas.append(dict(project=proj, particion=etiqueta, n_test=len(te),
                              MdMRE=m["MdMRE"], PRED25=m["PRED25"],
                              mediana_tr=b,
                              mediana_te=float(te.effort_hours.median())))
    r = pd.DataFrame(filas)
    res = (r.groupby("particion")
           .agg(proyectos=("project", "nunique"),
                MdMRE_mediana=("MdMRE", "median"),
                MdMRE_medio=("MdMRE", "mean"),
                PRED25=("PRED25", "mean")).round(4))
    print("\nPredictor CIEGO AL TEXTO (mediana del bloque de entrenamiento):")
    print(res.to_string())

    piv = r.pivot(index="project", columns="particion", values="MdMRE")
    peor = int((piv.cronologica > piv.aleatoria).sum())
    print(f"\n  proyectos donde la partición cronológica es MÁS difícil: "
          f"{peor} de {len(piv)}")
    print(f"  penalización mediana por partir por el final: "
          f"{float((piv.cronologica - piv.aleatoria).median()):+.4f} MdMRE")

    # ¿en qué se diferencia el tramo final?
    cron = r[r.particion == "cronologica"]
    razon = cron.mediana_te / cron.mediana_tr
    print(f"\n  razón mediana(último 25%) / mediana(primer 75%):")
    print(f"    mediana entre proyectos {float(razon.median()):.3f}   "
          f"p25 {float(razon.quantile(.25)):.3f}   "
          f"p75 {float(razon.quantile(.75)):.3f}")
    print(f"    proyectos donde el tramo final es MÁS pesado: "
          f"{int((razon > 1).sum())} de {len(razon)}")

    # composición por tipo de tarea
    sw3 = sw[(sw.source != "JiraPublico") & sw.split_t3.notna()]
    mez = (sw3.groupby(["split_t3", "task_kind"]).size()
           .unstack(fill_value=0).T)
    mez = (100 * mez / mez.sum()).round(1)
    mez["deriva_pp"] = (mez.get("test", 0) - mez.get("train", 0)).round(1)
    print("\n  composición por categoría (% del bloque):")
    print(mez.sort_values("deriva_pp", key=abs, ascending=False).head(8).to_string())
    r.to_csv("results/p1_deriva_temporal.csv", index=False)
    return res


# ===================================================================== P2
def p2_multidev(sw, emb):
    print("\n" + "=" * 112)
    print("P2 · Las filas de SiP con varios desarrolladores, ¿confunden al modelo?")
    print("=" * 112)
    v3 = sw[sw.source != "JiraPublico"]
    tr3, va3, te3 = (v3[v3.split_t3 == s] for s in ("train", "val", "test"))

    multi = tr3.n_dev.fillna(1) > 1
    print(f"\nfilas de entrenamiento con varios desarrolladores: "
          f"{int(multi.sum()):,} de {len(tr3):,} ({100*multi.mean():.1f}%)")
    for etiqueta, d in [("una sola persona", tr3[~multi]),
                        ("varias personas (reexpresadas por E3)", tr3[multi])]:
        print(f"  {etiqueta:38s} n={len(d):>6,}  "
              f"mediana {d.effort_hours.median():5.2f} h  "
              f"sd(log) {np.std(np.log(d.effort_hours)):.3f}")

    tm = te3.n_dev.fillna(1) > 1
    print(f"\nen el bloque de prueba: {int(tm.sum()):,} de {len(te3):,}")

    def rasgos(tr, cols_cat, usar_num):
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                           min_frequency=25).fit(tr[cols_cat].astype(str))
        sc = (StandardScaler().fit(np.log1p(tr[NUM].to_numpy(float)))
              if usar_num else None)
        def f(d):
            partes = [emb[d.fila_emb.to_numpy()],
                      oh.transform(d[cols_cat].astype(str))]
            if sc is not None:
                partes.append(sc.transform(np.log1p(d[NUM].to_numpy(float))))
            return np.hstack(partes)
        return f

    CAT = ["task_kind", "source"]
    filas = []
    for etiqueta, entren in [
            ("A · todo el entrenamiento (lo actual)", tr3),
            ("B · sin las filas de varias personas", tr3[~multi])]:
        f = rasgos(entren, CAT, True)
        m = hgb().fit(f(entren), np.log(entren.effort_hours.to_numpy()))
        for bloque, d in [("prueba completa", te3),
                          ("prueba · solo una persona", te3[~tm]),
                          ("prueba · solo varias personas", te3[tm])]:
            if len(d) < 30:
                continue
            mm = E.point_metrics(d.effort_hours.to_numpy(), np.exp(m.predict(f(d))))
            filas.append(dict(entrenamiento=etiqueta, bloque=bloque, n=len(d),
                              MdMRE=round(mm["MdMRE"], 4),
                              PRED25=round(mm["PRED25"], 4)))
    print()
    print(pd.DataFrame(filas).to_string(index=False))

    # ¿aportan n_dev y effort_task_total como rasgos?
    print("\n  ¿aportan `n_dev` y `effort_task_total` usados como rasgos?")
    filas2 = []
    for etiqueta, extra in [("sin ellas (lo actual)", []),
                            ("añadiendo n_dev", ["n_dev"]),
                            ("añadiendo n_dev y effort_task_total",
                             ["n_dev", "effort_task_total"])]:
        cols = NUM + extra
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                           min_frequency=25).fit(tr3[CAT].astype(str))
        sc = StandardScaler().fit(np.log1p(tr3[cols].to_numpy(float)))
        f = lambda d: np.hstack([emb[d.fila_emb.to_numpy()],
                                 oh.transform(d[CAT].astype(str)),
                                 sc.transform(np.log1p(d[cols].to_numpy(float)))])
        m = hgb().fit(f(tr3), np.log(tr3.effort_hours.to_numpy()))
        mv = E.point_metrics(va3.effort_hours.to_numpy(), np.exp(m.predict(f(va3))))
        mt = E.point_metrics(te3.effort_hours.to_numpy(), np.exp(m.predict(f(te3))))
        filas2.append(dict(rasgos=etiqueta, MdMRE_val=round(mv["MdMRE"], 4),
                           MdMRE_test=round(mt["MdMRE"], 4),
                           PRED25_test=round(mt["PRED25"], 4)))
    print(pd.DataFrame(filas2).to_string(index=False))
    print("\n  AVISO: effort_task_total se deriva del objetivo. Se prueba aquí solo")
    print("  para cuantificar la fuga que supondría usarla; NO está en el modelo.")
    pd.DataFrame(filas + filas2).to_csv("results/p2_multidev.csv", index=False)


# ===================================================================== P3
def p3_ablacion(sw, emb):
    print("\n" + "=" * 112)
    print("P3 · ¿Basta con el texto? Ablación de los bloques de rasgos")
    print("=" * 112)
    v3 = sw[sw.source != "JiraPublico"]
    jp = sw[sw.source == "JiraPublico"]
    tr3, va3, te3 = (v3[v3.split_t3 == s] for s in ("train", "val", "test"))
    trJ, ext = jp[jp.split_t3 == "train"], jp[jp.split_t3 == "externo"]
    pool = pd.concat([tr3, trJ], ignore_index=True)

    BLOQUES = [
        ("1 · solo el vector del texto", [], False),
        ("2 · texto + categoría", ["task_kind"], False),
        ("3 · texto + categoría + fuente", ["task_kind", "source"], False),
        ("4 · texto + categoría + fuente + desarrollador",
         ["task_kind", "source", "desarrollador"], False),
        ("5 · todo lo anterior + numéricas",
         ["task_kind", "source", "desarrollador"], True),
        ("6 · el que usamos (sin desarrollador)", ["task_kind", "source"], True),
    ]

    def construir(tr, cats, usar_num):
        oh = (OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                            min_frequency=25).fit(tr[cats].astype(str))
              if cats else None)
        sc = (StandardScaler().fit(np.log1p(tr[NUM].to_numpy(float)))
              if usar_num else None)
        def f(d):
            partes = [emb[d.fila_emb.to_numpy()]]
            if oh is not None:
                partes.append(oh.transform(d[cats].astype(str)))
            if sc is not None:
                partes.append(sc.transform(np.log1p(d[NUM].to_numpy(float))))
            return np.hstack(partes)
        return f

    filas = []
    for etiqueta, cats, usar_num in BLOQUES:
        # regimen A: entrenar con v3, leer la prueba fija de v3
        f = construir(tr3, cats, usar_num)
        m = hgb().fit(f(tr3), np.log(tr3.effort_hours.to_numpy()))
        mv = E.point_metrics(va3.effort_hours.to_numpy(), np.exp(m.predict(f(va3))))
        mt = E.point_metrics(te3.effort_hours.to_numpy(), np.exp(m.predict(f(te3))))
        # regimen B: entrenar con todo, leer los 150 proyectos nuevos
        f2 = construir(pool, cats, usar_num)
        m2 = hgb().fit(f2(pool), np.log(pool.effort_hours.to_numpy()))
        pe = np.exp(m2.predict(f2(ext)))
        me = E.point_metrics(ext.effort_hours.to_numpy(), pe)
        rho = float(pd.Series(ext.effort_hours.to_numpy())
                    .corr(pd.Series(pe), method="spearman"))
        filas.append(dict(
            rasgos=etiqueta, n_rasgos=f(te3).shape[1],
            val_v3=round(mv["MdMRE"], 4), test_v3=round(mt["MdMRE"], 4),
            PRED25_v3=round(mt["PRED25"], 4),
            externo=round(me["MdMRE"], 4), PRED25_ext=round(me["PRED25"], 4),
            rho_ext=round(rho, 3)))
        print(f"  {etiqueta:46s} val {filas[-1]['val_v3']:.4f}  "
              f"test {filas[-1]['test_v3']:.4f}  externo {filas[-1]['externo']:.4f}",
              flush=True)

    r = pd.DataFrame(filas)
    print("\n" + r.to_string(index=False))
    r.to_csv("results/p3_ablacion.csv", index=False)

    base = r.iloc[0]
    print(f"\n  partiendo de solo texto (val {base.val_v3:.4f}):")
    for _, x in r.iloc[1:].iterrows():
        d = 100 * (base.val_v3 - x.val_v3) / base.val_v3
        print(f"    {x.rasgos:46s} {d:+5.2f}% en validación, "
              f"{100*(base.externo - x.externo)/base.externo:+5.2f}% en externo")


def main():
    sw, emb = cargar()
    p1_deriva(sw)
    p2_multidev(sw, emb)
    p3_ablacion(sw, emb)


if __name__ == "__main__":
    main()
