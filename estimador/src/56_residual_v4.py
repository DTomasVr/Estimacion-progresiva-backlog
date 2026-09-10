"""Two controls and the model those controls imply.

Control 1 -- is the 29% "timespent equals the estimate exactly" rate an
artefact? SiP tracks time with a clock rather than by hand, so its exact-match
rate is the natural yardstick. If SiP sits near zero and public Jira near a
third, the equality is a logging habit and those rows are not evidence of good
estimating.

Control 2 -- the in-domain result showed the per-project median (MdMRE 0.5455)
beating the text model (0.6765). If the project's typical task size carries more
signal than the text, then asking the text model to predict the absolute number
wastes most of its capacity re-learning the project's scale. So predict the
RATIO instead: target log(hours) - log(median of that project), and multiply the
median back in at the end. The text then only has to say "bigger or smaller than
usual here", which is the question it can actually answer.

For a project Agix has never seen there is no median yet, so the arm is reported
in both regimes: with the project median available (steady state, after a
handful of finished tasks) and without it (cold start, falling back to the
global median).
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
DATA = "data/processed/agix_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
NUM = ["n_links", "n_links_estructurales", "n_componentes", "tiene_asignado"]
CAT = ["task_kind", "source"]


def cargar():
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    emb = np.load(f"{EMB}.npy")
    ids = np.load(f"{EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fila_emb"] = [pos[t] for t in sw.task_id]
    return sw, emb


def hacer_rasgos(emb, tr, min_freq=25):
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                       min_frequency=min_freq).fit(tr[CAT].astype(str))
    sc = StandardScaler().fit(np.log1p(tr[NUM].to_numpy(float)))
    return lambda d: np.hstack([emb[d.fila_emb.to_numpy()],
                                oh.transform(d[CAT].astype(str)),
                                sc.transform(np.log1p(d[NUM].to_numpy(float)))])


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                         learning_rate=0.06,
                                         l2_regularization=1.0, random_state=0)


def marca(nombre, y, p, filas, extra=""):
    m = E.point_metrics(y, p)
    rho = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    comp = float(np.std(np.log(np.clip(p, .01, None))) / np.std(np.log(y)))
    print(f"  {nombre:44s} MdMRE {m['MdMRE']:.4f}  PRED25 {m['PRED25']:.4f}  "
          f"PRED50 {m['PRED50']:.4f}  MAE {m['MAE_h']:6.2f}h  rho {rho:+.3f}  "
          f"compr {comp:.3f}  {extra}")
    filas.append(dict(arm=nombre, MdMRE=round(m["MdMRE"], 4),
                      PRED25=round(m["PRED25"], 4), PRED50=round(m["PRED50"], 4),
                      MAE_h=round(m["MAE_h"], 2), RMSE_log=round(m["RMSE_log"], 4),
                      spearman=round(rho, 3), compresion=round(comp, 3),
                      nota=extra))


def control_identidad(sw):
    print("=" * 118)
    print("CONTROL 1: cuando el tiempo registrado coincide EXACTAMENTE con la "
          "estimacion")
    print("=" * 118)
    for etiqueta, d in [("SiP (reloj automatico)",
                         sw[(sw.source == "SiP") & sw.HoursEstimate.notna()
                            & (sw.HoursEstimate > 0)]),
                        ("Jira publico (registro manual)",
                         sw[(sw.source == "JiraPublico") & sw.HoursEstimate.notna()
                            & (sw.HoursEstimate > 0)])]:
        if not len(d):
            continue
        ex = np.isclose(d.effort_hours, d.HoursEstimate, rtol=1e-9)
        cerca = (np.abs(np.log(d.effort_hours / d.HoursEstimate)) < 0.01)
        print(f"  {etiqueta:34s} n={len(d):>6,}  identicas {ex.sum():>5,} "
              f"({100*ex.mean():5.2f}%)   a menos del 1% {int(cerca.sum()):>5,} "
              f"({100*cerca.mean():5.2f}%)")
    print("  lectura: la igualdad exacta es un habito de registro, no una "
          "estimacion acertada.\n")


def main():
    sw, emb = cargar()
    control_identidad(sw)

    v3 = sw[sw.source != "JiraPublico"]
    jp = sw[sw.source == "JiraPublico"]
    tr3, va3, te3 = (v3[v3.split_t3 == s] for s in ("train", "val", "test"))
    trJ, vaJ, ext = (jp[jp.split_t3 == s] for s in ("train", "val", "externo"))
    pool = pd.concat([tr3, trJ], ignore_index=True)
    filas = []

    print("=" * 118)
    print("CONTROL 2: predecir la RAZON contra la mediana del proyecto")
    print("=" * 118)

    for etiqueta, tr, te in [
            ("dentro del corpus publico", trJ, vaJ),
            ("prueba fija de v3", pool, te3)]:
        print(f"\n--- {etiqueta}  (train {len(tr):,}  test {len(te):,}) ---")
        y = te.effort_hours.to_numpy()
        med = tr.groupby("project").effort_hours.median()
        glob = float(tr.effort_hours.median())
        base = te.project.map(med).fillna(glob).to_numpy()
        marca("linea base: mediana del proyecto", y, base, filas, etiqueta)

        R = hacer_rasgos(emb, tr)
        Xtr, Xte = R(tr), R(te)

        m_abs = hgb().fit(Xtr, np.log(tr.effort_hours.to_numpy()))
        p_abs = np.exp(m_abs.predict(Xte))
        marca("hgb absoluto (lo que veniamos haciendo)", y, p_abs, filas, etiqueta)

        # objetivo = razon contra la mediana del propio proyecto
        med_tr = tr.project.map(med).fillna(glob).to_numpy()
        obj = np.log(tr.effort_hours.to_numpy()) - np.log(med_tr)
        m_rat = hgb().fit(Xtr, obj)
        p_rat = np.exp(m_rat.predict(Xte)) * base
        marca("hgb sobre la razon x mediana del proyecto", y, p_rat, filas,
              etiqueta)

        # el mismo modelo de razon, pero sin mediana disponible (arranque frio)
        p_frio = np.exp(m_rat.predict(Xte)) * glob
        marca("razon x mediana global (arranque en frio)", y, p_frio, filas,
              etiqueta)

        # combinar el absoluto con la mediana del proyecto
        marca("media geometrica: absoluto x mediana proyecto", y,
              np.sqrt(p_abs * base), filas, etiqueta)

    # ------------------------------------------------------------------
    print("\n" + "=" * 118)
    print("EL MISMO MODELO DE RAZON EN ARRANQUE EN FRIO REAL "
          f"(150 proyectos nuevos, {len(ext):,} tareas)")
    print("=" * 118)
    y = ext.effort_hours.to_numpy()
    med = pool.groupby("project").effort_hours.median()
    glob = float(pool.effort_hours.median())
    R = hacer_rasgos(emb, pool)
    Xtr, Xte = R(pool), R(ext)
    med_tr = pool.project.map(med).fillna(glob).to_numpy()

    marca("mediana global", y, np.full(len(ext), glob), filas, "externo")
    m_abs = hgb().fit(Xtr, np.log(pool.effort_hours.to_numpy()))
    marca("hgb absoluto", y, np.exp(m_abs.predict(Xte)), filas, "externo")
    m_rat = hgb().fit(Xtr, np.log(pool.effort_hours.to_numpy()) - np.log(med_tr))
    marca("razon x mediana global", y, np.exp(m_rat.predict(Xte)) * glob, filas,
          "externo")

    # cuantas tareas hacen falta para que la mediana del proyecto sea util
    print("\n  cuantas tareas terminadas hacen falta para calibrar un proyecto "
          "nuevo:")
    rng = np.random.default_rng(11)
    p_rat_log = m_rat.predict(Xte)
    for k in (0, 3, 5, 10, 20, 50):
        preds, reales = [], []
        for proj, d in ext.groupby("project"):
            if len(d) < k + 10:
                continue
            idx = np.asarray(d.index)
            rng.shuffle(idx)
            vistas, resto = idx[:k], idx[k:]
            escala = (float(ext.loc[vistas].effort_hours.median()) if k
                      else glob)
            pos = ext.index.get_indexer(resto)
            preds.append(np.exp(p_rat_log[pos]) * escala)
            reales.append(ext.loc[resto].effort_hours.to_numpy())
        yy, pp = np.concatenate(reales), np.concatenate(preds)
        m = E.point_metrics(yy, pp)
        print(f"    {k:>3} tareas vistas -> MdMRE {m['MdMRE']:.4f}  "
              f"PRED25 {m['PRED25']:.4f}  (n={len(yy):,})")
        filas.append(dict(arm=f"razon, {k} tareas vistas del proyecto nuevo",
                          MdMRE=round(m["MdMRE"], 4),
                          PRED25=round(m["PRED25"], 4),
                          PRED50=round(m["PRED50"], 4),
                          MAE_h=round(m["MAE_h"], 2),
                          RMSE_log=round(m["RMSE_log"], 4), spearman=np.nan,
                          compresion=np.nan, nota="externo, calibracion en linea"))

    pd.DataFrame(filas).to_csv("results/residual_v4.csv", index=False)
    print("\nescrito results/residual_v4.csv")


if __name__ == "__main__":
    main()
