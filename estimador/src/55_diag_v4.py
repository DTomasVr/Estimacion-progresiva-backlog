"""What the public block answers that v3 alone could not.

The A/B said extra data does not move the fixed test set. That is a statement
about our test set, not about the model, so three separate questions remain and
each one is answerable only now that a second, much larger corpus exists:

  1. Cold start on organisations never seen at all -- 150 held-out projects,
     13,255 tasks, no row of which shares a project with training. This is the
     closest thing we have to Agix meeting a new customer.
  2. Is the ceiling ours or the data's? Train inside the public corpus and test
     inside it. If 53k in-domain rows still land near 0.68, the ceiling is a
     property of issue text, not of our pipeline or our sample size.
  3. Does the human estimate stay 2.4x better than the model outside SiP? Our
     steady-state story rests on one company and 18 developers; here there are
     ten organisations. Rows where timespent equals the estimate exactly are
     excluded from the strict reading, since Jira produces that equality
     mechanically.
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


def rasgos(emb, tr, *otros, min_freq=25):
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                       min_frequency=min_freq).fit(tr[CAT].astype(str))
    sc = StandardScaler().fit(np.log1p(tr[NUM].to_numpy(float)))
    f = lambda d: np.hstack([emb[d.fila_emb.to_numpy()],
                             oh.transform(d[CAT].astype(str)),
                             sc.transform(np.log1p(d[NUM].to_numpy(float)))])
    return (f(tr), *[f(o) for o in otros])


def contra_base(te, tr, p, minimo=12):
    y = te.effort_hours.to_numpy()
    gana = n = 0
    med = tr.groupby("project").effort_hours.median()
    glob = tr.effort_hours.median()
    for proj in te.project.unique():
        s = (te.project == proj).to_numpy()
        if s.sum() < minimo:
            continue
        b = float(med.get(proj, glob))
        n += 1
        if (E.point_metrics(y[s], p[s])["MdMRE"]
                < E.point_metrics(y[s], np.full(int(s.sum()), b))["MdMRE"]):
            gana += 1
    return gana, n


def linea(nombre, y, p, extra=""):
    m = E.point_metrics(y, p)
    rho = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    comp = float(np.std(np.log(np.clip(p, .01, None))) / np.std(np.log(y)))
    print(f"  {nombre:40s} MdMRE {m['MdMRE']:.4f}  PRED25 {m['PRED25']:.4f}  "
          f"MAE {m['MAE_h']:6.2f}h  rho {rho:+.3f}  compr {comp:.3f}  {extra}")
    return dict(escenario=nombre, MdMRE=round(m["MdMRE"], 4),
                PRED25=round(m["PRED25"], 4), PRED50=round(m["PRED50"], 4),
                MAE_h=round(m["MAE_h"], 2), RMSE_log=round(m["RMSE_log"], 4),
                spearman=round(rho, 3), compresion=round(comp, 3), nota=extra)


def main():
    sw, emb = cargar()
    v3 = sw[sw.source != "JiraPublico"]
    jp = sw[sw.source == "JiraPublico"]
    tr3 = v3[v3.split_t3 == "train"]
    trJ, vaJ, ext = (jp[jp.split_t3 == s] for s in ("train", "val", "externo"))
    pool = pd.concat([tr3, trJ], ignore_index=True)
    filas = []

    hgb = lambda: HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, learning_rate=0.06, l2_regularization=1.0,
        random_state=0)

    # ---------------------------------------------------------------- 1
    print("=" * 118)
    print("1. ARRANQUE EN FRIO: 150 proyectos que ningun modelo ha visto "
          f"({len(ext):,} tareas)")
    print("=" * 118)
    y = ext.effort_hours.to_numpy()
    filas.append(linea("mediana global del entrenamiento",
                       y, np.full(len(ext), float(pool.effort_hours.median()))))
    for etiqueta, tr in [("entrenado solo con v3", tr3),
                         ("entrenado solo con jira publico", trJ),
                         ("entrenado con ambos", pool)]:
        Xtr, Xte = rasgos(emb, tr, ext)
        m = hgb().fit(Xtr, np.log(tr.effort_hours.to_numpy()))
        p = np.exp(m.predict(Xte))
        g, n = contra_base(ext, tr, p)
        filas.append(linea(etiqueta, y, p, f"gana a la base {g}/{n}  "
                                           f"n_train={len(tr):,}"))

    # ---------------------------------------------------------------- 2
    print("\n" + "=" * 118)
    print(f"2. DENTRO DEL PROPIO CORPUS PUBLICO ({len(vaJ):,} tareas de prueba, "
          "mismos proyectos que el entrenamiento)")
    print("=" * 118)
    yj = vaJ.effort_hours.to_numpy()
    med = trJ.groupby("project").effort_hours.median()
    b = vaJ.project.map(med).fillna(trJ.effort_hours.median()).to_numpy()
    filas.append(linea("linea base: mediana del proyecto", yj, b))
    for frac in (0.1, 0.25, 0.5, 1.0):
        sub = trJ.sample(frac=frac, random_state=0) if frac < 1 else trJ
        Xtr, Xte = rasgos(emb, sub, vaJ)
        m = hgb().fit(Xtr, np.log(sub.effort_hours.to_numpy()))
        p = np.exp(m.predict(Xte))
        g, n = contra_base(vaJ, sub, p)
        filas.append(linea(f"hgb con {int(frac*100)}% del entrenamiento",
                           yj, p, f"gana a la base {g}/{n}  n_train={len(sub):,}"))

    # ---------------------------------------------------------------- 3
    print("\n" + "=" * 118)
    print("3. LA ESTIMACION HUMANA FUERA DE SiP")
    print("=" * 118)
    est = jp[jp.HoursEstimate.notna() & (jp.HoursEstimate > 0)].copy()
    lim = est[est.estimacion_igual_a_real == 0]
    print(f"  filas publicas con estimacion humana : {len(est):,}")
    print(f"  de ellas, timespent == estimacion    : "
          f"{int(est.estimacion_igual_a_real.sum()):,} "
          f"({100*est.estimacion_igual_a_real.mean():.1f}%)")
    print()
    for etiqueta, d in [("jira publico, todas", est),
                        ("jira publico, sin las identicas", lim)]:
        filas.append(linea(f"estimacion humana -- {etiqueta}",
                           d.effort_hours.to_numpy(),
                           d.HoursEstimate.to_numpy(), f"n={len(d):,}"))
    sip = v3[v3.HoursEstimate.notna() & (v3.HoursEstimate > 0)]
    filas.append(linea("estimacion humana -- SiP (referencia)",
                       sip.effort_hours.to_numpy(),
                       sip.HoursEstimate.to_numpy(), f"n={len(sip):,}"))

    print("\n  por organizacion (excluyendo las identicas):")
    for f, d in lim.groupby("fuente_jira"):
        if len(d) < 40:
            continue
        m = E.point_metrics(d.effort_hours.to_numpy(), d.HoursEstimate.to_numpy())
        rho = d.effort_hours.corr(d.HoursEstimate, method="spearman")
        print(f"    {f:16s} n={len(d):>5,}  MdMRE {m['MdMRE']:.4f}  "
              f"PRED25 {m['PRED25']:.4f}  rho {rho:+.3f}")

    print("\n  el modelo sobre esas mismas filas, para comparar:")
    Xtr, Xe = rasgos(emb, pool, lim)
    m = hgb().fit(Xtr, np.log(pool.effort_hours.to_numpy()))
    pm = np.exp(m.predict(Xe))
    filas.append(linea("modelo (agrupado) sobre las mismas filas",
                       lim.effort_hours.to_numpy(), pm, f"n={len(lim):,}"))
    ph = lim.HoursEstimate.to_numpy()
    comb = np.sqrt(np.clip(pm, .01, None) * ph)
    filas.append(linea("media geometrica modelo x estimacion",
                       lim.effort_hours.to_numpy(), comb))
    k = float(np.exp(np.median(np.log(lim.effort_hours / ph))))
    filas.append(linea("estimacion humana x factor unico",
                       lim.effort_hours.to_numpy(), ph * k, f"factor {k:.3f}"))

    r = pd.DataFrame(filas)
    r.to_csv("results/diag_v4.csv", index=False)
    print("\nescrito results/diag_v4.csv")


if __name__ == "__main__":
    main()
