"""How much do the metrics change when the model also receives the user's estimate?

Cold start gives the model only the task text. The alternative production design
asks the user for a rough guess first and lets the model use it. This script
measures that alternative on the only rows where a real human estimate exists:
SiP (HoursEstimate) and the Public Jira block (timeoriginalestimate). TAWOS has
story points only, which were already shown not to predict hours.

Two test scenarios, both untouched by training and selection:
  SiP test        same company, chronological, developers already known
  Jira externo    150 organisations never seen, developers never seen
Selection between arms is made on the validation blocks only.

Arms, all scored on exactly the same tasks:
  R0  cold model: text + category + numeric, NO estimate   (the reference)
  H0  the user's estimate, as typed
  H1  estimate x one global correction factor learned on train
  H2  estimate x per-developer factor, shrunk towards the global one
      (unseen developers fall back to the global factor, as they would in Agix)
  M1  model on the estimate + category only, no text      (isolates the text)
  M2  model on text + estimate                            (estimate as a feature)
  M3  corrector: text + estimate, target log(actual/estimate)
  G1  geometric mean of H0 and R0

Leakage guards: `estimacion_igual_a_real`, `effort_task_total`, `effort_hours_raw`,
`log_effort` and `n_merged` are derived from the target or from cleaning and never
enter a feature matrix. The estimate itself is known before the work starts.

Sensitivity: about a third of the rows have actual == estimate exactly. A model
that learns "copy the estimate" wins those for free, so every arm is also scored
on the rows where the two differ.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
DATA = "data/processed/agix_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
OUT = Path("results/estimacion_usuario")
NUM = ["n_links", "n_links_estructurales", "n_componentes", "tiene_asignado"]
CAT = ["task_kind", "source"]
PROHIBIDAS = {"estimacion_igual_a_real", "effort_task_total", "effort_hours_raw",
              "log_effort", "n_merged"}


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0)


def sa(y, p, corridas=500):
    rng, n = np.random.default_rng(0), len(y)
    base = []
    for _ in range(corridas):
        i = rng.integers(0, n - 1, n)
        i += i >= np.arange(n)
        base.append(np.mean(np.abs(y - y[i])))
    return 1 - np.mean(np.abs(y - p)) / np.mean(base)


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    m = E.point_metrics(y, p)
    return dict(MdMRE=m["MdMRE"], PRED25=m["PRED25"], PRED50=m["PRED50"],
                MAE_h=m["MAE_h"],
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")),
                SA=sa(y, p))


def comparar(y, a, b, n=2000):
    """Verdict of arm a against arm b on the same tasks."""
    rng = np.random.default_rng(1)
    ra, rb = np.abs(y - a) / y, np.abs(y - b) / y
    d = [np.median(ra[i]) - np.median(rb[i])
         for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    lo, hi = np.percentile(d, [2.5, 97.5])
    ea, eb = np.abs(np.log(y / a)), np.abs(np.log(y / b))
    p = wilcoxon(ea, eb).pvalue if np.any(ea != eb) else 1.0
    ver = "mejor" if hi < 0 else "peor" if lo > 0 else "empate"
    return f"{ver} [{lo:+.3f},{hi:+.3f}] p={p:.3g}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    emb = np.load(f"{EMB}.npy")
    ids = np.load(f"{EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fe"] = [pos[t] for t in sw.task_id]
    sw["est"] = sw.HoursEstimate.where(sw.HoursEstimate > 0)

    pool = sw[sw.split_t3 == "train"]                      # todo el entrenamiento
    con = sw[sw.est.notna() & sw.source.isin(["SiP", "JiraPublico"])]
    tr = con[con.split_t3 == "train"]
    va = con[con.split_t3 == "val"]
    pruebas = {"SiP · prueba (misma empresa)": con[(con.source == "SiP") & (con.split_t3 == "test")],
               "Jira · externo (cliente nuevo)": con[con.split_t3 == "externo"]}
    print(f"entrenamiento con estimacion {len(tr):,} · validacion {len(va):,} · "
          + " · ".join(f"{k} {len(v):,}" for k, v in pruebas.items()))

    # ------------------------------------------------ matrices de rasgos
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                       min_frequency=25).fit(pool[CAT].astype(str))
    sc = StandardScaler().fit(np.log1p(pool[NUM].to_numpy(float)))
    base = lambda d: np.hstack([emb[d.fe.to_numpy()], oh.transform(d[CAT].astype(str)),
                                sc.transform(np.log1p(d[NUM].to_numpy(float)))])
    loge = lambda d: np.log(d.est.to_numpy())[:, None]
    solo_est = lambda d: np.hstack([loge(d), oh.transform(d[CAT].astype(str))])
    con_est = lambda d: np.hstack([base(d), loge(d)])
    usadas = set(CAT) | set(NUM) | {"HoursEstimate"}
    assert not usadas & PROHIBIDAS, "una columna derivada del objetivo entro como rasgo"

    y_tr = np.log(tr.effort_hours.to_numpy())
    r0 = hgb().fit(base(pool), np.log(pool.effort_hours.to_numpy()))
    m1 = hgb().fit(solo_est(tr), y_tr)
    m2 = hgb().fit(con_est(tr), y_tr)
    m3 = hgb().fit(con_est(tr), y_tr - np.log(tr.est.to_numpy()))

    razon_tr = np.log(tr.effort_hours / tr.est)
    k_global = float(np.exp(np.median(razon_tr)))
    por_dev = (pd.DataFrame({"dev": tr.desarrollador, "r": razon_tr})
               .dropna().groupby("dev").r.agg(["sum", "size"]))

    def factor_dev(d, fuerza):
        mu = np.log(k_global)
        f = ((por_dev["sum"] + fuerza * mu) / (por_dev["size"] + fuerza))
        return np.exp(d.desarrollador.map(f).fillna(mu).to_numpy())

    def brazos(d, fuerza):
        e = d.est.to_numpy()
        r0p = np.exp(r0.predict(base(d)))
        return {"R0 · modelo en frío (sin estimación)": r0p,
                "H0 · estimación del usuario, tal cual": e,
                "H1 · estimación × factor global": e * k_global,
                "H2 · estimación × factor por desarrollador": e * factor_dev(d, fuerza),
                "M1 · modelo con estimación, sin texto": np.exp(m1.predict(solo_est(d))),
                "M2 · modelo con texto + estimación": np.exp(m2.predict(con_est(d))),
                "M3 · corrector: texto + estimación": e * np.exp(m3.predict(con_est(d))),
                "G1 · media geométrica estimación × frío": np.sqrt(e * r0p)}

    # ------------------------------------------------ seleccion en validacion
    y_va = va.effort_hours.to_numpy()
    fuerzas = {f: E.point_metrics(y_va, va.est.to_numpy() * factor_dev(va, f))["MdMRE"]
               for f in (1, 5, 10, 20, 50, 100)}
    fuerza = min(fuerzas, key=fuerzas.get)
    sel = {k: E.point_metrics(y_va, v)["MdMRE"] for k, v in brazos(va, fuerza).items()}
    print(f"\nfactor global {k_global:.3f} · contraccion por desarrollador elegida en "
          f"validacion: {fuerza} ({fuerzas})")
    print("MdMRE en validacion: " + " · ".join(f"{k[:2]} {v:.4f}" for k, v in sel.items()))
    ganador = min((k for k in sel if k[0] in "MG"), key=sel.get)
    print(f"brazo con modelo elegido en validacion: {ganador}")

    # ------------------------------------------------ lectura de prueba
    filas, preds = [], []
    for escenario, d in pruebas.items():
        b = brazos(d, fuerza)
        y = d.effort_hours.to_numpy()
        distinto = ~np.isclose(d.effort_hours, d.est)
        for subconjunto, m in (("todas", np.ones(len(d), bool)),
                               ("real ≠ estimación", distinto)):
            for nombre, p in b.items():
                fila = dict(escenario=escenario, subconjunto=subconjunto, brazo=nombre,
                            n=int(m.sum()), **metricas(y[m], p[m]))
                if not nombre.startswith("R0"):
                    fila["vs_frio_R0"] = comparar(y[m], np.clip(p[m], .01, None),
                                                  b["R0 · modelo en frío (sin estimación)"][m])
                if not nombre.startswith("H0"):
                    fila["vs_usuario_H0"] = comparar(y[m], np.clip(p[m], .01, None),
                                                     b["H0 · estimación del usuario, tal cual"][m])
                fila["elegido_en_val"] = nombre == ganador
                filas.append(fila)
        preds.append(pd.DataFrame({"escenario": escenario, "task_id": d.task_id.to_numpy(),
                                   "real": y, "estimacion": d.est.to_numpy(),
                                   **{k[:2]: v for k, v in b.items()}}))

    r = pd.DataFrame(filas)
    r.to_csv(OUT / "resumen.csv", index=False)
    pd.concat(preds).to_parquet(OUT / "predicciones.parquet", index=False)
    json.dump({"k_global": k_global, "contraccion_dev": fuerza, "val_MdMRE": sel,
               "ganador_val": ganador, "n_train": len(tr), "n_val": len(va)},
              open(OUT / "seleccion.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    for escenario in pruebas:
        for subconjunto in ("todas", "real ≠ estimación"):
            v = r[(r.escenario == escenario) & (r.subconjunto == subconjunto)]
            print(f"\n{'=' * 150}\n{escenario} · {subconjunto} · n={v.n.iloc[0]:,}\n{'=' * 150}")
            print(v[["brazo", "MdMRE", "PRED25", "PRED50", "MAE_h", "spearman", "SA",
                     "vs_frio_R0", "vs_usuario_H0"]].round(4).to_string(index=False))
    print(f"\nescrito {OUT}/resumen.csv, predicciones.parquet, seleccion.json")


if __name__ == "__main__":
    main()
