"""Does the Public Jira block actually improve the model? A/B on a fixed test.

Every arm is judged on the SAME 4,225 v3 test rows, and every arm is selected on
the SAME 2,528 v3 validation rows. Only the training material changes. That is
the only way the question "does more data help?" has an answer -- comparing
models across different test sets would compare the difficulty of the tests, not
the quality of the models.

The added block is not drawn from the same distribution as the target: public
Jira work logs have a median of 0.83 h against 2.0 h for TAWOS and SiP. So
several arms differ in HOW the extra data is used, not just whether:

  A  v3 only                       -- the control, our published protocol arm
  B  pooled, naive                 -- concatenate and hope
  C  pooled + domain indicator     -- let the model know which world a row is from
  D  pooled + median recalibration -- fit pooled, then shift log-predictions by
                                      the offset measured on v3 training rows
  E  pooled, v3 rows upweighted    -- sample weights, w for v3, 1 for public
  F  transfer: fit on public, use its prediction as a feature for a v3-only model
  G  public rows filtered to the v3 effort range

D and F are the two that respect the shift instead of ignoring it: D corrects
the location afterwards, F lets the public block contribute its ordering
(Spearman) while v3 keeps control of the scale.
"""
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

DATA = "data/processed/pbe_effort_v4.parquet"
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


def bloques(sw):
    v3 = sw[sw.source != "JiraPublico"]
    jp = sw[sw.source == "JiraPublico"]
    return (v3[v3.split_t3 == "train"], v3[v3.split_t3 == "val"],
            v3[v3.split_t3 == "test"], jp[jp.split_t3 == "train"],
            jp[jp.split_t3 == "val"], jp[jp.split_t3 == "externo"])


class Rasgos:
    """One-hot + scaler fitted on the arm's own training rows, never on test."""

    def __init__(self, emb, min_freq=25):
        self.emb = emb
        self.oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                min_frequency=min_freq)
        self.sc = StandardScaler()

    def fit(self, tr):
        self.oh.fit(tr[CAT].astype(str))
        self.sc.fit(np.log1p(tr[NUM].to_numpy(float)))
        return self

    def __call__(self, d):
        return np.hstack([self.emb[d.fila_emb.to_numpy()],
                          self.oh.transform(d[CAT].astype(str)),
                          self.sc.transform(np.log1p(d[NUM].to_numpy(float)))])


def puntuar(nombre, p_va, p_te, va, te, tr_v3, filas, seg, nota=""):
    y_va, y_te = va.effort_hours.to_numpy(), te.effort_hours.to_numpy()
    mv, mt = E.point_metrics(y_va, p_va), E.point_metrics(y_te, p_te)
    gana = n = 0
    for proj in te.project.unique():
        s = (te.project == proj).to_numpy()
        if s.sum() < 12:
            continue
        b = float(tr_v3[tr_v3.project == proj].effort_hours.median())
        if not np.isfinite(b):
            continue
        n += 1
        if (E.point_metrics(y_te[s], p_te[s])["MdMRE"]
                < E.point_metrics(y_te[s], np.full(int(s.sum()), b))["MdMRE"]):
            gana += 1
    filas.append(dict(
        arm=nombre, MdMRE_val=round(mv["MdMRE"], 4),
        MdMRE_test=round(mt["MdMRE"], 4), PRED25=round(mt["PRED25"], 4),
        PRED50=round(mt["PRED50"], 4), MAE_h=round(mt["MAE_h"], 2),
        RMSE_log=round(mt["RMSE_log"], 4),
        compresion=round(float(np.std(np.log(np.clip(p_te, .01, None)))
                               / np.std(np.log(y_te))), 3),
        spearman=round(float(pd.Series(y_te).corr(pd.Series(p_te),
                                                  method="spearman")), 3),
        gana_base=f"{gana}/{n}", seg=round(seg, 1), nota=nota))
    f = filas[-1]
    print(f"  {nombre:34s} val {f['MdMRE_val']:.4f}  test {f['MdMRE_test']:.4f}  "
          f"PRED25 {f['PRED25']:.3f}  rho {f['spearman']:.3f}  {f['seg']:6.1f}s")
    return filas[-1]


def main():
    sw, emb = cargar()
    tr3, va3, te3, trJ, vaJ, extJ = bloques(sw)
    print(f"v3   train {len(tr3):,}  val {len(va3):,}  test {len(te3):,}")
    print(f"jira train {len(trJ):,}  val {len(vaJ):,}  externo {len(extJ):,}")
    print(f"mediana de esfuerzo -- v3 {tr3.effort_hours.median():.2f} h   "
          f"jira {trJ.effort_hours.median():.2f} h\n")
    filas = []

    base_va = va3.project.map(tr3.groupby("project").effort_hours.median())
    base_te = te3.project.map(tr3.groupby("project").effort_hours.median())
    g = tr3.effort_hours.median()
    puntuar("linea base (mediana proyecto)", base_va.fillna(g).to_numpy(),
            base_te.fillna(g).to_numpy(), va3, te3, tr3, filas, 0.0)

    pool = pd.concat([tr3, trJ], ignore_index=True)
    y3, yp = np.log(tr3.effort_hours.to_numpy()), np.log(pool.effort_hours.to_numpy())

    modelos = {
        "ridge": lambda: Ridge(alpha=1.0),
        "hgb": lambda: HistGradientBoostingRegressor(
            max_iter=300, max_depth=6, learning_rate=0.06,
            l2_regularization=1.0, random_state=0),
        "rf": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, max_features="sqrt",
            n_jobs=-1, random_state=0),
    }

    # ---------------- A: solo v3 (control) ----------------
    rasg3 = Rasgos(emb).fit(tr3)
    X3, Xva, Xte = rasg3(tr3), rasg3(va3), rasg3(te3)
    print("--- A: entrenado solo con v3 (control) ---")
    ajus_A = {}
    for nom, f in modelos.items():
        t0 = time.time()
        m = f().fit(X3, y3)
        ajus_A[nom] = m
        puntuar(f"A solo v3 / {nom}", np.exp(m.predict(Xva)),
                np.exp(m.predict(Xte)), va3, te3, tr3, filas, time.time() - t0,
                f"n_train={len(tr3):,}")

    # ---------------- B, C, D, E: agrupado ----------------
    rasgP = Rasgos(emb).fit(pool)
    XP, XvaP, XteP = rasgP(pool), rasgP(va3), rasgP(te3)
    print(f"\n--- B: agrupado sin ajuste ({XP.shape[1]} rasgos) ---")
    ajus_B = {}
    for nom, f in modelos.items():
        t0 = time.time()
        m = f().fit(XP, yp)
        ajus_B[nom] = m
        puntuar(f"B agrupado / {nom}", np.exp(m.predict(XvaP)),
                np.exp(m.predict(XteP)), va3, te3, tr3, filas,
                time.time() - t0, f"n_train={len(pool):,}")

    # C: el indicador de dominio ya viaja en el one-hot de `source`; lo que
    # falta es forzar el dominio de destino al predecir
    print("\n--- C: agrupado, prediciendo como si toda fila fuera del dominio v3 ---")
    for nom in modelos:
        m = ajus_B[nom]
        va_c, te_c = va3.copy(), te3.copy()
        puntuar(f"C dominio forzado / {nom}", np.exp(m.predict(rasgP(va_c))),
                np.exp(m.predict(rasgP(te_c))), va3, te3, tr3, filas, 0.0,
                "source real de cada fila")

    print("\n--- D: agrupado + recalibrado de mediana con filas v3 ---")
    for nom in modelos:
        m = ajus_B[nom]
        delta = float(np.median(y3 - m.predict(rasgP(tr3))))
        puntuar(f"D recalibrado / {nom}", np.exp(m.predict(XvaP) + delta),
                np.exp(m.predict(XteP) + delta), va3, te3, tr3, filas, 0.0,
                f"delta={delta:+.3f} log ({np.exp(delta):.2f}x)")

    print("\n--- E: agrupado con filas v3 sobreponderadas ---")
    for w in (3, 6, 12):
        peso = np.r_[np.full(len(tr3), float(w)), np.ones(len(trJ))]
        for nom in ("ridge", "hgb"):
            t0 = time.time()
            m = modelos[nom]().fit(XP, yp, sample_weight=peso)
            puntuar(f"E peso x{w} / {nom}", np.exp(m.predict(XvaP)),
                    np.exp(m.predict(XteP)), va3, te3, tr3, filas,
                    time.time() - t0)

    # ---------------- F: transferencia por rasgo ----------------
    print("\n--- F: prediccion del modelo publico como rasgo de un modelo v3 ---")
    rasgJ = Rasgos(emb).fit(trJ)
    for nom in ("ridge", "hgb"):
        t0 = time.time()
        mj = modelos[nom]().fit(rasgJ(trJ), np.log(trJ.effort_hours.to_numpy()))
        anadir = lambda d, X: np.hstack([X, mj.predict(rasgJ(d))[:, None]])
        m = modelos[nom]().fit(anadir(tr3, X3), y3)
        puntuar(f"F transferencia / {nom}", np.exp(m.predict(anadir(va3, Xva))),
                np.exp(m.predict(anadir(te3, Xte))), va3, te3, tr3, filas,
                time.time() - t0, "rasgo extra = log-horas del modelo publico")

    # ---------------- G: bloque publico recortado al rango de v3 -------------
    lo, hi = tr3.effort_hours.quantile([.05, .95])
    trJf = trJ[(trJ.effort_hours >= lo) & (trJ.effort_hours <= hi)]
    poolF = pd.concat([tr3, trJf], ignore_index=True)
    rasgF = Rasgos(emb).fit(poolF)
    XF = rasgF(poolF)
    yF = np.log(poolF.effort_hours.to_numpy())
    print(f"\n--- G: publico recortado a [{lo:.2f}, {hi:.2f}] h "
          f"({len(trJf):,} de {len(trJ):,}) ---")
    for nom in ("ridge", "hgb"):
        t0 = time.time()
        m = modelos[nom]().fit(XF, yF)
        puntuar(f"G recortado / {nom}", np.exp(m.predict(rasgF(va3))),
                np.exp(m.predict(rasgF(te3))), va3, te3, tr3, filas,
                time.time() - t0, f"n_train={len(poolF):,}")

    r = pd.DataFrame(filas).sort_values("MdMRE_val")
    print("\n" + "=" * 150)
    print("ORDENADO POR VALIDACION v3 (la seleccion honesta)")
    print("=" * 150)
    print(r.to_string(index=False))
    r.to_csv("results/exp_v4.csv", index=False)

    mejor = r.iloc[0]
    ctrl = r[r.arm.str.startswith("A solo v3")].sort_values("MdMRE_val").iloc[0]
    print(f"\nmejor por validacion : {mejor.arm}  val {mejor.MdMRE_val:.4f}  "
          f"test {mejor.MdMRE_test:.4f}")
    print(f"control (solo v3)    : {ctrl.arm}  val {ctrl.MdMRE_val:.4f}  "
          f"test {ctrl.MdMRE_test:.4f}")
    d = (ctrl.MdMRE_test - mejor.MdMRE_test) / ctrl.MdMRE_test * 100
    print(f"cambio en prueba     : {d:+.2f}%  "
          f"({'mejora' if d > 0 else 'empeora'})")
    for _, x in r.iterrows():
        E.record("v4", x.arm.replace(" ", "_"), "split_t3_v4",
                 {"MdMRE": x.MdMRE_test, "PRED25": x.PRED25, "MAE_h": x.MAE_h,
                  "RMSE_log": x.RMSE_log, "MdMRE_val": x.MdMRE_val})
    json.dump({"mejor": mejor.arm, "control": ctrl.arm,
               "MdMRE_test_mejor": float(mejor.MdMRE_test),
               "MdMRE_test_control": float(ctrl.MdMRE_test)},
              open("results/v4_resumen.json", "w"), indent=1)
    print("\nescrito results/exp_v4.csv")


if __name__ == "__main__":
    main()
