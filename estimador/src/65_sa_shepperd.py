"""Standardised Accuracy (Shepperd & MacDonell, 2012) over every regime.

The Conte (1986) criteria that the thesis is being judged against -- MMRE <= 0.25
and PRED(25) >= 0.75 -- were shown to be biased: Kitchenham et al. (2001) and
Foss et al. (2003) demonstrated that MMRE systematically favours models that
under-estimate, so a model can win on MMRE by being timid rather than by being
right. Shepperd & MacDonell proposed replacing them with a measure that asks a
question no threshold can dodge: is this predictor better than guessing at
random?

    MAR   = mean absolute residual of the predictor
    MAR_0 = mean absolute residual of random guessing, defined as predicting for
            each case a value drawn uniformly from the other n-1 actuals
    SA    = 1 - MAR / MAR_0

SA is the fraction of random guessing's error that the predictor removes. SA <= 0
means the predictor is worth nothing. The effect size is Glass's delta against
the spread of the guessing runs, so that a small SA on a large sample is not
mistaken for a real effect.

Random guessing draws from the TEST actuals, per the original definition. That
makes it a demanding baseline -- it already knows the distribution the model had
to learn -- which is the point.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
DATA = "data/processed/agix_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
NUM = ["n_links", "n_links_estructurales", "n_componentes", "tiene_asignado"]
CAT = ["task_kind", "source"]
CORRIDAS = 1000


def mar_azar(y, corridas=CORRIDAS, semilla=0):
    """MAR of random guessing, and the spread across runs."""
    y = np.asarray(y, float)
    n = len(y)
    rng = np.random.default_rng(semilla)
    out = np.empty(corridas)
    for k in range(corridas):
        idx = rng.integers(0, n - 1, n)
        idx += (idx >= np.arange(n))          # nunca se elige a si mismo
        out[k] = float(np.mean(np.abs(y - y[idx])))
    return float(out.mean()), float(out.std(ddof=1))


def sa(y, p, corridas=CORRIDAS):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0.01, None)
    mar = float(np.mean(np.abs(y - p)))
    m0, s0 = mar_azar(y, corridas)
    return dict(n=len(y), MAR=round(mar, 3), MAR_azar=round(m0, 3),
                SA=round(1 - mar / m0, 4),
                delta=round((mar - m0) / s0, 1))


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


def main():
    sw, emb = cargar()
    v3 = sw[sw.source != "JiraPublico"]
    jp = sw[sw.source == "JiraPublico"]
    tr3, te3 = v3[v3.split_t3 == "train"], v3[v3.split_t3 == "test"]
    trJ, vaJ, ext = (jp[jp.split_t3 == s] for s in ("train", "val", "externo"))
    pool = pd.concat([tr3, trJ], ignore_index=True)

    def R(tr):
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                           min_frequency=25).fit(tr[CAT].astype(str))
        sc = StandardScaler().fit(np.log1p(tr[NUM].to_numpy(float)))
        return lambda d: np.hstack([emb[d.fila_emb.to_numpy()],
                                    oh.transform(d[CAT].astype(str)),
                                    sc.transform(np.log1p(d[NUM].to_numpy(float)))])

    filas = []

    def anotar(escenario, quien, y, p):
        r = sa(y, p)
        m = E.point_metrics(y, p)
        filas.append(dict(escenario=escenario, quien=quien, **r,
                          MdMRE=round(m["MdMRE"], 4),
                          PRED25=round(m["PRED25"], 4)))
        f = filas[-1]
        print(f"  {escenario:34s} {quien:36s} SA {f['SA']:+.4f}  "
              f"delta {f['delta']:+7.1f}  MdMRE {f['MdMRE']:.4f}", flush=True)

    print("=" * 124)
    print("STANDARDISED ACCURACY · SA = 1 - MAR/MAR_azar   "
          f"({CORRIDAS} corridas de azar por escenario)")
    print("=" * 124)

    # ---- prueba fija de v3
    f = R(pool)
    m = hgb().fit(f(pool), np.log(pool.effort_hours.to_numpy()))
    anotar("prueba fija (Dataset 1)", "modelo v4 agrupado",
           te3.effort_hours.to_numpy(), np.exp(m.predict(f(te3))))
    med3 = tr3.groupby("project").effort_hours.median()
    anotar("prueba fija (Dataset 1)", "linea base mediana proyecto",
           te3.effort_hours.to_numpy(),
           te3.project.map(med3).fillna(tr3.effort_hours.median()).to_numpy())

    # ---- dentro del corpus publico
    fJ = R(trJ)
    medJ = trJ.groupby("project").effort_hours.median()
    globJ = float(trJ.effort_hours.median())
    baseJ = vaJ.project.map(medJ).fillna(globJ).to_numpy()
    mr = hgb().fit(fJ(trJ), np.log(trJ.effort_hours.to_numpy())
                   - np.log(trJ.project.map(medJ).fillna(globJ).to_numpy()))
    anotar("dentro del corpus publico", "modelo en razon x mediana",
           vaJ.effort_hours.to_numpy(), np.exp(mr.predict(fJ(vaJ))) * baseJ)
    anotar("dentro del corpus publico", "linea base mediana proyecto",
           vaJ.effort_hours.to_numpy(), baseJ)

    # ---- curva de despliegue sobre organizaciones nuevas
    medP = pool.groupby("project").effort_hours.median()
    globP = float(pool.effort_hours.median())
    fp = R(pool)
    mrp = hgb().fit(fp(pool), np.log(pool.effort_hours.to_numpy())
                    - np.log(pool.project.map(medP).fillna(globP).to_numpy()))
    e = ext.copy()
    e["o"] = e.created.fillna(pd.Timestamp("2000-01-01"))
    e = e.sort_values(["project", "o", "task_id"]).reset_index(drop=True)
    razon = mrp.predict(fp(e))
    reg = []
    for proj, d in e.groupby("project", sort=False):
        idx, h = d.index.to_numpy(), d.effort_hours.to_numpy()
        for j in range(len(idx)):
            reg.append((j, float(np.exp(razon[idx[j]])
                                 * (np.median(h[:j]) if j >= 3 else globP)),
                        h[j]))
    r = pd.DataFrame(reg, columns=["vistas", "pred", "real"])
    anotar("150 organizaciones nuevas", "modelo + calibracion en linea",
           r.real.to_numpy(), r.pred.to_numpy())
    for lo, hi, etiq in [(0, 4, "arranque en frio (0-4)"),
                         (50, 10 ** 9, "regimen (50+)")]:
        s = r[(r.vistas >= lo) & (r.vistas <= hi)]
        anotar("150 organizaciones nuevas", etiq, s.real.to_numpy(),
               s.pred.to_numpy())

    # ---- la estimacion humana, el mismo rasero
    sip = v3[v3.HoursEstimate.notna() & (v3.HoursEstimate > 0)]
    anotar("SiP con estimacion", "experto humano",
           sip.effort_hours.to_numpy(), sip.HoursEstimate.to_numpy())
    pub = jp[jp.HoursEstimate.notna() & (jp.HoursEstimate > 0)]
    anotar("Jira publico con estimacion", "experto humano",
           pub.effort_hours.to_numpy(), pub.HoursEstimate.to_numpy())
    fp2 = R(pool)
    mp = hgb().fit(fp2(pool), np.log(pool.effort_hours.to_numpy()))
    pm = np.exp(mp.predict(fp2(pub)))
    anotar("Jira publico con estimacion", "modelo v4", pub.effort_hours.to_numpy(), pm)
    anotar("Jira publico con estimacion", "media geometrica humano x modelo",
           pub.effort_hours.to_numpy(),
           np.sqrt(pm * pub.HoursEstimate.to_numpy()))

    res = pd.DataFrame(filas)
    print("\n" + "=" * 124)
    print(res.to_string(index=False))
    res.to_csv("results/sa_shepperd.csv", index=False)

    print("\n" + "=" * 124)
    print("LECTURA")
    print("=" * 124)
    print("  SA = fraccion del error del azar que el predictor elimina.")
    print("  SA <= 0  ->  no vale mas que adivinar.")
    print("  delta    ->  tamano del efecto (Glass) contra la dispersion del azar;")
    print("               negativo y grande = mejor que el azar de forma solida.")
    peor = res[res.SA <= 0]
    print(f"\n  escenarios que NO superan al azar: {len(peor)} de {len(res)}")
    if len(peor):
        print(peor[["escenario", "quien", "SA"]].to_string(index=False))
    print(f"\n  mejor SA de un modelo  : "
          f"{res[~res.quien.str.contains('humano|linea base')].SA.max():.4f}")
    print(f"  mejor SA global        : {res.SA.max():.4f}  "
          f"({res.loc[res.SA.idxmax(), 'quien']})")


if __name__ == "__main__":
    main()
