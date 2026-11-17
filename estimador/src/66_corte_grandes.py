"""Metrics when large tasks are excluded from the TEST set.

This narrows the question rather than improving the model: it answers "how good
is it on tasks under N hours", which is a legitimate scoped claim only if the
cutoff is stated every time the number is. The training set is untouched.
"""
import sys, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
sys.path.insert(0, "src")
import exp_common as E
warnings.filterwarnings("ignore"); pd.set_option("display.width", 220)
NUM = ["n_links", "n_links_estructurales", "n_componentes", "tiene_asignado"]
CAT = ["task_kind", "source"]

df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
sw = df[df.is_software_task == 1].copy()
emb = np.load("data/processed/embeddings/v4_mpnet_congelado.npy")
ids = np.load("data/processed/embeddings/v4_mpnet_congelado.ids.npy", allow_pickle=True)
pos = {t: i for i, t in enumerate(ids)}
sw["fe"] = [pos[t] for t in sw.task_id]
v3, jp = sw[sw.source != "JiraPublico"], sw[sw.source == "JiraPublico"]
tr3, te3 = v3[v3.split_t3 == "train"], v3[v3.split_t3 == "test"]
pool = pd.concat([tr3, jp[jp.split_t3 == "train"]], ignore_index=True)
ext = jp[jp.split_t3 == "externo"]

def R(tr):
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=25).fit(tr[CAT].astype(str))
    sc = StandardScaler().fit(np.log1p(tr[NUM].to_numpy(float)))
    return lambda d: np.hstack([emb[d.fe.to_numpy()], oh.transform(d[CAT].astype(str)),
                                sc.transform(np.log1p(d[NUM].to_numpy(float)))])
def hgb(): return HistGradientBoostingRegressor(max_iter=300, max_depth=6,
    learning_rate=.06, l2_regularization=1., random_state=0)
def SA(y, p, n=300):
    y = np.asarray(y, float); mar = np.mean(np.abs(y - p)); rng = np.random.default_rng(0)
    m0 = np.mean([np.mean(np.abs(y - y[(i := rng.integers(0, len(y)-1, len(y))) + (i >= np.arange(len(y)))])) for _ in range(n)])
    return 1 - mar/m0

f = R(pool); m = hgb().fit(f(pool), np.log(pool.effort_hours.to_numpy()))
p_te = np.exp(m.predict(f(te3)))

medP = pool.groupby("project").effort_hours.median(); g = float(pool.effort_hours.median())
mr = hgb().fit(f(pool), np.log(pool.effort_hours.to_numpy()) - np.log(pool.project.map(medP).fillna(g).to_numpy()))
e = ext.copy(); e["o"] = e.created.fillna(pd.Timestamp("2000-01-01"))
e = e.sort_values(["project", "o", "task_id"]).reset_index(drop=True); rz = mr.predict(f(e))
reg = []
for _, d in e.groupby("project", sort=False):
    idx, h = d.index.to_numpy(), d.effort_hours.to_numpy()
    for j in range(len(idx)):
        reg.append((j, float(np.exp(rz[idx[j]]) * (np.median(h[:j]) if j >= 3 else g)), h[j]))
rr = pd.DataFrame(reg, columns=["v", "p", "y"])

filas = []
for nom, y, p in [("prueba fija (Dataset 1)", te3.effort_hours.to_numpy(), p_te),
                  ("150 organizaciones nuevas", rr.y.to_numpy(), rr.p.to_numpy()),
                  ("  idem, regimen 50+", rr[rr.v >= 50].y.to_numpy(), rr[rr.v >= 50].p.to_numpy())]:
    for c in (np.inf, 40, 16, 8, 4):
        s = y <= c
        if s.sum() < 50: continue
        mm = E.point_metrics(y[s], p[s])
        filas.append(dict(bloque=nom, corte=("sin corte" if c == np.inf else f"<= {c:g} h"),
            n=int(s.sum()), pct=round(100*s.mean(), 1), MdMRE=round(mm["MdMRE"], 4),
            PRED25=round(mm["PRED25"], 4), PRED50=round(mm["PRED50"], 4),
            MAE_h=round(mm["MAE_h"], 2), SA=round(SA(y[s], p[s]), 3)))
print(pd.DataFrame(filas).to_string(index=False))
