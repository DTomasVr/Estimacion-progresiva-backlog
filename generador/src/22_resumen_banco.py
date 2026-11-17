"""Scores the bench runs: speed first (that is the claim), then fidelity.

Quality is the same soft set-F1 used everywhere else, computed on the STORIES
against the project's real stories. Contamination is reported next to it: the
share of projects each LLM said it recognised, because its score on those is
not a clean out-of-sample number.
"""
import importlib.util
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE = r"F:\Software Task generation Model"
spec = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
PAT = ev.re.compile(r"as\s+(?:an?|the)\s+[^,.]{2,60}[,;]?\s+i\s*(?:'d|\s+would)?\s*"
                    r"(?:want|would\s+like|need|wish|should|must|am\s+able)", ev.re.I)

R = [json.loads(l) for l in open(f"{BASE}/results/20_banco.jsonl", encoding="utf-8")]
R = pd.DataFrame(R)
R["sistema"] = np.where(R.sistema == "pbe", "pbe (nuestro)", R.sistema + " " + R.condicion)

D = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_dalpiaz_test.parquet")
T = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_test.parquet")
ref = pd.concat([D, T[T.estilo == "historia"]]).groupby("parent_uid").objetivo.apply(list).to_dict()

sb = SentenceTransformer(ev.SBERT, device="cuda")
R = R[R.caso.isin(ref)]
s = ev.puntuar(sb, [list(x) for x in R.historias], [ref[c] for c in R.caso])
R["sF1"] = s["mat"][:, 0]
R["cob"] = s["mat"][:, 5]
R["formato"] = [np.mean([bool(PAT.match(t)) for t in h]) if len(h) else 0 for h in R.historias]
R["reconoce"] = R.get("sonda", pd.Series([""] * len(R))).fillna("").str.strip().str.lower().str.startswith(("yes", "si"))

g = R.groupby(["sistema", "fuente"]).agg(n=("caso", "size"), seg_mediana=("segundos", "median"),
                                         seg_p90=("segundos", lambda x: x.quantile(.9)),
                                         historias=("historias", lambda x: np.mean([len(i) for i in x])),
                                         tareas=("n_tareas", "mean"), sF1=("sF1", "mean"),
                                         cobertura=("cob", "mean"), formato=("formato", "mean"),
                                         reconoce=("reconoce", "mean")).round(3).reset_index()
print(g.to_string(index=False))
g.to_csv(f"{BASE}/results/22_banco_resumen.csv", index=False)

base = R[R.sistema == "pbe (nuestro)"].segundos.median()
v = R.groupby("sistema").segundos.median().sort_values()
print("\nveces mas lento que el nuestro (mediana):")
print((v / base).round(1).to_string())

# ---- paired fidelity: the deck claims a delta, so it must be computed here and
# on the SAME projects for both sides (a pooled mean over different project sets
# is not a comparison). Projects a system failed on are dropped from ITS pair only.
P = R.pivot_table(index="caso", columns="sistema", values="sF1")
nuestro = "pbe (nuestro)"
rng = np.random.default_rng(ev.SEED)
filas = []
for sis in [c for c in P.columns if c != nuestro]:
    d = P[[nuestro, sis]].dropna()
    dif = (d[sis] - d[nuestro]).values
    bs = dif[rng.integers(0, len(dif), (2000, len(dif)))].mean(1)
    filas.append(dict(sistema=sis, n=len(d), sF1_nuestro=round(d[nuestro].mean(), 4),
                      sF1_otro=round(d[sis].mean(), 4), delta=round(dif.mean(), 4),
                      ic95_lo=round(np.quantile(bs, .025), 4), ic95_hi=round(np.quantile(bs, .975), 4),
                      gana_otro=bool(np.quantile(bs, .025) > 0)))
C = pd.DataFrame(filas).sort_values("delta")
print("\nfidelidad pareada por proyecto (mismos proyectos en los dos lados):")
print(C.to_string(index=False))
C.to_csv(f"{BASE}/results/22_banco_pareado.csv", index=False)
