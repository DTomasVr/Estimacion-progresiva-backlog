"""E120c - Lo que deja E120b: el canal ajustado no aporta fuera de pliegue.

En E120b la mezcla base + C es PEOR que la base sola, con los dos codificadores
(large: SA 0,0232 contra 0,0268; small: 0,0053 contra 0,0207). Si eso aguanta,
sobran los dos codificadores ajustados del artefacto y basta con UNO.

Aqui se compara lo que de verdad se elegiria: base sola, ajustada a la rejilla,
large contra small, con intervalo por remuestreo de proyectos.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path("results/large_contra_small")


def main():
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    P = np.load(OUT / "predicciones.npz")
    rej = np.sort(pd.Series(y).value_counts().index.to_numpy()[:30])
    snap = lambda p: rej[np.abs(np.log(rej)[None, :] - np.log(p)[:, None]).argmin(1)]
    mar = np.abs(np.median(y) - y).mean()
    up = np.unique(proy)
    idx = {p: np.flatnonzero(proy == p) for p in up}
    rng = np.random.default_rng(0)
    B = [np.concatenate([idx[p] for p in rng.choice(up, len(up))]) for _ in range(1000)]
    filas, d = [], {}
    for k in ("large", "small"):
        p = P[f"{k}_base"]
        q = snap(p)
        d[k] = (p, q)
        filas.append({"codificador": k, "PRED25 rejilla": np.mean(np.abs(q - y) / y <= .25),
                      "SA": 1 - np.abs(p - y).mean() / mar,
                      "spearman": spearmanr(p, y).statistic,
                      "MAE_h": np.abs(p - y).mean()})
    r = pd.DataFrame(filas)
    print(r.round(4).to_string(index=False))
    res = {}
    for nom, f in (("PRED25", lambda p, q, s: np.mean(np.abs(q[s] - y[s]) / y[s] <= .25)),
                   ("SA", lambda p, q, s: 1 - np.abs(p[s] - y[s]).mean()
                    / np.abs(np.median(y[s]) - y[s]).mean()),
                   ("spearman", lambda p, q, s: spearmanr(p[s], y[s]).statistic)):
        full = np.arange(len(y))
        dif = f(*d["small"], full) - f(*d["large"], full)
        bs = [f(*d["small"], s) - f(*d["large"], s) for s in B]
        res[nom] = {"small_menos_large": float(dif),
                    "IC95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}
    r.to_csv(OUT / "base_sola.csv", index=False)
    json.dump(res, open(OUT / "base_sola.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
