"""Repeats the PBE arm of E-272 (PBE-Dataset/src/272_banco_generacion.py) with the current
pipeline, WITHOUT touching the original results, and scores it with the same metric.

E-272 ran without PBE_SEED (top-p sampling with no seed), so a new run cannot reproduce
row by row: what is measured is how far it moves from the 0,387.

  python src/104_repetir_e272.py generar   (GPU, ~12 min)
  python src/104_repetir_e272.py medir
out: PBE-Dataset/results/banco_generacion_rerun_18nov/{pbe_actual.jsonl, comparacion.csv}
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AD = Path(r"C:\Users\User\Documents\Programacao\PBE-Dataset")
os.chdir(AD)
sys.stdout.reconfigure(encoding="utf-8")
s = importlib.util.spec_from_file_location("e272", AD / "src/272_banco_generacion.py")
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
NEW = AD / "results/banco_generacion_rerun_18nov"
OLD = AD / "results/banco_generacion/pbe_actual.jsonl"


def generate():
    NEW.mkdir(parents=True, exist_ok=True)
    m.OUT = NEW
    m.arm_pbe()


def measure():
    from sentence_transformers import SentenceTransformer
    ev = m.module("ev", m.GEN / "src" / "11_eval_gen.py")
    C = m.casos()
    ref = dict(zip(C.parent_uid, C.refs))
    sb = SentenceTransformer(ev.SBERT, device="cuda")
    judge = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device="cuda")
    rows = []
    for name, path in (("07/11 (paper)", OLD), ("18/11 (rerun)", NEW / "pbe_actual.jsonl")):
        # raw JSONL field names (caso, fuente, segundos, n_tareas, historias, tareas) -> English
        R = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8")])
        R = R[R.caso.isin(ref)].reset_index(drop=True)
        sc = ev.puntuar(sb, [list(x) for x in R.historias], [list(ref[c]) for c in R.caso])
        R["sF1"], R["coverage"] = sc["mat"][:, 0], sc["mat"][:, 5]
        dup = []
        for tt in R.tareas:
            t = [x.strip() for h in tt for x in h if x and x.strip()]
            if len(t) < 2:
                dup.append(np.nan)
                continue
            E = judge.encode(t, normalize_embeddings=True, show_progress_bar=False)
            S = E @ E.T
            np.fill_diagonal(S, -1)
            dup.append(float(np.mean(S.max(1) >= .90)))
        R["dup90"] = dup
        R["run"] = name
        R = R.rename(columns={"caso": "case", "fuente": "source", "segundos": "seconds", "n_tareas": "n_tasks"})
        rows.append(R[["run", "case", "source", "seconds", "n_tasks", "sF1", "coverage", "dup90"]])
    D = pd.concat(rows)
    D.to_csv(NEW / "comparacion.csv", index=False)
    print(D.groupby("run").agg(n=("case", "size"), sF1=("sF1", "mean"), coverage=("coverage", "mean"),
                               tasks=("n_tasks", "mean"), dup90=("dup90", "mean"),
                               sec_median=("seconds", "median")).round(4).to_string())
    P = D.pivot_table(index="case", columns="run", values="sF1").dropna()
    d = (P.iloc[:, 1] - P.iloc[:, 0]).to_numpy()
    b = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1)
    print(f"paired difference {d.mean():+.4f}  CI95 [{np.quantile(b, .025):+.4f}, "
          f"{np.quantile(b, .975):+.4f}]  (n = {len(d)})")


{"generar": generate, "medir": measure}[sys.argv[1]]()
