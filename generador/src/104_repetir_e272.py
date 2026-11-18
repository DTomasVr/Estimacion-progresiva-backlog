"""Repite el brazo PBE de E-272 (PBE-Dataset/src/272_banco_generacion.py) con el pipeline
vigente, SIN tocar los resultados originales, y lo puntua con la misma metrica.

E-272 corrio sin PBE_SEED (muestreo top-p sin semilla), asi que una nueva corrida no puede
reproducir fila a fila: lo que se mide es cuanto se aparta del 0,387.

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
NUEVO = AD / "results/banco_generacion_rerun_18nov"
VIEJO = AD / "results/banco_generacion/pbe_actual.jsonl"


def generar():
    NUEVO.mkdir(parents=True, exist_ok=True)
    m.OUT = NUEVO
    m.brazo_pbe()


def medir():
    from sentence_transformers import SentenceTransformer
    ev = m.modulo("ev", m.GEN / "src" / "11_eval_gen.py")
    C = m.casos()
    ref = dict(zip(C.parent_uid, C.refs))
    sb = SentenceTransformer(ev.SBERT, device="cuda")
    juez = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device="cuda")
    filas = []
    for nombre, ruta in (("07/11 (paper)", VIEJO), ("18/11 (repeticion)", NUEVO / "pbe_actual.jsonl")):
        R = pd.DataFrame([json.loads(l) for l in open(ruta, encoding="utf-8")])
        R = R[R.caso.isin(ref)].reset_index(drop=True)
        sc = ev.puntuar(sb, [list(x) for x in R.historias], [list(ref[c]) for c in R.caso])
        R["sF1"], R["cobertura"] = sc["mat"][:, 0], sc["mat"][:, 5]
        dup = []
        for tt in R.tareas:
            t = [x.strip() for h in tt for x in h if x and x.strip()]
            if len(t) < 2:
                dup.append(np.nan)
                continue
            E = juez.encode(t, normalize_embeddings=True, show_progress_bar=False)
            S = E @ E.T
            np.fill_diagonal(S, -1)
            dup.append(float(np.mean(S.max(1) >= .90)))
        R["dup90"] = dup
        R["corrida"] = nombre
        filas.append(R[["corrida", "caso", "fuente", "segundos", "n_tareas", "sF1", "cobertura", "dup90"]])
    D = pd.concat(filas)
    D.to_csv(NUEVO / "comparacion.csv", index=False)
    print(D.groupby("corrida").agg(n=("caso", "size"), sF1=("sF1", "mean"), cobertura=("cobertura", "mean"),
                                   tareas=("n_tareas", "mean"), dup90=("dup90", "mean"),
                                   seg_mediana=("segundos", "median")).round(4).to_string())
    P = D.pivot_table(index="caso", columns="corrida", values="sF1").dropna()
    d = (P.iloc[:, 1] - P.iloc[:, 0]).to_numpy()
    b = np.random.default_rng(0).choice(d, (10000, len(d))).mean(1)
    print(f"diferencia emparejada {d.mean():+.4f}  IC95 [{np.quantile(b, .025):+.4f}, "
          f"{np.quantile(b, .975):+.4f}]  (n = {len(d)})")


{"generar": generar, "medir": medir}[sys.argv[1]]()
