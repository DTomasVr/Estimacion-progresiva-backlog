"""Does training on rewritten inputs fix the user-story mismatch?

Same test projects, three input conditions, two models (before / after):
  A jira       - the original Jira-style parent text
  B reescrita  - the same item rewritten as a user story (mistral)
  C real       - test parents that ALREADY were user stories, untouched.
                 This is the honest judge: human-written stories with the
                 tasks the team really created.

Targets are identical in all three, so any difference is the input format.
"""
import importlib.util
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BASE = r"F:\Software Task generation Model"
spec = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
CFG = "beam4_actual"
SEED = 20260922
MODELOS = {"antes": f"{BASE}/models/mt5-small-pruned_pbe_v1",
           "despues": f"{BASE}/models/mt5-small-pruned_hist_pbe_v1"}


def main():
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_hist_test.parquet")
    d = d[d.mitad == "tasks"].reset_index(drop=True)
    H = set(pd.read_parquet(f"{BASE}/data/interim/historias_formato.parquet").uid)
    cond = {
        "A_jira": d[d.estilo == "jira"].reset_index(drop=True),
        "B_reescrita": d[d.estilo == "historia_reescrita"].reset_index(drop=True),
        "C_real": d[(d.estilo == "jira") & d.parent_uid.isin(H)].reset_index(drop=True),
    }
    sb = SentenceTransformer(ev.SBERT, device="cuda")
    filas, por_ej = [], {}
    for mname, ruta in MODELOS.items():
        tok, m = ev.cargar_modelo(ruta)
        for cname, dd in cond.items():
            gens = ev.generar(tok, m, dd, ev.CFGS[CFG])
            s = ev.puntuar(sb, gens, [list(x) for x in dd.tareas])
            por_ej[(mname, cname)] = s["por_ejemplo"]
            filas.append(dict(modelo=mname, condicion=cname, n=len(dd),
                              **{c: round(s[c], 4) for c in s["cols"]}))
            print(f"  {mname:<8} {cname:<12} n={len(dd):<5} sF1 {s['sF1']:.4f}  "
                  f"prec {s['sPrec']:.4f}  cob {s['sCob']:.4f}  ratio {s['ratio']:.2f}", flush=True)
            if cname == "C_real" and mname == "despues":
                pd.DataFrame(dict(parent_uid=dd.parent_uid, gen=[" ; ".join(g) for g in gens],
                                  ref=[" ; ".join(t) for t in dd.tareas])).to_csv(
                    f"{BASE}/results/16_muestras_historias_reales.csv", index=False)
        del m
        torch.cuda.empty_cache()
    R = pd.DataFrame(filas)
    print(R.to_string(index=False))
    R.to_csv(f"{BASE}/results/16_eval_historias.csv", index=False)

    rng = np.random.default_rng(SEED)
    comp = []
    for c in cond:
        a, b = por_ej[("despues", c)], por_ej[("antes", c)]
        n = min(len(a), len(b))
        dd = a[:n] - b[:n]
        bs = dd[rng.integers(0, n, (2000, n))].mean(1)
        comp.append(dict(condicion=c, n=n, delta_sF1=dd.mean(),
                         ic95_lo=np.quantile(bs, .025), ic95_hi=np.quantile(bs, .975)))
    C = pd.DataFrame(comp).round(4)
    print("\ndespues - antes:\n", C.to_string(index=False))
    C.to_csv(f"{BASE}/results/16_eval_historias_delta.csv", index=False)


if __name__ == "__main__":
    main()
