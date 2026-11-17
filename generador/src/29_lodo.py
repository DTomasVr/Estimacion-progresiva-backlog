"""Leave-one-domain-out: what happens with a KIND of project never seen?

Splitting by project already measures new projects. This measures new domains:
for each held-out domain, a model is trained on everything EXCEPT that domain
and then scored on it. The comparison that matters is, for the SAME model:

  fuera   its score on the held-out domain (new kind of project)
  dentro  its score on test projects of the domains it did train on
          (new projects, familiar kind)

The gap between the two is the honest answer to "this only works for the
project types it was trained on".

Also checks whether retrieval similarity works as an "I don't know" signal:
if max neighbour cosine is lower out-of-domain AND predicts a lower score,
the pipeline can warn the user instead of pretending.

usage: python src/29_lodo.py [DOMINIOS...]   (default 2 3 4)
"""
import importlib.util
import json
import os
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers.optimization import Adafactor

BASE = r"F:\Software Task generation Model"
MODELO = f"{BASE}/models/mt5-small-pruned"
SEED = 20260922
EPOCAS, BS, LR, MAX_OUT = 5, 16, 1.4e-3, 192
N_EVAL = 350

spec = importlib.util.spec_from_file_location("tr", f"{BASE}/src/09_train_mt5.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)
spec2 = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ev)


def entrenar_sin(dom, TR, VA, tag):
    m = tr.modelo(MODELO)
    opt = Adafactor(m.parameters(), lr=LR, relative_step=False, scale_parameter=False)
    pasos = math.ceil(len(TR) / BS)
    total = pasos * EPOCAS
    w = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / w) * max(0.0, (total - s) / max(1, total - w)))
    rng = np.random.default_rng(SEED)
    mejor, sin_mejora, estado = float("inf"), 0, None
    for ep in range(1, EPOCAS + 1):
        t = time.time()
        for ii in tr.lotes(TR, BS, rng):
            X, A, Y = tr.colar(TR, ii)
            tr.paso(m, opt, X, A, Y)
            sched.step()
        v = tr.val_loss(m, VA, BS)
        print(f"    [{tag}] epoca {ep}: val {v:.4f}  {(time.time()-t)/60:.1f} min", flush=True)
        if v < mejor - 1e-4:
            mejor, sin_mejora = v, 0
            # keep the best weights in RAM: a GPU copy pushes past ~7 GB and WDDM
            # starts spilling, which made each epoch slower than the last (7 -> 20 min)
            estado = {k: x.detach().to('cpu', copy=True) for k, x in m.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= 1:
                break
    if estado:
        m.load_state_dict({k: v.cuda() for k, v in estado.items()})
    m.eval()
    m.config.use_cache = True
    return m, mejor


def main():
    doms = [int(x) for x in sys.argv[1:]] or [2, 3, 4]
    DOM = pd.read_parquet(f"{BASE}/data/processed/dominios.parquet")
    d_de = dict(zip(DOM.project_uid, DOM.dominio))
    tok = AutoTokenizer.from_pretrained(MODELO)   # SBERT is loaded only to score:
    # keeping it on the GPU during training pushed the process past 7 GB and WDDM
    # spilled to shared RAM, so epochs got slower and slower (7 -> 20 min)
    import os
    os.environ["DATOS"] = "hist"
    TR0 = tr.cargar("train", MAX_OUT)
    VA0 = tr.cargar("val", MAX_OUT)
    TE0 = tr.cargar("test", MAX_OUT)
    for d in (TR0, VA0, TE0):
        d["dominio"] = d.proj.map(d_de)
    filas = []
    for dom in doms:
        t0 = time.time()
        TR = TR0[TR0.dominio != dom].reset_index(drop=True)
        VA = VA0[VA0.dominio != dom].reset_index(drop=True)
        print(f"\n== dominio {dom} fuera: train {len(TR):,} (de {len(TR0):,})", flush=True)
        m, val = entrenar_sin(dom, TR, VA, f"sin{dom}")

        # BOTH sides come from TEST-split projects only. Pooling train-split projects
        # into the held-out side biased it easy: those are the big, formulaic projects.
        fuera = TE0[(TE0.dominio == dom) & (TE0.mitad == "tasks")].drop_duplicates("parent_uid")
        fuera = fuera.sample(min(N_EVAL, len(fuera)), random_state=SEED).reset_index(drop=True)
        dentro = TE0[(TE0.dominio != dom) & (TE0.mitad == "tasks")]
        dentro = dentro.sample(min(N_EVAL, len(dentro)), random_state=SEED).reset_index(drop=True)

        sb = SentenceTransformer(ev.SBERT, device="cuda")
        for etq, dd in [("fuera", fuera), ("dentro", dentro)]:
            gens = ev.generar(tok, m, dd, ev.CFGS["beam4_actual"])
            s = ev.puntuar(sb, gens, [list(x) for x in dd.tareas])
            sim = np.array([x[0] if len(x) else np.nan for x in dd.vecinos_sim])
            filas.append(dict(dominio=dom, conjunto=etq, n=len(dd), sF1=s["sF1"],
                              precision=s["sPrec"], cobertura=s["sCob"],
                              sim_vecino=float(np.nanmedian(sim)), val_loss=val,
                              corr_sim_sF1=float(np.corrcoef(sim[~np.isnan(sim)],
                                                            s["mat"][:, 0][~np.isnan(sim)])[0, 1])))
            print(f"  dominio {dom} {etq:<7} sF1 {s['sF1']:.4f}  cob {s['sCob']:.4f}  "
                  f"sim vecino {np.nanmedian(sim):.3f}  corr(sim,sF1) {filas[-1]['corr_sim_sF1']:+.3f}",
                  flush=True)
        del m, sb
        torch.cuda.empty_cache()
        print(f"  dominio {dom} listo en {(time.time()-t0)/60:.1f} min", flush=True)

    R = pd.DataFrame(filas).round(4)
    csv = f"{BASE}/results/29_lodo_v2.csv"
    R.to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
    R = pd.read_csv(csv).drop_duplicates(["dominio", "conjunto"], keep="last")
    print("\n" + R.to_string(index=False))
    piv = R.pivot(index="dominio", columns="conjunto", values="sF1")
    piv["caida"] = (piv["fuera"] - piv["dentro"]).round(4)
    piv["caida_%"] = (100 * piv.caida / piv["dentro"]).round(1)
    print("\n" + piv.to_string())
    json.dump(R.to_dict("records"), open(f"{BASE}/results/29_lodo.json", "w"), indent=1)


if __name__ == "__main__":
    main()
