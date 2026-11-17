"""Hyperparameter search for the lower half (story -> tasks), Optuna + ASHA.

Everything trained so far used values picked with short manual probes. This is
the systematic search the plan asked for: each trial trains on a 20 % subset
and is scored with the SAME metric the thesis reports (soft set-F1 on 120 val
examples), not with validation loss -- mT5-base had the better loss and the
worse generations, so loss is a poor proxy here.

ASHA (SuccessiveHalvingPruner) kills bad trials after the first epoch, so the
budget goes to the promising ones. The winner is then trained on FULL data by
22_final.py (or by hand).

usage: python src/21_busqueda_hp.py [HORAS] [N_TRIALS]
"""
import importlib.util
import json
import math
import sys
import time

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.pruners import SuccessiveHalvingPruner
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from transformers.optimization import Adafactor

BASE = r"F:\Software Task generation Model"
MODELO = f"{BASE}/models/mt5-small-pruned"
SEED = 20260922
N_VAL = 120

spec = importlib.util.spec_from_file_location("tr", f"{BASE}/src/09_train_mt5.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)
spec2 = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ev)

import os  # noqa: E402
os.environ["DATOS"] = "hist"
TR = tr.cargar("train", 192).sample(frac=0.2, random_state=SEED).reset_index(drop=True)
VA_FULL = tr.cargar("val", 192)
VA = VA_FULL[VA_FULL.mitad == "tasks"].sample(N_VAL, random_state=SEED).reset_index(drop=True)
SB = SentenceTransformer(ev.SBERT, device="cuda")
TOK = AutoTokenizer.from_pretrained(MODELO)
REFS = [list(x) for x in VA.tareas]
print(f"busqueda: train {len(TR):,} (20 %)  val {len(VA)}", flush=True)


def calidad(m):
    m.eval()
    m.config.use_cache = True
    gens = ev.generar(TOK, m, VA, ev.CFGS["beam4_actual"], bs=16)
    s = ev.puntuar(SB, gens, REFS)
    m.config.use_cache = False
    m.train()
    return s["sF1"], s["sCob"]


def objetivo(trial):
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    bs = trial.suggest_categorical("bs", [8, 16, 24])
    warm = trial.suggest_float("warmup", 0.02, 0.15)
    wd = trial.suggest_categorical("clip", [0.5, 1.0, 2.0])
    epocas = 3
    m = tr.modelo(MODELO)
    opt = Adafactor(m.parameters(), lr=lr, relative_step=False, scale_parameter=False)
    pasos_ep = math.ceil(len(TR) / bs)
    total = pasos_ep * epocas
    w = max(1, int(warm * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / w) * max(0.0, (total - s) / max(1, total - w)))
    rng = np.random.default_rng(SEED)
    mejor = 0.0
    try:
        for ep in range(1, epocas + 1):
            for ii in tr.lotes(TR, bs, rng):
                X, A, Y = tr.colar(TR, ii)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = m(input_ids=X, attention_mask=A, labels=Y).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), wd)
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
            sf1, cob = calidad(m)
            mejor = max(mejor, sf1)
            trial.report(sf1, ep)
            print(f"  trial {trial.number} ep{ep}: sF1 {sf1:.4f} cob {cob:.4f} "
                  f"(lr {lr:.2e} bs {bs})", flush=True)
            if trial.should_prune():
                raise optuna.TrialPruned()
    finally:
        del m, opt
        torch.cuda.empty_cache()
    return mejor


def main():
    horas = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    est = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                              pruner=SuccessiveHalvingPruner(min_resource=1, reduction_factor=3),
                              study_name="pbe_abajo",
                              storage=f"sqlite:///{BASE}/results/21_optuna.db", load_if_exists=True)
    est.optimize(objetivo, n_trials=n, timeout=horas * 3600, gc_after_trial=True)
    print("\nmejor:", est.best_params, round(est.best_value, 4))
    df = est.trials_dataframe()
    df.to_csv(f"{BASE}/results/21_trials.csv", index=False)
    json.dump(dict(mejor=est.best_params, sF1=est.best_value,
                   n_trials=len(est.trials),
                   podados=sum(t.state.name == "PRUNED" for t in est.trials)),
              open(f"{BASE}/results/21_busqueda.json", "w"), indent=1)
    print(df.sort_values("value", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
