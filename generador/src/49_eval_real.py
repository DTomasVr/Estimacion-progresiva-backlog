"""The honest test: generate stories for 6 real consumer projects and score them
against what PEOPLE actually wrote, not against what the teacher model wrote.

This is the only measurement in the distillation pilot that cannot be explained
by teacher mimicry. Everything else used qwen-written references.

n = 6 projects, so the confidence interval is wide by construction; the per-project
table is reported next to the pooled number, and the generated stories are printed
so they can be read.

  PBE_DEVICE=cpu python src/49_eval_real.py [viejo|nuevo|ambos]

out: results/49_real.csv
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PBE_DEVICE", "cpu")
spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
spec2 = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ev)

# se pueden apuntar a otra version con PBE_ARRIBA / PBE_ABAJO
NUEVO_ARRIBA = os.environ.get(
    "PBE_ARRIBA", f"{BASE}/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1_mixa_pbe_v1")
NUEVO_ABAJO = os.environ.get(
    "PBE_ABAJO", f"{BASE}/models/mt5-small-pruned_hist_pbe_v1_mix_pbe_v1")
SUFIJO = os.environ.get("PBE_SUFIJO", "")
T = json.load(open(f"{BASE}/data/reales/test_real.json", encoding="utf-8"))
cual = sys.argv[1] if len(sys.argv) > 1 else "ambos"

variantes = []
if cual in ("viejo", "ambos"):
    variantes.append(("viejo", pl.M_ARRIBA, pl.M_ABAJO))
if cual in ("nuevo", "ambos"):
    variantes.append(("nuevo", NUEVO_ARRIBA, NUEVO_ABAJO))

from sentence_transformers import SentenceTransformer  # noqa: E402

sb = SentenceTransformer(ev.SBERT, device=pl.DISPOSITIVO)
filas, puntos = [], {}
for etiqueta, arriba, abajo in variantes:
    pl.M_ARRIBA, pl.M_ABAJO = arriba, abajo
    P = pl.Pipeline(conjunto="no", traducir=False)
    gens, refs = [], []
    print(f"\n{'#'*100}\n##### {etiqueta}")
    for caso in T:
        r = P(caso["descripcion"])
        H = [h["historia"] for h in r["historias"]]
        gens.append(H)
        refs.append(caso["historias"])
        s = ev.puntuar(sb, [H], [caso["historias"]])
        filas.append(dict(variante=etiqueta, proyecto=caso["proyecto"], n_ref=len(caso["historias"]),
                          n_gen=len(H), sF1=round(s["sF1"], 4), precision=round(s["sPrec"], 4),
                          cobertura=round(s["sCob"], 4), segundos=round(r["tiempos"]["total"], 1),
                          votos_max=max(h["votos"] for h in r["historias"]),
                          relevancia=round(np.mean([h["relevancia"] for h in r["historias"]]), 3)))
        print(f"\n=== {caso['proyecto']}  sF1 {s['sF1']:.3f}  prec {s['sPrec']:.3f}  "
              f"cob {s['sCob']:.3f}  votos max {filas[-1]['votos_max']}")
        for h in r["historias"][:5]:
            print(f"   [{h['votos']}v {h['relevancia']:.2f}] {h['historia'][:115]}")
            for t in h["tareas"][:2]:
                print(f"        - {t[:95]}")
    s = ev.puntuar(sb, gens, refs)
    puntos[etiqueta] = s["mat"][:, 0]
    print(f"\n##### {etiqueta}: sF1 conjunto {s['sF1']:.4f}  precision {s['sPrec']:.4f}  "
          f"cobertura {s['sCob']:.4f}")
    del P

R = pd.DataFrame(filas)
print("\n=== por proyecto")
print(R.pivot_table(index="proyecto", columns="variante",
                    values=["sF1", "cobertura", "votos_max"]).round(3).to_string())
if len(puntos) == 2:
    d = puntos["nuevo"] - puntos["viejo"]
    rng = np.random.default_rng(ev.SEED)
    bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    lo, hi = np.quantile(bs, .025), np.quantile(bs, .975)
    print(f"\nnuevo - viejo sobre historias HUMANAS: {d.mean():+.4f} IC95 [{lo:+.4f}, {hi:+.4f}] "
          f"(n={len(d)} proyectos, intervalo ancho por construccion)")
    print(f"  gana el nuevo en {int((d > 0).sum())} de {len(d)} proyectos")
R.to_csv(f"{BASE}/results/49_real{SUFIJO}.csv", index=False)
print(f"\nescrito results/49_real.csv")
