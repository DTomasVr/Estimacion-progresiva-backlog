"""Raise coverage of the lower half without wrecking precision or speed.

Today: precision 0.58, coverage 0.44, ~0.48 generated tasks per real task --
the model proposes about half the backlog. The LLMs in the bench propose 4-8x
more tasks, which is most of their coverage advantage.

Criterion, fixed BEFORE looking at results: maximise soft F2 (coverage counts
double) subject to precision >= 0.50. Wall-clock per example is reported too:
a coverage gain that costs 10x the time would destroy the thesis' claim.

Strategies, all of them decoding-only (no retraining):
  beam            what we ship today
  muestreoN       N sampled lists, merged and de-duplicated
  beam+muestreoN  union of both
  un_ejemplo      one prompt per retrieved neighbour, union of the two
  dos_modelos     union of the two trained variants (bs16 and bs8)
"""
import importlib.util
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BASE = r"F:\Software Task generation Model"
spec = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
M1 = f"{BASE}/models/mt5-small-pruned_hist_agix_v1"
M2 = f"{BASE}/models/mt5-small-pruned_hist_agix_v1_bs8_lr9.2e-4"
SEED = 20260922
N_VAL = 300
PISO_PRECISION = 0.50


def unir(sb, listas, umbral=0.90):
    """merge task lists, dropping near-duplicates, keeping first-seen order"""
    salida = []
    for grupos in zip(*listas):
        plano = [t for g in grupos for t in g]
        if not plano:
            salida.append([])
            continue
        E = sb.encode(plano, convert_to_tensor=True, normalize_embeddings=True,
                      show_progress_bar=False)
        S = (E @ E.T).float().cpu().numpy()
        vistos, out = np.zeros(len(plano), bool), []
        for i in range(len(plano)):
            if vistos[i]:
                continue
            vistos |= (S[i] >= umbral)
            out.append(plano[i])
        salida.append(out)
    return salida


def f2(p, r):
    return 0.0 if p + r == 0 else 5 * p * r / (4 * p + r)


def main():
    sb = SentenceTransformer(ev.SBERT, device="cuda")
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_hist_val.parquet")
    d = d[(d.mitad == "tasks") & (d.estilo == "jira")].sample(N_VAL, random_state=SEED).reset_index(drop=True)
    refs = [list(x) for x in d.tareas]
    tok, m1 = ev.cargar_modelo(M1)

    def corre(cfg, datos=None):
        t = time.time()
        g = ev.generar(tok, m1, datos if datos is not None else d, cfg)
        return g, (time.time() - t) / len(d)

    cache = {}
    filas = []

    def anota(nombre, gens, seg):
        s = ev.puntuar(sb, gens, refs)
        filas.append(dict(estrategia=nombre, sF1=s["sF1"], precision=s["sPrec"], cobertura=s["sCob"],
                          F2=f2(s["sPrec"], s["sCob"]), ratio=s["ratio"],
                          tareas=float(np.mean([len(x) for x in gens])), seg_ej=seg,
                          cumple_piso=s["sPrec"] >= PISO_PRECISION))
        print(f"  {nombre:<20} F2 {filas[-1]['F2']:.4f}  prec {s['sPrec']:.4f}  cob {s['sCob']:.4f}  "
              f"sF1 {s['sF1']:.4f}  tareas {filas[-1]['tareas']:.1f}  {seg:.2f}s/ej", flush=True)

    beam, t_beam = corre(ev.CFGS["beam4_actual"])
    cache["beam"] = beam
    anota("beam (actual)", beam, t_beam)

    for n in (3, 5, 8):
        cfg = dict(do_sample=True, top_p=0.95, temperature=1.0, repetition_penalty=1.1, _muestras=n)
        g, t = corre(cfg)
        cache[f"m{n}"] = g
        anota(f"muestreo{n}", g, t)
        u = unir(sb, [beam, g])
        anota(f"beam+muestreo{n}", u, t + t_beam)

    # one prompt per retrieved neighbour, then union
    d1 = d.copy()
    d2 = d.copy()
    partes = [s.split(" | examples: ") for s in d.src]
    d1["src"] = [p[0] + " | examples: " + (p[1].split(" || ")[0] if len(p) > 1 else "") for p in partes]
    d2["src"] = [p[0] + " | examples: " + (p[1].split(" || ")[-1] if len(p) > 1 else "") for p in partes]
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=f"{BASE}/models/pruned_tok/spiece.model")
    for dd in (d1, d2):
        dd["input_ids"] = [x[:255] + [1] for x in sp.encode(dd.src.tolist())]
    g1, t1 = corre(ev.CFGS["beam4_actual"], d1)
    g2, t2 = corre(ev.CFGS["beam4_actual"], d2)
    anota("un_ejemplo x2", unir(sb, [g1, g2]), t1 + t2)
    anota("beam+un_ejemplo x2", unir(sb, [beam, g1, g2]), t_beam + t1 + t2)

    del m1
    torch.cuda.empty_cache()
    tok2, m2 = ev.cargar_modelo(M2)
    t = time.time()
    gb = ev.generar(tok2, m2, d, ev.CFGS["beam4_actual"])
    tb = (time.time() - t) / len(d)
    anota("dos_modelos", unir(sb, [beam, gb]), t_beam + tb)
    anota("dos_modelos+muestreo5", unir(sb, [beam, gb, cache["m5"]]), t_beam + tb + t_beam)

    R = pd.DataFrame(filas).round(4)
    R.to_csv(f"{BASE}/results/23_cobertura.csv", index=False)
    val = R[R.cumple_piso].sort_values("F2", ascending=False)
    print("\n" + R.sort_values("F2", ascending=False).to_string(index=False))
    print(f"\nGANA (F2 con precision >= {PISO_PRECISION}): {val.estrategia.iat[0]}  "
          f"F2 {val.F2.iat[0]:.4f} vs {R[R.estrategia=='beam (actual)'].F2.iat[0]:.4f} del actual")


if __name__ == "__main__":
    main()
