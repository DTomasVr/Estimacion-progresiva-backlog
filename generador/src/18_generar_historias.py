"""One2One inference: sample many candidate stories for one feature/project
description, then turn them into a SET.

Sampling N times gives N candidates; near-duplicates are collapsed with the
fine-tuned SBERT (cosine >= UMBRAL) and each surviving story is scored by how
many samples voted for it -- a story the model produces again and again is one
it is confident about. The top K by votes is the returned backlog.

  --barrido MODEL    tune N / threshold on the VAL projects
  --test MODEL       held-out Dalpiaz projects (4, never trained on) + o2o test
"""
import importlib.util
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

BASE = r"F:\Software Task generation Model"
spec = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
SEED = 20260922
# same pattern as 12_buscar_historias.py: real stories also say "I would like" / "I need"
PAT = ev.re.compile(r"as\s+(?:an?|the)\s+[^,.]{2,60}[,;]?\s+i\s*(?:'d|\s+would)?\s*"
                    r"(?:want|would\s+like|need|wish|should|must|am\s+able)", ev.re.I)
MODELO = f"{BASE}/models/mt5-small-pruned_o2o_pbe_v1"


@torch.no_grad()
def candidatos(tok, m, ids, n, bs=8):
    """N sampled continuations per input -> list of lists of strings"""
    out = []
    for a in range(0, len(ids), bs):
        lote = [list(x) for x in ids[a:a + bs]]
        mx = max(map(len, lote))
        X = torch.zeros(len(lote), mx, dtype=torch.long)
        A = torch.zeros(len(lote), mx, dtype=torch.long)
        for k, x in enumerate(lote):
            X[k, :len(x)] = torch.tensor(x)
            A[k, :len(x)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = m.generate(input_ids=X.cuda(), attention_mask=A.cuda(), max_new_tokens=64,
                           do_sample=True, top_p=0.95, temperature=1.0,
                           num_return_sequences=n)
        t = tok.batch_decode(g, skip_special_tokens=True)
        out += [t[i * n:(i + 1) * n] for i in range(len(lote))]
    return out


def a_conjunto(sb, cands, umbral, k, solo_formato=True):
    """collapse near-duplicates, keep the most-voted K.
    solo_formato: when the story style was requested, drop candidates that did
    not come out as a user story (the model produces both); fall back to the
    rest only if too few survive."""
    salidas = []
    for c in cands:
        c = [x.strip() for x in c if len(x.strip()) >= 15]
        if solo_formato:
            en_fmt = [x for x in c if PAT.match(x)]
            c = en_fmt if len(en_fmt) >= 3 else c
        if not c:
            salidas.append([])
            continue
        E = sb.encode(c, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        S = (E @ E.T).float()
        usado = np.zeros(len(c), bool)
        grupos = []
        for i in np.argsort([-len(x) for x in c]):
            if usado[i]:
                continue
            m = (S[i] >= umbral).cpu().numpy() & ~usado
            usado |= m
            grupos.append((int(m.sum()), c[i]))
        grupos.sort(key=lambda x: -x[0])
        salidas.append([t for _, t in grupos[:k]])
    return salidas


def evaluar(sb, gens, refs, etiqueta):
    s = ev.puntuar(sb, gens, refs)
    fmt = np.mean([np.mean([bool(PAT.match(t)) for t in g]) if g else 0 for g in gens])
    print(f"  {etiqueta:<28} sF1 {s['sF1']:.4f}  prec {s['sPrec']:.4f}  cob {s['sCob']:.4f}  "
          f"n_gen {np.mean([len(g) for g in gens]):.1f}  formato {fmt:.0%}", flush=True)
    return dict(etiqueta=etiqueta, **{c: round(s[c], 4) for c in s["cols"]}, formato=round(fmt, 3),
                n_gen=round(float(np.mean([len(g) for g in gens])), 1))


def por_proyecto(d):
    """one row per project/epic: its description and all its real stories"""
    g = d.groupby("parent_uid").agg(src=("src", "first"), input_ids=("input_ids", "first"),
                                    refs=("objetivo", list), proj=("proj", "first"))
    return g.reset_index()


def main():
    modo, ruta = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else MODELO)
    t0 = time.time()
    sb = SentenceTransformer(ev.SBERT, device="cuda")
    tok, m = ev.cargar_modelo(ruta)
    filas = []
    if modo == "--barrido":
        d = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_val.parquet")
        d = d[d.estilo == "historia"]
        P = por_proyecto(d).head(40)
        refs = list(P.refs)
        print(f"  referencias en formato: {np.mean([bool(PAT.match(t)) for r in refs for t in r]):.0%}")
        for n in (30, 60):
            cands = candidatos(tok, m, list(P.input_ids), n)
            for solo in (True, False):
                for k in (10, 20):
                    gens = a_conjunto(sb, cands, 0.85, k, solo)
                    filas.append(evaluar(sb, gens, refs, f"N={n} K={k} solo_formato={solo}"))
        R = pd.DataFrame(filas).sort_values("sF1", ascending=False)
        R.to_csv(f"{BASE}/results/18_barrido_o2o.csv", index=False)
        print(R.to_string(index=False))
    else:
        for nombre, f in [("dalpiaz_test(4 proyectos)", "s2s_o2o_dalpiaz_test"),
                          ("o2o_test_historias", "s2s_o2o_test")]:
            d = pd.read_parquet(f"{BASE}/data/processed/{f}.parquet")
            d = d[d.estilo == "historia"]
            P = por_proyecto(d)
            if len(P) > 60:
                P = P.sample(60, random_state=SEED)
            base = [ev.base_recuperacion(x) for x in P.src]
            filas.append(evaluar(sb, base, list(P.refs), nombre + " [solo recuperacion]"))
            cands = candidatos(tok, m, list(P.input_ids), 30)
            gens = a_conjunto(sb, cands, 0.85, 20)
            filas.append(evaluar(sb, gens, list(P.refs), nombre + " [modelo]"))
            with open(f"{BASE}/results/18_muestras_{f}.txt", "w", encoding="utf-8") as fh:
                for (_, r), g in zip(P.iterrows(), gens):
                    fh.write(f"=== {r.parent_uid}\nSRC: {r.src[:220]}\nREF: {' ; '.join(r.refs[:6])}\n"
                             f"GEN: {' ; '.join(g[:8])}\n\n")
        pd.DataFrame(filas).to_csv(f"{BASE}/results/18_test_o2o.csv", index=False)
    print(f"total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
