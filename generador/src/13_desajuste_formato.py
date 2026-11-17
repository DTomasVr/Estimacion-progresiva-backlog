"""How much does the model lose when the input is a real user story?

The lower half was trained on Jira-style parents ("Command line parser"), but
in PBE the input will be "As a <role>, I want <goal>, so that <benefit>".
To measure that mismatch without confounders, the SAME test examples are
rewritten into user-story form with a local LLM (llama3.1:8b, temperature 0)
and the reference tasks are left untouched. Any drop is the format mismatch.

Also reports the natural subset: test parents that already are user stories.
"""
import json
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, r"F:\Software Task generation Model\src")
import importlib.util  # noqa: E402

BASE = r"F:\Software Task generation Model"
spec = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

OLLAMA = "http://127.0.0.1:11434/api/generate"      # 127.0.0.1: localhost costs ~2 s per call on Windows
LLM = "llama3.1:8b"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150
SEED = 20260922
CFG = "beam4_actual"
MODELO = f"{BASE}/models/mt5-small-pruned_pbe_v1"
PROMPT = ("Rewrite this software backlog item as ONE user story, in exactly this form:\n"
          "As a <role>, I want <goal>, so that <benefit>.\n"
          "Keep the same scope and the same technical terms. Do not add new features. "
          "Output only the user story, nothing else.\n\nBacklog item: ")


def reescribir(texto):
    body = json.dumps({"model": LLM, "prompt": PROMPT + texto, "stream": False,
                       "options": {"temperature": 0, "num_predict": 120}}).encode()
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=180))["response"].strip()
    r = r.split("\n")[0].strip().strip('"')
    return re.sub(r"\s+", " ", r)


def main():
    t0 = time.time()
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_pruned_test.parquet")
    d = d[d.mitad == "tasks"].sample(N, random_state=SEED).reset_index(drop=True)
    partes = d.src.str.split(" | examples: ", regex=False)
    d["cabeza"] = [p[0].replace("decompose story into tasks: ", "") for p in partes]
    d["ejemplos"] = [p[1] if len(p) > 1 else "" for p in partes]

    hist = []
    for i, t in enumerate(d.cabeza):
        hist.append(reescribir(t))
        if i % 25 == 0:
            hecho = i + 1
            eta = (len(d) - hecho) * (time.time() - t0) / hecho
            print(f"  reescritas {hecho}/{len(d)}  ETA {eta/60:4.1f} min", flush=True)
    d["historia"] = hist
    ok = d.historia.str.contains(r"^as an? .+ i want", case=False, na=False)
    print(f"[{time.time()-t0:5.0f}s] reescritura lista; en formato correcto: {ok.mean():.1%}")
    for x in d.historia.head(3):
        print("   -", x[:160])

    sp = spm.SentencePieceProcessor(model_file=f"{BASE}/models/pruned_tok/spiece.model")
    d["src_historia"] = ("decompose story into tasks: " + d.historia + " | examples: " + d.ejemplos)
    ids_h = [x[:255] + [1] for x in sp.encode(d.src_historia.tolist())]

    tok, m = ev.cargar_modelo(MODELO)
    sb = SentenceTransformer(ev.SBERT, device="cuda")
    refs = [list(x) for x in d.tareas]
    res = {}
    for nombre, col in [("original", d.input_ids.tolist()), ("historia_reescrita", ids_h)]:
        dd = pd.DataFrame({"input_ids": pd.Series(col)})
        gens = ev.generar(tok, m, dd, ev.CFGS[CFG])
        res[nombre] = (ev.puntuar(sb, gens, refs), gens)
        s = res[nombre][0]
        print(f"  {nombre:<18} sF1 {s['sF1']:.4f}  prec {s['sPrec']:.4f}  cob {s['sCob']:.4f}  "
              f"ratio {s['ratio']:.2f}", flush=True)

    a = res["historia_reescrita"][0]["por_ejemplo"] - res["original"][0]["por_ejemplo"]
    rng = np.random.default_rng(SEED)
    bs = a[rng.integers(0, len(a), (2000, len(a)))].mean(1)
    print(f"\ndelta sF1 (historia - original): {a.mean():+.4f}  "
          f"IC95 [{np.quantile(bs,.025):+.4f}, {np.quantile(bs,.975):+.4f}]  n={len(a)}")

    with open(f"{BASE}/results/13_desajuste_muestras.txt", "w", encoding="utf-8") as fh:
        for i in range(min(12, len(d))):
            fh.write(f"=== {d.parent_uid[i]}\nORIGINAL: {d.cabeza[i][:200]}\n"
                     f"HISTORIA: {d.historia[i]}\nREF: {' ; '.join(refs[i][:6])}\n"
                     f"GEN(original): {' ; '.join(res['original'][1][i])}\n"
                     f"GEN(historia): {' ; '.join(res['historia_reescrita'][1][i])}\n\n")
    json.dump({k: {kk: float(vv) for kk, vv in v[0].items() if kk not in ("por_ejemplo", "mat", "cols")}
               for k, v in res.items()} | {"delta": float(a.mean()), "n": len(a),
                                           "formato_ok": float(ok.mean())},
              open(f"{BASE}/results/13_desajuste_formato.json", "w"), indent=1)
    d[["parent_uid", "cabeza", "historia"]].to_parquet(f"{BASE}/results/13_historias_reescritas.parquet",
                                                       index=False)
    print(f"total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
