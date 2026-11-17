"""Generation tuning + evaluation for the seq2seq halves.

There is no single right decomposition, so scoring is similarity-based, not
exact match. Primary metric (fixed before looking at test): soft set-F1 in the
style of BERTScore, computed with the fine-tuned SBERT --

    P = mean over generated tasks of max cosine to any reference task
    R = mean over reference tasks of max cosine to any generated task
    F1 = 2PR/(P+R)

Secondary: hard F1 with a 0.7 cosine match threshold, ROUGE-1 F1 over the
joined list, and the generated/reference task-count ratio.

Baseline: retrieval-only -- the tasks of the nearest neighbour project, which
are already inside the prompt (this is the "fully retrieval v1" fallback).

modes:
  --barrido MODEL [N]    sweep generation configs on VAL
  --test MODEL [MODEL2]  evaluate on TEST with the config picked on VAL
  --dalpiaz MODEL        upper half on the Dalpiaz projects (out of domain)
"""
import itertools
import json
import re
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, MT5ForConditionalGeneration

BASE = r"F:\Software Task generation Model"
SBERT = f"{BASE}/models/sbert_agix_v1"
SEED = 20260922
BS = 16


def cargar_modelo(ruta):
    tok = AutoTokenizer.from_pretrained(ruta)
    m = MT5ForConditionalGeneration.from_pretrained(ruta, dtype=torch.float32).cuda().eval()
    m.config.use_cache = True
    return tok, m


def separar(texto):
    """decoded string -> list of tasks, trimmed and de-duplicated case-insensitively"""
    out, vistos = [], set()
    for t in texto.split(";"):
        t = re.sub(r"\s+", " ", t).strip(" .")
        k = t.lower()
        if len(t) >= 5 and k not in vistos:
            vistos.add(k)
            out.append(t)
    return out


@torch.no_grad()
def generar(tok, m, d, cfg, bs=BS):
    salidas = []
    ids = d.input_ids.tolist()
    orden = np.argsort([len(x) for x in ids])
    por_idx = {}
    for a in range(0, len(orden), bs):
        sel = orden[a:a + bs]
        lote = [list(ids[i]) for i in sel]
        mx = max(map(len, lote))
        X = torch.zeros(len(lote), mx, dtype=torch.long)
        A = torch.zeros(len(lote), mx, dtype=torch.long)
        for k, x in enumerate(lote):
            X[k, :len(x)] = torch.tensor(x)
            A[k, :len(x)] = 1
        n_m = cfg.get("_muestras", 1)
        c = {k: v for k, v in cfg.items() if k != "_muestras"}
        if n_m > 1:
            c["num_return_sequences"] = n_m
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = m.generate(input_ids=X.cuda(), attention_mask=A.cuda(), max_new_tokens=192, **c)
        txt = tok.batch_decode(g, skip_special_tokens=True)
        for j, i in enumerate(sel):
            # several samples are merged into one set, then de-duplicated
            por_idx[i] = separar(" ; ".join(txt[j * n_m:(j + 1) * n_m]))
    salidas = [por_idx[i] for i in range(len(ids))]
    return salidas


def base_recuperacion(src):
    """tasks of the first retrieved neighbour, as embedded in the prompt"""
    m = re.search(r"\| examples: (.*)$", src, re.S)
    if not m:
        return []
    primero = m.group(1).split(" || ")[0]
    return separar(primero.split(" => ", 1)[1]) if " => " in primero else []


def rouge1(gen, ref):
    g = re.findall(r"[a-z0-9]+", " ; ".join(gen).lower())
    r = re.findall(r"[a-z0-9]+", " ; ".join(ref).lower())
    if not g or not r:
        return 0.0
    from collections import Counter
    c = Counter(g) & Counter(r)
    s = sum(c.values())
    p, rc = s / len(g), s / len(r)
    return 0.0 if p + rc == 0 else 2 * p * rc / (p + rc)


def puntuar(sb, gens, refs):
    """soft set-F1 (BERTScore style) + hard F1@0.7 + rouge1 + count ratio"""
    planos = [t for x in gens + refs for t in x]
    if not planos:
        return dict(sF1=0, hF1=0, rouge1=0, ratio=0)
    E = sb.encode(planos, batch_size=256, convert_to_tensor=True, normalize_embeddings=True,
                  show_progress_bar=False)
    pos, i = {}, 0
    for x in gens + refs:
        for t in x:
            pos.setdefault(t, i)
            i += 1
    filas = []
    off_g = 0
    idx_g, idx_r = [], []
    for x in gens:
        idx_g.append(list(range(off_g, off_g + len(x))))
        off_g += len(x)
    off_r = off_g
    for x in refs:
        idx_r.append(list(range(off_r, off_r + len(x))))
        off_r += len(x)
    for gi, ri, g, r in zip(idx_g, idx_r, gens, refs):
        if not g or not r:
            filas.append((0.0, 0.0, rouge1(g, r), len(g) / max(1, len(r)), 0.0, 0.0))
            continue
        S = (E[gi] @ E[ri].T).float()
        P, R = S.max(1).values.mean().item(), S.max(0).values.mean().item()
        sF1 = 0 if P + R == 0 else 2 * P * R / (P + R)
        emp = (S >= 0.7)
        tp = min(int(emp.any(1).sum()), int(emp.any(0).sum()))
        hp, hr = tp / len(g), tp / len(r)
        hF1 = 0 if hp + hr == 0 else 2 * hp * hr / (hp + hr)
        filas.append((sF1, hF1, rouge1(g, r), len(g) / len(r), P, R))
    a = np.array(filas)
    cols = ["sF1", "hF1", "rouge1", "ratio", "sPrec", "sCob"]
    return dict(**{c: a[:, i].mean() for i, c in enumerate(cols)}, por_ejemplo=a[:, 0], mat=a, cols=cols)


CFGS = {
    # the model loops instead of stopping, so everything here is about repetition
    "beam4_actual": dict(num_beams=4, no_repeat_ngram_size=4, repetition_penalty=1.3),
    "beam4_ngram6": dict(num_beams=4, no_repeat_ngram_size=6),
    "beam4_ngram8_rp1.15": dict(num_beams=4, no_repeat_ngram_size=8, repetition_penalty=1.15),
    "beam4_ngram10": dict(num_beams=4, no_repeat_ngram_size=10),
    # (group beam search was moved out of transformers v5 core: needs remote code)
    "beam4_ngram6_rp1.05": dict(num_beams=4, no_repeat_ngram_size=6, repetition_penalty=1.05),
    "muestreo3_p0.92": dict(do_sample=True, top_p=0.92, temperature=0.9, repetition_penalty=1.1,
                            _muestras=3),
    "muestreo5_p0.95": dict(do_sample=True, top_p=0.95, temperature=1.0, repetition_penalty=1.1,
                            _muestras=5),
    "beam4_ngram6_lp2": dict(num_beams=4, no_repeat_ngram_size=6, length_penalty=2.0),
}


def barrido(ruta, n=250):
    t0 = time.time()
    tok, m = cargar_modelo(ruta)
    sb = SentenceTransformer(SBERT, device="cuda")
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_pruned_val.parquet")
    d = d.groupby("mitad", group_keys=False).apply(lambda x: x.sample(min(n // 2, len(x)), random_state=SEED))
    d = d.reset_index(drop=True)
    refs = [list(x) for x in d.tareas]
    filas = []
    br = puntuar(sb, [base_recuperacion(s) for s in d.src], refs)
    filas.append(dict(config="solo_recuperacion",
                      **{k: round(v, 4) for k, v in br.items() if k not in ("por_ejemplo", "mat", "cols")}))
    print(f"  solo_recuperacion  sF1 {br['sF1']:.4f}  ratio {br['ratio']:.2f}", flush=True)
    for nombre, cfg in CFGS.items():
        t1 = time.time()
        gens = generar(tok, m, d, cfg)
        s = puntuar(sb, gens, refs)
        filas.append(dict(config=nombre,
                          **{k: round(v, 4) for k, v in s.items() if k not in ("por_ejemplo", "mat", "cols")},
                          seg=round(time.time() - t1)))
        print(f"  {nombre:<18} sF1 {s['sF1']:.4f}  hF1 {s['hF1']:.4f}  rouge1 {s['rouge1']:.4f}  "
              f"ratio {s['ratio']:.2f}  ({time.time()-t1:.0f}s)", flush=True)
    R = pd.DataFrame(filas).sort_values("sF1", ascending=False)
    tag = ruta.rstrip("/").split("/")[-1]
    R.to_csv(f"{BASE}/results/11_barrido_{tag}.csv", index=False)
    print(R.to_string(index=False))
    print(f"mejor: {R.config.iat[0]}   total {(time.time()-t0)/60:.1f} min")


def test(rutas, cfg_nombre):
    t0 = time.time()
    sb = SentenceTransformer(SBERT, device="cuda")
    d = pd.read_parquet(os.environ.get("AGIX_TEST_PARQUET")
                       or f"{BASE}/data/processed/s2s_pruned_test.parquet").reset_index(drop=True)
    refs = [list(x) for x in d.tareas]
    res = {}
    br = puntuar(sb, [base_recuperacion(s) for s in d.src], refs)
    res["solo_recuperacion"] = br
    for ruta in rutas:
        tok, m = cargar_modelo(ruta)
        gens = generar(tok, m, d, CFGS[cfg_nombre])
        res[ruta.rstrip("/").split("/")[-1]] = puntuar(sb, gens, refs)
        pd.DataFrame(dict(parent_uid=d.parent_uid, mitad=d.mitad, gen=[" ; ".join(g) for g in gens],
                          ref=[" ; ".join(r) for r in refs])).to_parquet(
            f"{BASE}/results/11_gen_test_{ruta.rstrip('/').split('/')[-1]}.parquet", index=False)
        del m
        torch.cuda.empty_cache()
        print(f"  {ruta.split('/')[-1]}: sF1 {res[list(res)[-1]]['sF1']:.4f}", flush=True)

    rng = np.random.default_rng(SEED)
    filas = []
    for nombre, s in res.items():
        for mitad in ["todas", "tasks", "stories"]:
            m_ = np.ones(len(d), bool) if mitad == "todas" else (d.mitad == mitad).values
            filas.append(dict(modelo=nombre, mitad=mitad, n=int(m_.sum()),
                              **{c: s["mat"][m_, i].mean() for i, c in enumerate(s["cols"])}))
    R = pd.DataFrame(filas)
    # paired bootstrap: each model vs the retrieval baseline, and base vs small
    pares = [(a, "solo_recuperacion") for a in res if a != "solo_recuperacion"]
    if len(res) == 3:
        m1, m2 = [k for k in res if k != "solo_recuperacion"]
        pares.append((m2, m1))
    comp = []
    for a, b in pares:
        dd = res[a]["por_ejemplo"] - res[b]["por_ejemplo"]
        bs = dd[rng.integers(0, len(dd), (2000, len(dd)))].mean(1)
        comp.append(dict(a=a, b=b, delta_sF1=dd.mean(), ic95_lo=np.quantile(bs, .025),
                         ic95_hi=np.quantile(bs, .975)))
    C = pd.DataFrame(comp)
    print(R.round(4).to_string(index=False))
    print(C.round(4).to_string(index=False))
    R.round(4).to_csv(f"{BASE}/results/11_test_metricas.csv", index=False)
    C.round(4).to_csv(f"{BASE}/results/11_test_comparaciones.csv", index=False)
    print(f"total {(time.time()-t0)/60:.1f} min")


def dalpiaz(rutas, cfg_nombre):
    sb = SentenceTransformer(SBERT, device="cuda")
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_pruned_dalpiaz.parquet").reset_index(drop=True)
    refs = [list(x) for x in d.tareas]
    filas = []
    br = puntuar(sb, [base_recuperacion(s) for s in d.src], refs)
    filas.append(dict(modelo="solo_recuperacion",
                      **{k: round(v, 4) for k, v in br.items() if k not in ("por_ejemplo", "mat", "cols")}))
    guardar = {}
    for ruta in rutas:
        tok, m = cargar_modelo(ruta)
        gens = generar(tok, m, d, CFGS[cfg_nombre], bs=8)
        s = puntuar(sb, gens, refs)
        tag = ruta.rstrip("/").split("/")[-1]
        filas.append(dict(modelo=tag,
                          **{k: round(v, 4) for k, v in s.items() if k not in ("por_ejemplo", "mat", "cols")}))
        guardar[tag] = gens
        del m
        torch.cuda.empty_cache()
    R = pd.DataFrame(filas)
    print(R.to_string(index=False))
    R.to_csv(f"{BASE}/results/11_dalpiaz.csv", index=False)
    with open(f"{BASE}/results/11_dalpiaz_muestras.txt", "w", encoding="utf-8") as fh:
        for i in range(len(d)):
            fh.write(f"=== {d.project[i]} | {d.descripcion[i][:200]}\nREF: {' ; '.join(refs[i][:6])}\n")
            for tag, g in guardar.items():
                fh.write(f"{tag}: {' ; '.join(g[i])}\n")
            fh.write("\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "--barrido":
        barrido(a[1], int(a[2]) if len(a) > 2 else 250)
    elif a[0] == "--test":
        test(a[1:-1], a[-1])
    elif a[0] == "--dalpiaz":
        dalpiaz(a[1:-1], a[-1])
