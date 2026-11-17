"""mT5 One2Seq generator: "decompose story into tasks" / "decompose feature
into user stories", conditioned on retrieved examples (see 08_preprocess.py).

fp32 master weights + bf16 autocast (pure-bf16 weights with Adafactor lose
small updates to rounding). Pre-tokenized data, dynamic padding, length-bucketed
batches so padding waste is minimal. Validation loss every epoch, early
stopping, best checkpoint kept.

modes:
  --probe  MODEL                 VRAM / speed grid over batch x output length
  --mini   MODEL BS MAXOUT LR    200 examples, 30 epochs: must overfit
  (full)   MODEL BS MAXOUT LR [EPOCHS]
"""
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from transformers.optimization import Adafactor

BASE = r"F:\Software Task generation Model"
CACHE = f"{BASE}/models/hf_cache"
SEED = 20260922
VRAM_MAX = 6.5
torch.manual_seed(SEED)


def cargar(split, max_out, eos=1, podado=False):
    # DATOS=hist -> the story-aware set built by 15_datos_historias.py
    d_env = os.environ.get("DATOS")
    # every DATOS value must be listed here: an unknown one silently trains on
    # the lower-half data instead (cost me a 20 min run)
    conocidos = {"hist": "hist_", "o2o": "o2o_", "o2ofmt": "o2ofmt_",
                 "o2oes": "o2oes_", "o2oesfmt": "o2oesfmt_",
                 "mix": "mix_", "mixa": "mixa_",      # destilacion: real + sintetico
                 "v3": "v3_", "v3a": "v3a_", "v3r": "v3r_", "v3b": "v3b_", "v5": "v5_", "v7": "v7_", "v8": "v8_",
                 "v5nb": "v5nb_",
                 "v6": "v6_", "v6r": "v6r_",
                 "v8f": "v8f_", "v8fr": "v8fr_", "v8ff": "v8ff_",
                 # 88_retrotraducir_datos.py: v8ff + la descripcion pasada
                 # por EN->ES->EN con los Marian del producto (aumento de entrada)
                 "v9bt": "v9bt_",
                 # 89_dedup_historias_profesor.py: v8ff con las historias
                 # casi-duplicadas del profesor fuera (curacion, 3,5 %)
                 "v9df": "v9df_"}          # tanda v3: roles y coherencia
    if d_env and d_env not in conocidos:
        raise SystemExit(f"DATOS={d_env} desconocido; usa {list(conocidos)}")
    pre = conocidos.get(d_env, "pruned_" if podado else "")
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_{pre}{split}.parquet")
    labs = []
    for l in d["labels"]:
        l = list(l)
        labs.append(l if len(l) <= max_out else l[:max_out - 1] + [eos])
    d["labels"] = labs
    return d


def lotes(d, bs, rng, barajar=True):
    """length-bucketed batches: sort within chunks of 50 batches, shuffle batch order"""
    idx = rng.permutation(len(d)) if barajar else np.arange(len(d))
    L = np.array([len(d["input_ids"].iat[i]) + len(d["labels"].iat[i]) for i in idx])
    out = []
    trozo = bs * 50
    for a in range(0, len(idx), trozo):
        sub = idx[a:a + trozo][np.argsort(-L[a:a + trozo], kind="stable")]
        out += [sub[b:b + bs] for b in range(0, len(sub), bs)]
    if barajar:
        rng.shuffle(out)
    return out


def colar(d, ii, dev="cuda"):
    x = [list(d["input_ids"].iat[i]) for i in ii]
    y = [list(d["labels"].iat[i]) for i in ii]
    mx, my = max(map(len, x)), max(map(len, y))
    X = torch.zeros(len(ii), mx, dtype=torch.long)
    A = torch.zeros(len(ii), mx, dtype=torch.long)
    Y = torch.full((len(ii), my), -100, dtype=torch.long)
    for k, (a, b) in enumerate(zip(x, y)):
        X[k, :len(a)] = torch.tensor(a)
        A[k, :len(a)] = 1
        Y[k, :len(b)] = torch.tensor(b)
    return X.to(dev, non_blocking=True), A.to(dev, non_blocking=True), Y.to(dev, non_blocking=True)


def modelo(nombre):
    m = MT5ForConditionalGeneration.from_pretrained(nombre, cache_dir=CACHE, dtype=torch.float32).cuda()
    m.config.use_cache = False
    return m


def paso(m, opt, X, A, Y):
    l = micro(m, X, A, Y, 1)
    aplicar(m, opt)
    return l


def micro(m, X, A, Y, acum):
    """solo adelante y atras: el gradiente se acumula en los parametros.

    La perdida se divide entre acum para que el gradiente acumulado sea el mismo
    que el de un lote acum veces mayor (la media, no la suma). Hace falta porque
    el modelo sin podar solo cabe a lote 8 y el podado se entreno a 16: sin
    igualar el lote EFECTIVO la comparacion mediria el lote, no la poda.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(input_ids=X, attention_mask=A, labels=Y).loss
    (loss / acum).backward()
    return loss.item()


def aplicar(m, opt):
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)


@torch.no_grad()
def val_loss(m, d, bs):
    m.eval()
    tot, n = 0.0, 0
    # same batch size as training: with a 75k vocab the logits alone are ~1.8 GB at bs 32
    for ii in lotes(d, bs, np.random.default_rng(0), barajar=False):
        X, A, Y = colar(d, ii)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = m(input_ids=X, attention_mask=A, labels=Y).loss
        k = int((Y != -100).sum())
        tot += l.item() * k
        n += k
    m.train()
    return tot / n


# ------------------------------------------------------------------- probe
def probe(nombre):
    m = modelo(nombre)
    opt = Adafactor(m.parameters(), lr=1e-3, relative_step=False, scale_parameter=False)
    res = {}
    for L_out in (128, 192, 256):
        for bs in (4, 8, 12, 16, 24):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            x = torch.randint(5, 30000, (bs, 256), device="cuda")
            a = torch.ones_like(x)
            y = torch.randint(5, 30000, (bs, L_out), device="cuda")
            paso(m, opt, x, a, y)
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(3):
                paso(m, opt, x, a, y)
            torch.cuda.synchronize()
            dt = (time.time() - t) / 3
            pk = torch.cuda.max_memory_allocated() / 2**30
            ok = pk <= VRAM_MAX
            res[f"out{L_out}_bs{bs}"] = dict(s_paso=round(dt, 3), ej_s=round(bs / dt, 1), pico_GB=round(pk, 2), ok=ok)
            print(f"  in256 out{L_out:<3} bs{bs:>3}: {dt:6.3f} s/paso {bs/dt:6.1f} ej/s  pico {pk:5.2f} GB "
                  f"{'' if ok else '<- supera 6.5 GB'}", flush=True)
            if not ok:
                break
    tag = nombre.split("/")[-1]
    json.dump(res, open(f"{BASE}/results/09_probe_{tag}.json", "w"), indent=1)


# ------------------------------------------------------------------- train
def entrenar(nombre, bs, max_out, lr, epocas, mini, frac=None, acum=1):
    t0 = time.time()
    tag = nombre.rstrip("/").split("/")[-1] + (f"_{os.environ['DATOS']}" if os.environ.get("DATOS") else "") + ("_mini" if mini else "") + (f"_sub{frac}_lr{lr}" if frac else "")
    out_dir = f"{BASE}/models/{tag}_agix_v1"
    tok = AutoTokenizer.from_pretrained(nombre, cache_dir=CACHE)
    podado = "pruned" in nombre
    tr = cargar("train", max_out, podado=podado)
    va = cargar("val", max_out, podado=podado)
    if frac:
        tr = tr.sample(frac=frac, random_state=SEED).reset_index(drop=True)
    if mini:
        tr = tr.sample(200, random_state=SEED).reset_index(drop=True)
        va = tr.copy()                           # gate: can it fit what it sees?
    m = modelo(nombre)
    opt = Adafactor(m.parameters(), lr=lr, relative_step=False, scale_parameter=False, warmup_init=False)
    micro_ep = math.ceil(len(tr) / bs)
    pasos_ep = math.ceil(micro_ep / acum)      # pasos del optimizador, no microlotes
    total = pasos_ep * epocas
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    rng = np.random.default_rng(SEED)
    print(f"{tag}: train {len(tr):,}  val {len(va):,}  bs {bs}x{acum} (efectivo {bs*acum})  "
          f"max_out {max_out}  lr {lr}  {epocas} epocas x {pasos_ep} pasos", flush=True)
    v0 = val_loss(m, va, bs)
    print(f"  val loss inicial {v0:.4f}", flush=True)
    hist, mejor, sin_mejora, paciencia = [], float("inf"), 0, 2
    torch.cuda.reset_peak_memory_stats()
    s = 0
    for ep in range(1, epocas + 1):
        te = time.time()
        acc = []
        micro_lotes = lotes(tr, bs, rng)
        for j, ii in enumerate(micro_lotes, 1):
            X, A, Y = colar(tr, ii)
            acc.append(micro(m, X, A, Y, acum))
            if j % acum and j < len(micro_lotes):
                continue                       # sigue acumulando
            aplicar(m, opt)
            sched.step()
            s += 1
            if not mini and s % 100 == 0:
                dt = time.time() - te
                done = s - (ep - 1) * pasos_ep
                print(f"    paso {s:>6}/{total}  loss {np.mean(acc[-100:]):.4f}  "
                      f"{done/dt:4.2f} pasos/s  ETA epoca {(pasos_ep-done)/(done/dt)/60:4.1f} min  "
                      f"VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GB", flush=True)
        v = val_loss(m, va, bs)
        dt = time.time() - te
        hist.append(dict(epoca=ep, train_loss=float(np.mean(acc)), val_loss=v, seg=round(dt)))
        restante = (epocas - ep) * dt
        print(f"  epoca {ep:>2}: train {np.mean(acc):.4f}  val {v:.4f}  {dt/60:4.1f} min  "
              f"(tope restante {restante/60:5.1f} min)", flush=True)
        if v < mejor - 1e-4:
            mejor, sin_mejora = v, 0
            if not mini:      # --sub tambien guarda: el tag lleva la fraccion
                m.save_pretrained(out_dir)
                tok.save_pretrained(out_dir)
        else:
            sin_mejora += 1
            if not mini and sin_mejora >= paciencia:
                print("  parada temprana", flush=True)
                break

    # sanity generations
    m.eval()
    muestra = va.sample(min(5, len(va)), random_state=1)
    X, A, _ = colar(muestra, range(len(muestra)))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        g = m.generate(input_ids=X, attention_mask=A, max_new_tokens=max_out, num_beams=4,
                       no_repeat_ngram_size=4, repetition_penalty=1.3)
    gens = tok.batch_decode(g, skip_special_tokens=True)
    for (_, r), gen in zip(muestra.iterrows(), gens):
        # One2One data has one `objetivo` per row instead of a `tareas` list
        ref = (" ; ".join(r.tareas) if "tareas" in muestra.columns
               else str(getattr(r, "objetivo", "")))
        print(f"\n  SRC: {r.src[:160]}\n  REF: {ref[:300]}\n  GEN: {gen[:300]}")
    json.dump(dict(modelo=nombre, bs=bs, acum=acum, bs_efectivo=bs * acum, max_out=max_out,
                   lr=lr, epocas=epocas, hist=hist, v0=v0,
                   mejor_val=mejor, vram_pico=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   min=round((time.time() - t0) / 60, 1), gens=gens),
              open(f"{BASE}/results/09_train_{tag}.json", "w"), indent=1)
    print(f"\nTOTAL {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "--probe":
        probe(a[1])
    elif a[0] == "--sub":
        entrenar(a[1], int(a[2]), int(a[3]), float(a[4]), 1, False, float(a[5]),
                 acum=int(os.environ.get("ACUM", 1)))
    elif a[0] == "--mini":
        entrenar(a[1], int(a[2]), int(a[3]), float(a[4]), 30, True,
                 acum=int(os.environ.get("ACUM", 1)))
    else:
        entrenar(a[0], int(a[1]), int(a[2]), float(a[3]), int(a[4]) if len(a) > 4 else 12, False,
                 acum=int(os.environ.get("ACUM", 1)))
