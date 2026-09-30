"""E35b - El mismo ajuste fino que E35, pero configurado para la maquina que tenemos.

E35 heredo la configuracion de E10 sin revisarla y costo 64 minutos por epoca. El
brazo C se abandono a las 3 h 20 min. Aqui estan las cuatro cosas que estaban mal,
por orden de lo que cuestan:

  1 LA MATRIZ DE EMBEDDINGS ESTABA APRENDIENDO. multilingual-e5-large es XLM-R,
    con vocabulario de 250.000 piezas: la tabla de embeddings sola son ~256 de los
    ~560 millones de parametros, casi la mitad del modelo. Con AdamW eso son 256M
    x 12 bytes = 3,1 GB de VRAM en gradientes y estados del optimizador, y otro
    tanto de trabajo en cada paso, para reajustar vectores de piezas que en un
    corpus de titulos en ingles la mayoria ni aparecen. Congelarla libera memoria
    Y acelera el paso del optimizador.

  2 GRADIENT CHECKPOINTING ACTIVADO. Recalcula las activaciones en la marcha atras
    para ahorrar memoria, a cambio de un 30-40 % de velocidad. Tiene sentido con
    secuencias largas; aqui los titulos son de 48 piezas y las activaciones no son
    el problema. Se paga el peaje sin necesitarlo.

  3 SE TOKENIZABA EN CADA PASO. 693 pasos x 3 epocas x 2 brazos, tokenizando los
    mismos titulos una y otra vez en el hilo que alimenta a la GPU. Se hace una
    vez y se deja en memoria.

  4 LOTE DE 24 Y fp16 CON ESCALADOR. Con la memoria que liberan 1 y 2 caben lotes
    mucho mayores, y esta GPU admite bf16, que no necesita escalador.

Ademas, dos cosas que faltaban y que costaron caro:

  - IMPRIMIR PROGRESO DENTRO DE LA EPOCA. Sin eso no se puede saber si va bien ni
    estimar cuando acaba, que fue exactamente lo que paso.
  - GUARDAR LOS VECTORES EN CUANTO MEJORA. Si se corta, lo hecho no se pierde.

    python src/147_finetune_rapido.py banco      mide y compara configuraciones
    python src/147_finetune_rapido.py B C        ajusta esos brazos
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 205)
BASE = "intfloat/multilingual-e5-large"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/finetune_rapido")
MAXLEN, EPOCAS, LR = 48, 3, 1e-5
S = "=" * 116


def cargar():
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    return sw


def construir(dev, congelar_emb, checkpointing, K):
    import torch.nn as nn
    from transformers import AutoModel

    base = AutoModel.from_pretrained(BASE).to(dev)
    if checkpointing:
        base.gradient_checkpointing_enable()
    if congelar_emb:
        for p in base.embeddings.word_embeddings.parameters():
            p.requires_grad = False

    class Red(nn.Module):
        def __init__(self, b, k):
            super().__init__()
            self.b = b
            self.cab = nn.Sequential(nn.Dropout(.1), nn.Linear(b.config.hidden_size, k))

        def vector(self, ids, mask):
            o = self.b(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            return (o * m).sum(1) / m.sum(1).clamp(min=1e-9)

        def forward(self, ids, mask):
            return self.cab(self.vector(ids, mask))

    return Red(base, K).to(dev)


def tokenizar(textos, dev):
    import torch
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(BASE)
    e = tk(textos, padding="max_length", truncation=True, max_length=MAXLEN,
           return_tensors="pt")
    return e["input_ids"].to(dev), e["attention_mask"].to(dev)


def banco():
    """Mide pasos por segundo de cada configuracion antes de comprometerse."""
    import torch
    import torch.nn as nn
    dev = "cuda"
    sw = cargar()
    tit = ("query: " + sw.title.fillna("").astype(str)).tolist()
    IDS, MASK = tokenizar(tit, dev)
    tr = np.flatnonzero((sw.split_t3 == "train").to_numpy())
    obj = torch.tensor(np.random.default_rng(0).integers(0, 44, len(sw)), device=dev)
    ce = nn.CrossEntropyLoss()
    cfgs = [("E35: checkpointing, embeddings libres, lote 24, fp16", True, False, 24, "fp16"),
            ("sin checkpointing, lote 24, bf16", False, False, 24, "bf16"),
            ("+ embeddings congelados, lote 24, bf16", False, True, 24, "bf16"),
            ("+ lote 64", False, True, 64, "bf16"),
            ("+ lote 128", False, True, 128, "bf16")]
    filas = []
    for etq, chk, cong, lote, prec in cfgs:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            net = construir(dev, cong, chk, 44)
            opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                    lr=LR, weight_decay=.01)
            esc = torch.amp.GradScaler("cuda", enabled=prec == "fp16")
            dt = torch.float16 if prec == "fp16" else torch.bfloat16
            n_ent = sum(p.numel() for p in net.parameters() if p.requires_grad)
            rng = np.random.default_rng(0)
            for paso in range(15):                      # calentamiento y medida
                if paso == 5:
                    torch.cuda.synchronize()
                    t0 = time.time()
                j = rng.choice(tr, lote, replace=False)
                with torch.amp.autocast("cuda", dtype=dt):
                    p = ce(net(IDS[j], MASK[j]), obj[j])
                opt.zero_grad()
                esc.scale(p).backward()
                esc.step(opt)
                esc.update()
            torch.cuda.synchronize()
            seg = (time.time() - t0) / 10
            pasos_ep = int(np.ceil(len(tr) / lote))
            filas.append(dict(configuracion=etq, params_M=round(n_ent / 1e6),
                              lote=lote, s_por_paso=round(seg, 4),
                              min_por_epoca=round(seg * pasos_ep / 60, 1),
                              VRAM_GB=round(torch.cuda.max_memory_allocated() / 1e9, 2)))
            print(f"  {etq:52s} {seg * pasos_ep / 60:5.1f} min/epoca · "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
            del net, opt
        except RuntimeError as e:
            print(f"  {etq:52s} FALLO: {str(e)[:60]}", flush=True)
            filas.append(dict(configuracion=etq, lote=lote, s_por_paso=np.nan,
                              min_por_epoca=np.nan, VRAM_GB=np.nan))
        torch.cuda.empty_cache()
    t = pd.DataFrame(filas)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "banco.csv", index=False)
    print(f"\n{S}\nQUE CUESTA CADA COSA\n{S}")
    print(t.to_string(index=False))
    b = t.dropna(subset=["min_por_epoca"])
    if len(b) > 1:
        v, m = b.min_por_epoca.iloc[0], b.min_por_epoca.min()
        print(f"\n  de {v:.1f} a {m:.1f} min por epoca -> x{v / m:.1f} mas rapido")
        print(f"  las tres epocas de un brazo pasan de {3 * v / 60:.1f} h a {3 * m / 60:.1f} h")
    print(f"\nescrito {OUT}/banco.csv")


def ajustar(modo, lote, dev="cuda"):
    import torch
    import torch.nn as nn
    OUT.mkdir(parents=True, exist_ok=True)
    sw = cargar()
    tit = ("query: " + sw.title.fillna("").astype(str)).tolist()
    ly = np.log(sw.effort_hours.to_numpy()).astype(np.float32)
    tr = np.flatnonzero((sw.split_t3 == "train").to_numpy())
    va = np.flatnonzero((sw.split_t3 == "val").to_numpy())
    rej = np.array(json.load(open("results/tasa_acierto/meta.json",
                                  encoding="utf-8"))["rejilla"])
    lrej = np.log(rej)
    cl = torch.tensor(np.abs(lrej[None, :] - ly[:, None]).argmin(1), device=dev)
    lrej_t = torch.tensor(lrej, dtype=torch.float32, device=dev)
    IDS, MASK = tokenizar(tit, dev)
    net = construir(dev, True, False, len(rej))
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=LR, weight_decay=.01)
    pasos_ep = int(np.ceil(len(tr) / lote))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                              total_steps=pasos_ep * EPOCAS, pct_start=.1)
    ce = nn.CrossEntropyLoss(label_smoothing=.05)
    print(f"[{dev}] brazo {modo} · lote {lote} · {pasos_ep} pasos por epoca · "
          f"{sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6:.0f}M "
          f"parametros entrenables", flush=True)

    @torch.no_grad()
    def inferir(idx):
        net.eval()
        V, P = [], []
        for i in range(0, len(idx), 256):
            j = idx[i:i + 256]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                v = net.vector(IDS[j], MASK[j])
                o = net.cab(v)
            V.append(v.float().cpu().numpy())
            P.append(torch.softmax(o.float(), 1).cpu().numpy())
        net.train()
        return np.concatenate(V), np.concatenate(P)

    mejor = 1e9
    for ep in range(EPOCAS):
        t0 = time.time()
        perm = np.random.default_rng(ep).permutation(tr)
        for k, i in enumerate(range(0, len(perm), lote)):
            j = torch.tensor(perm[i:i + lote], device=dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o = net(IDS[j], MASK[j])
                p = ce(o, cl[j])
                if modo == "C":
                    e = (torch.softmax(o.float(), 1) * lrej_t).sum(1)
                    t = torch.tensor(ly[perm[i:i + lote]], device=dev)
                    d = t[:, None] - t[None, :]
                    g = (d.abs() > .5).float().triu(1)
                    if g.sum() > 0:
                        s = (e[:, None] - e[None, :]) * torch.sign(d)
                        p = p + .3 * ((torch.relu(.5 - s) * g).sum() / g.sum())
            opt.zero_grad()
            p.backward()
            opt.step()
            sch.step()
            if k and k % 40 == 0:
                hecho = (k + 1) / pasos_ep
                print(f"    ep{ep + 1} paso {k}/{pasos_ep} ({hecho:.0%}) · "
                      f"{time.time() - t0:.0f} s · faltan "
                      f"{(time.time() - t0) * (1 / hecho - 1) / 60:.0f} min", flush=True)
        V, P = inferir(np.arange(len(sw)))
        v = float(np.mean((ly[va] - (P[va] @ lrej)) ** 2))
        print(f"  epoca {ep + 1}/{EPOCAS} · MSE val {v:.4f} · "
              f"{time.time() - t0:.0f} s", flush=True)
        if v < mejor:                           # se guarda EN CUANTO mejora
            mejor = v
            np.save(CACHE / f"e5-ajustado-{modo}.npy", V.astype(np.float32))
            print(f"    guardado e5-ajustado-{modo}.npy", flush=True)
    del net, opt
    torch.cuda.empty_cache()


if __name__ == "__main__":
    if "banco" in sys.argv:
        banco()
    else:
        lote = int(next((a.split("=")[1] for a in sys.argv if a.startswith("lote=")), 64))
        for m in [a for a in sys.argv[1:] if a in ("B", "C")]:
            ajustar(m, lote)
