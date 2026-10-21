"""E120a - El brazo C por pliegue, con multilingual-e5-SMALL en vez de large.

Copia de 160_finetune_por_pliegue.py con lo minimo cambiado, para que la
comparacion large/small sea con la MISMA receta: mismo objetivo (entropia sobre
la rejilla + bisagra de orden), embeddings de palabra congelados, validacion
interna por proyectos, mejor epoca, un ajuste por pliegue de GroupKFold.

Cambios: el codificador (118 M parametros, 384 dimensiones, contra 560 M y 1024)
y la tasa de aprendizaje, 1e-5 -> 3e-5, la habitual para un modelo pequeno.
Ademas se guarda el codificador base sin ajustar, para el brazo en frio.
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
BASE = "intfloat/multilingual-e5-small"
CACHE = Path("data/processed/embeddings/codificadores/pliegues")
MAXLEN, EPOCAS, LR, LOTE = 48, 3, 3e-5, 16
PLIEGUES = 5


def main():
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupKFold
    from transformers import AutoModel, AutoTokenizer

    CACHE.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = ("query: " + sw.title.fillna("").astype(str)).tolist()
    y = sw.effort_hours.to_numpy()
    ly = np.log(y).astype(np.float32)
    proy = sw.project.to_numpy()
    rej = np.array(json.load(open("results/tasa_acierto/meta.json",
                                  encoding="utf-8"))["rejilla"])
    lrej = np.log(rej)
    K = len(rej)

    DIM = 384
    tk = AutoTokenizer.from_pretrained(BASE)
    fb = Path("data/processed/embeddings/codificadores/e5-small.npy")
    if not fb.exists():
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(BASE, device=dev)
        np.save(fb, st.encode(tit, batch_size=256, normalize_embeddings=False,
                              show_progress_bar=False).astype(np.float32))
        del st
        torch.cuda.empty_cache()
        print("  e5-small base codificado", flush=True)
    e = tk(tit, padding="max_length", truncation=True, max_length=MAXLEN,
           return_tensors="pt")
    IDS, MASK = e["input_ids"].to(dev), e["attention_mask"].to(dev)
    cl = torch.tensor(np.abs(lrej[None, :] - ly[:, None]).argmin(1).astype(np.int64),
                      device=dev)
    ly_t = torch.tensor(ly, device=dev)
    lrej_t = torch.tensor(lrej, dtype=torch.float32, device=dev)
    largo = MASK.sum(1).cpu().numpy()
    print(f"{len(sw):,} tareas · {K} clases · {PLIEGUES} pliegues", flush=True)

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

    def recorte(j):
        L = int(MASK[j].sum(1).max())
        return IDS[j][:, :L], MASK[j][:, :L]

    V = np.zeros((len(sw), DIM), np.float32)
    t0 = time.time()
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(np.zeros(len(sw)), y, proy)):
        f = CACHE / f"e5sC_pliegue{nf}.npy"
        if f.exists():
            V[b] = np.load(f)
            print(f"  pliegue {nf + 1}/{PLIEGUES} en cache", flush=True)
            continue
        # una parte de los proyectos de entrenamiento hace de validacion interna
        rng = np.random.default_rng(nf)
        pr = np.unique(proy[a])
        pin = set(rng.choice(pr, max(2, int(.15 * len(pr))), replace=False))
        ain = a[np.array([p in pin for p in proy[a]])]
        afu = a[np.array([p not in pin for p in proy[a]])]
        base = AutoModel.from_pretrained(BASE).to(dev)
        for p in base.embeddings.word_embeddings.parameters():
            p.requires_grad = False
        net = Red(base, K).to(dev)
        opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                lr=LR, weight_decay=.01)
        pasos = int(np.ceil(len(afu) / LOTE))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR,
                                                  total_steps=pasos * EPOCAS, pct_start=.1)
        ce = nn.CrossEntropyLoss(label_smoothing=.05)

        @torch.no_grad()
        def infer(idx):
            net.eval()
            o = idx[np.argsort(largo[idx], kind="stable")]
            pos = {t: k for k, t in enumerate(idx)}
            dest = np.array([pos[t] for t in o])
            out = np.zeros((len(idx), DIM), np.float32)
            pr_ = np.zeros((len(idx), K), np.float32)
            for i in range(0, len(o), 192):
                j = torch.tensor(o[i:i + 192], device=dev)
                ids, mask = recorte(j)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    v = net.vector(ids, mask)
                    z = net.cab(v)
                out[dest[i:i + 192]] = v.float().cpu().numpy()
                pr_[dest[i:i + 192]] = torch.softmax(z.float(), 1).cpu().numpy()
            net.train()
            torch.cuda.empty_cache()
            return out, pr_

        mejor, mejor_v = 1e9, None
        for ep in range(EPOCAS):
            te = time.time()
            perm = torch.tensor(rng.permutation(afu), device=dev)
            for i in range(0, len(perm), LOTE):
                j = perm[i:i + LOTE]
                ids, mask = recorte(j)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    z = net(ids, mask)
                    p = ce(z, cl[j])
                    s = (torch.softmax(z.float(), 1) * lrej_t).sum(1)
                    t = ly_t[j]
                    dd = t[:, None] - t[None, :]
                    gg = (dd.abs() > .5).float().triu(1)
                    if gg.sum() > 0:
                        sc = (s[:, None] - s[None, :]) * torch.sign(dd)
                        p = p + .3 * ((torch.relu(.5 - sc) * gg).sum() / gg.sum())
                opt.zero_grad()
                p.backward()
                opt.step()
                sch.step()
            _, pin_ = infer(ain)
            v = float(np.mean((ly[ain] - (pin_ @ lrej)) ** 2))
            print(f"  pliegue {nf + 1} epoca {ep + 1}/{EPOCAS} · MSE val {v:.4f} · "
                  f"{time.time() - te:.0f} s · total {time.time() - t0:.0f} s", flush=True)
            if v < mejor:
                mejor = v
                mejor_v = infer(b)[0]
        np.save(f, mejor_v)
        V[b] = mejor_v
        del net, base, opt
        torch.cuda.empty_cache()
        print(f"  pliegue {nf + 1}/{PLIEGUES} guardado · {time.time() - t0:.0f} s",
              flush=True)
    np.save("data/processed/embeddings/codificadores/e5-small-C-porpliegue.npy", V)
    print(f"escrito e5-small-C-porpliegue.npy · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
