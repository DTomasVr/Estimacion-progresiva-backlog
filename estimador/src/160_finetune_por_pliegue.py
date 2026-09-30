"""E54 - Reajustar el codificador POR PLIEGUE, para que la metrica de proyecto pueda usarlo.

El brazo C de E35 aporta (R2 del apilado 0,199 -> 0,204, Spearman 0,416 -> 0,422,
MAE 4,919 -> 4,904 en la prueba fija), pero NO se puede meter en la metrica de
proyecto tal cual. El codificador se ajusto sobre split_t3 == "train" de todos
los proyectos, asi que ya vio las etiquetas de las tareas de los proyectos que
cada pliegue deja fuera. Esa es exactamente la fuga que se detecto en E42, donde
daba un R2 de 0,398 en proyectos "nuevos", mejor que en la prueba fija.

La forma honesta es ajustar un codificador por pliegue, cada uno viendo solo los
proyectos de entrenamiento de ese pliegue. Son cinco ajustes de tres epocas. Con
la configuracion medida (embeddings congelados, recorte dinamico, lote 16,
bf16) cada epoca cuesta unos cinco minutos, asi que el total ronda los 75-90
minutos, que es asumible y deja la cifra de proyecto sin asteriscos.

Se guardan los vectores de cada pliegue en cuanto ese pliegue termina, para que
un corte no tire lo hecho.
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
BASE = "intfloat/multilingual-e5-large"
CACHE = Path("data/processed/embeddings/codificadores/pliegues")
MAXLEN, EPOCAS, LR, LOTE = 48, 3, 1e-5, 16
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

    tk = AutoTokenizer.from_pretrained(BASE)
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

    V = np.zeros((len(sw), 1024), np.float32)
    t0 = time.time()
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(np.zeros(len(sw)), y, proy)):
        f = CACHE / f"e5C_pliegue{nf}.npy"
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
            out = np.zeros((len(idx), 1024), np.float32)
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
    np.save("data/processed/embeddings/codificadores/e5-ajustado-C-porpliegue.npy", V)
    print(f"escrito e5-ajustado-C-porpliegue.npy · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
