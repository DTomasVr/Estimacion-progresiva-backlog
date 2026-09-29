"""E35 - Reajustar el codificador con un objetivo mejor que el error cuadratico.

El canal que mas pesa en el apilado final es el codificador ajustado de extremo a
extremo: 0,390 de peso, mas que la bolsa de siete arboles. Pero se ajusto con
error cuadratico sobre log-horas y una cabeza escalar, que es el objetivo mas
pobre disponible: obliga a la red a colapsar todo lo que sabe del texto en un
numero, y ademas en el numero equivocado (la media, cuando las metricas que
reportamos quieren la mediana y las ventanas).

Hay dos formas conocidas de darle mas senal al codificador con los mismos datos:

  CLASIFICACION  en vez de predecir un escalar, predecir la distribucion sobre la
                 rejilla de valores. Es "regresion por clasificacion": la red
                 recibe gradiente sobre la forma entera y no solo sobre la
                 posicion, y en la practica aprende representaciones mejores.

  ORDENACION     una perdida por pares. El Spearman del modelo es 0,42, o sea que
                 ordenar tareas por tamano es donde mas flojea, y una perdida de
                 ordenacion ataca eso directamente en vez de por accidente.

Tres brazos, mismo casco, mismos datos, mismas particiones:

  A  escalar con error cuadratico          lo que ya hay (models/e5_ajustado)
  B  cabeza de clasificacion sobre rejilla
  C  clasificacion + ordenacion por pares

Para aislar el codificador, despues de cada ajuste se extraen los vectores y se
ajusta EL MISMO HGB que usa el canal congelado, igual que hizo E10. Lo unico que
cambia entre lecturas es quien produce los vectores.
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
OUT = Path("results/finetune_objetivos")
MAXLEN, LOTE, EPOCAS, LR = 48, 24, 3, 1e-5
S = "=" * 122


def metricas(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    q = np.log(p / y)
    r = np.abs(p - y) / y
    m = float(np.median(q))
    return dict(MAE_h=float(np.abs(p - y).mean()),
                MdAElog=float(np.median(np.abs(q))),
                zeta_pc=float((np.exp(np.median(np.abs(q))) - 1) * 100),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100),
                R2log=float(1 - ((np.log(y) - np.log(p)) ** 2).mean() / np.log(y).var()),
                PRED25=float(np.mean(r <= .25)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                rho=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def main():
    import torch
    import torch.nn as nn
    from sklearn.ensemble import HistGradientBoostingRegressor
    from transformers import AutoModel, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = ("query: " + sw.title.fillna("").astype(str)).tolist()
    y = sw.effort_hours.to_numpy()
    ly = np.log(y).astype(np.float32)
    tr = np.flatnonzero((sw.split_t3 == "train").to_numpy())
    va = np.flatnonzero((sw.split_t3 == "val").to_numpy())
    bl = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
          "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
          "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    rej = np.array(json.load(open("results/tasa_acierto/meta.json", encoding="utf-8"))["rejilla"])
    lrej = np.log(rej)
    cl = np.abs(lrej[None, :] - ly[:, None]).argmin(1).astype(np.int64)
    K = len(rej)
    print(f"[{dev}] train {len(tr):,} · val {len(va):,} · {K} clases", flush=True)

    tk = AutoTokenizer.from_pretrained(BASE)
    lrej_t = torch.tensor(lrej, dtype=torch.float32, device=dev)

    class Red(nn.Module):
        def __init__(self, b, k, escalar):
            super().__init__()
            self.b = b
            h = b.config.hidden_size
            self.cab = nn.Sequential(nn.Dropout(.1), nn.Linear(h, 1 if escalar else k))

        def vector(self, ids, mask):
            o = self.b(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            return (o * m).sum(1) / m.sum(1).clamp(min=1e-9)

        def forward(self, ids, mask):
            return self.cab(self.vector(ids, mask))

    def lotes(idx):
        e = tk([tit[i] for i in idx], padding=True, truncation=True,
               max_length=MAXLEN, return_tensors="pt")
        return e["input_ids"].to(dev), e["attention_mask"].to(dev)

    def ajustar(modo):
        escalar = modo == "A"
        base = AutoModel.from_pretrained(BASE).to(dev)
        base.gradient_checkpointing_enable()
        net = Red(base, K, escalar).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=.01)
        pasos = (len(tr) // LOTE + 1) * EPOCAS
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=pasos, pct_start=.1)
        esc = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
        ce = nn.CrossEntropyLoss(label_smoothing=.05)
        mse = nn.MSELoss()

        @torch.no_grad()
        def inferir(idx):
            net.eval()
            V, P = [], []
            for i in range(0, len(idx), 96):
                ids, mask = lotes(idx[i:i + 96])
                with torch.amp.autocast("cuda", enabled=dev == "cuda"):
                    v = net.vector(ids, mask)
                    o = net.cab(v)
                V.append(v.float().cpu().numpy())
                P.append(o.float().cpu().numpy())
            net.train()
            return np.concatenate(V), np.concatenate(P)

        mejor, mejor_v = 1e9, None
        for ep in range(EPOCAS):
            t0 = time.time()
            perm = np.random.default_rng(ep).permutation(tr)
            for i in range(0, len(perm), LOTE):
                idx = perm[i:i + LOTE]
                ids, mask = lotes(idx)
                with torch.amp.autocast("cuda", enabled=dev == "cuda"):
                    o = net(ids, mask)
                    if escalar:
                        p = mse(o.squeeze(-1), torch.tensor(ly[idx], device=dev))
                    else:
                        p = ce(o, torch.tensor(cl[idx], device=dev))
                        if modo == "C":
                            # ordenacion por pares dentro del lote, solo pares
                            # separados de verdad (mas de medio log) para no
                            # aprender el ruido de las diferencias minimas
                            e = (torch.softmax(o, 1) * lrej_t).sum(1)
                            t = torch.tensor(ly[idx], device=dev)
                            d = t[:, None] - t[None, :]
                            g = (d.abs() > .5).float().triu(1)
                            if g.sum() > 0:
                                s = (e[:, None] - e[None, :]) * torch.sign(d)
                                p = p + .3 * ((torch.relu(.5 - s) * g).sum() / g.sum())
                opt.zero_grad()
                esc.scale(p).backward()
                esc.step(opt)
                esc.update()
                sch.step()
            _, ov = inferir(va)
            pv = ov.squeeze(-1) if escalar else (torch.softmax(torch.tensor(ov), 1).numpy() @ lrej)
            v = float(np.mean((ly[va] - pv) ** 2))
            print(f"    epoca {ep + 1}/{EPOCAS} · MSE val {v:.4f} · "
                  f"{time.time() - t0:.0f} s", flush=True)
            if v < mejor:
                mejor = v
                mejor_v = inferir(np.arange(len(sw)))[0]
        del net, base
        torch.cuda.empty_cache()
        return mejor_v

    def hgb():
        return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                             l2_regularization=1., random_state=0)

    vectores = {"A · escalar cuadratico (el actual)": np.load(CACHE / "e5-ajustado.npy")}
    for modo, etq in (("B", "B · cabeza de clasificacion"),
                      ("C", "C · clasificacion + ordenacion")):
        f = CACHE / f"e5-ajustado-{modo}.npy"
        if f.exists():
            vectores[etq] = np.load(f)
            print(f"  {etq} en cache")
            continue
        print(f"\n  ajustando {etq}", flush=True)
        V = ajustar(modo)
        np.save(f, V.astype(np.float32))
        vectores[etq] = V

    filas = []
    for etq, V in vectores.items():
        m = hgb().fit(V[tr], ly[tr])
        p = np.exp(m.predict(V))
        for b, k in bl.items():
            filas.append(dict(bloque=b, codificador=etq, n=int(k.sum()), **metricas(y[k], p[k])))
        print(f"  {etq} evaluado", flush=True)
    t = pd.DataFrame(filas)
    cols = ["codificador", "MAE_h", "MdAElog", "zeta_pc", "R2log", "PRED25", "factor2", "rho"]
    for b in bl:
        print(f"\n{S}\n{b}  (HGB identico sobre cada juego de vectores)\n{S}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
