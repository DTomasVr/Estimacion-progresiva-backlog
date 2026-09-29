"""E10 · Ajustar e5-large de extremo a extremo sobre el corpus limpio.

El unico ajuste fino que se hizo en el proyecto fue sobre mpnet, con el corpus
entero dominado por Apache y la vista de titulo + descripcion, y no gano nada
(0,686 frente a 0,685). Desde entonces han cambiado las tres cosas: el
codificador (e5-large da R² 0,149 congelado frente a 0,113 de mpnet), el corpus
(sin Apache) y la vista (solo titulos, de 7-8 palabras).

Los titulos cortos ayudan mucho aqui: con max_len 48 caben lotes grandes y el
ajuste de un modelo de 560M es viable en la GPU de 8 GB.

Esta vez **se guardan los pesos**, que en el ajuste anterior no se guardaron y
el modelo quedo inservible para codificar texto nuevo.

Para que la comparacion aisle al codificador y no a la cabeza, despues del
ajuste se extraen los vectores y se vuelve a ajustar EL MISMO HGB que usa el
canal congelado. Asi lo unico que cambia entre las dos lecturas es quien produce
los vectores.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 205)
DATA = "data/processed/agix_effort_v4.parquet"
BASE = "intfloat/multilingual-e5-large"
SALIDA = Path("models/e5_ajustado")
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/finetune_e5")
MAXLEN, LOTE, EPOCAS, LR = 48, 24, 3, 1e-5
SEP = "=" * 112


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
                PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
                sesgo=float(np.exp(np.median(lp - ly))),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def main():
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    SALIDA.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    tit = ("query: " + sw.title.fillna("").astype(str)).tolist()
    y = sw.effort_hours.to_numpy()
    ly = np.log(y).astype(np.float32)
    tr = np.flatnonzero((sw.split_t3 == "train").to_numpy())
    va = np.flatnonzero((sw.split_t3 == "val").to_numpy())
    print(f"[{dev}] ajuste de {BASE} · train {len(tr):,} · val {len(va):,}")
    if dev == "cuda":
        print(f"  {torch.cuda.get_device_name(0)} · "
              f"{torch.cuda.mem_get_info()[0] / 1e9:.1f} GB libres")

    tk = AutoTokenizer.from_pretrained(BASE)
    base = AutoModel.from_pretrained(BASE).to(dev)
    base.gradient_checkpointing_enable()

    class Regresor(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b
            self.cabeza = nn.Sequential(nn.Dropout(0.1), nn.Linear(b.config.hidden_size, 1))

        def vector(self, ids, mask):
            o = self.b(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            return (o * m).sum(1) / m.sum(1).clamp(min=1e-9)

        def forward(self, ids, mask):
            return self.cabeza(self.vector(ids, mask)).squeeze(-1)

    net = Regresor(base).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    pasos = (len(tr) // LOTE + 1) * EPOCAS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=pasos,
                                                pct_start=0.1)
    escala = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    perdida = nn.MSELoss()

    def lote_tensores(idx):
        e = tk([tit[i] for i in idx], padding=True, truncation=True,
               max_length=MAXLEN, return_tensors="pt")
        return e["input_ids"].to(dev), e["attention_mask"].to(dev)

    @torch.no_grad()
    def inferir(idx, vectores=False):
        net.eval()
        salida = []
        for i in range(0, len(idx), 128):
            ids, mask = lote_tensores(idx[i:i + 128])
            with torch.amp.autocast("cuda", enabled=dev == "cuda"):
                v = net.vector(ids, mask) if vectores else net(ids, mask)
            salida.append(v.float().cpu().numpy())
        net.train()
        return np.concatenate(salida)

    mejor, mejor_ep = 1e9, -1
    for ep in range(EPOCAS):
        t0 = time.time()
        perm = np.random.default_rng(ep).permutation(tr)
        acum = 0.0
        for k, i in enumerate(range(0, len(perm), LOTE)):
            idx = perm[i:i + LOTE]
            ids, mask = lote_tensores(idx)
            obj = torch.tensor(ly[idx], device=dev)
            with torch.amp.autocast("cuda", enabled=dev == "cuda"):
                p = perdida(net(ids, mask), obj)
            opt.zero_grad(set_to_none=True)
            escala.scale(p).backward()
            escala.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            escala.step(opt)
            escala.update()
            sched.step()
            acum += float(p)
            if k % 150 == 0:
                print(f"    ep{ep + 1} paso {k:>4}/{len(perm) // LOTE} "
                      f"pérdida {acum / (k + 1):.4f} · {time.time() - t0:.0f} s", flush=True)
        v = float(np.mean(np.abs(ly[va] - inferir(va))))
        print(f"  época {ep + 1}/{EPOCAS} · {time.time() - t0:.0f} s · "
              f"MAE log val {v:.4f}", flush=True)
        if v < mejor:
            mejor, mejor_ep = v, ep + 1
            net.b.save_pretrained(SALIDA)
            tk.save_pretrained(SALIDA)
            torch.save(net.cabeza.state_dict(), SALIDA / "cabeza.pt")
            print(f"    guardado en {SALIDA}", flush=True)

    # vectores del codificador ajustado, y el MISMO HGB encima
    print(f"\nmejor época {mejor_ep} (MAE log {mejor:.4f}); extrayendo vectores")
    from transformers import AutoModel as AM
    net.b = AM.from_pretrained(SALIDA).to(dev)
    V = inferir(np.arange(len(sw)), vectores=True).astype(np.float32)
    np.save(CACHE / "e5-ajustado.npy", V)

    from sklearn.ensemble import HistGradientBoostingRegressor
    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "prueba TAWOS": ((sw.split_t3 == "test") & (sw.source == "TAWOS")).to_numpy(),
               "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    tr_m = (sw.split_t3 == "train").to_numpy()
    filas = []
    for etq, X in (("e5-large congelado", np.load(CACHE / "e5-large.npy")),
                   ("e5-large ajustado", V)):
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                          l2_regularization=1., random_state=0).fit(
            X[tr_m], ly[tr_m])
        for b, k in bloques.items():
            filas.append(dict(codificador=etq, bloque=b, n=int(k.sum()),
                              **metricas(y[k], np.exp(m.predict(X[k])))))
    # y la cabeza del propio ajuste, sin HGB
    lp = inferir(np.arange(len(sw)))
    for b, k in bloques.items():
        filas.append(dict(codificador="cabeza del ajuste (sin HGB)", bloque=b,
                          n=int(k.sum()), **metricas(y[k], np.exp(lp[k]))))
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nAJUSTE FINO DE e5-large\n{SEP}")
    for b in bloques:
        print(f"\n{b}")
        print(t[t.bloque == b].drop(columns=["bloque"]).round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"base": BASE, "maxlen": MAXLEN, "lote": LOTE, "epocas": EPOCAS, "lr": LR,
               "mejor_epoca": mejor_ep, "mae_log_val": mejor},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv y pesos en {SALIDA}")


if __name__ == "__main__":
    main()
