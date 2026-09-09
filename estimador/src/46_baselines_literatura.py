"""Reimplementations of the published baselines, on our data and our splits.

Deep-SE (Choetkiertikul et al.) -- word embeddings learned from the training
text, an LSTM over the token sequence, dropout, linear output, L1 loss. The
published model predicts story points; here the target is log-hours, which is
the only honest adaptation since our data has no story points for most rows.

GPT2SP (Fu & Tantithamthavorn) -- GPT-2 fine-tuned end to end with a regression
head over the pooled final hidden states.

Both train on the pooled `split_t3 == train` block, early-stop on `val`, and
are read on `test` once. Our own ridge is refit in the same pooled regime so
the comparison is like for like, and the per-project median baseline is the
reference the literature scores against.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

pd.set_option("display.width", 250)
DATA = "data/processed/agix_effort_v3.parquet"
MAXLEN, BATCH = 100, 32


# --------------------------------------------------------------------------
def cargar():
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    return (sw[sw.split_t3 == "train"], sw[sw.split_t3 == "val"],
            sw[sw.split_t3 == "test"])


def metricas(y, p, etiqueta, tr, te):
    m = E.point_metrics(y, p)
    porproj = []
    for proj in te.project.unique():
        s = te.project == proj
        if s.sum() < 12:
            continue
        base = float(tr[tr.project == proj].effort_hours.median())
        if not np.isfinite(base):
            continue
        mb = E.point_metrics(y[s.to_numpy()], np.full(int(s.sum()), base))
        mm = E.point_metrics(y[s.to_numpy()], p[s.to_numpy()])
        porproj.append(mm["MdMRE"] < mb["MdMRE"])
    return dict(modelo=etiqueta, MdMRE=round(m["MdMRE"], 4),
                PRED25=round(m["PRED25"], 4), MAE_h=round(m["MAE_h"], 2),
                RMSE_log=round(m["RMSE_log"], 4),
                spearman=round(float(pd.Series(y).corr(pd.Series(p),
                                                       method="spearman")), 4),
                gana_base=f"{sum(porproj)}/{len(porproj)}")


# --------------------------------------------------------------------------
def deep_se(tr, va, te, dev):
    import torch
    import torch.nn as nn
    import re
    from collections import Counter

    tok = lambda s: re.findall(r"[a-z0-9<>]+", str(s).lower())[:MAXLEN]
    cnt = Counter(w for t in tr.text for w in tok(t))
    vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common(20000)) if c >= 2}
    print(f"  Deep-SE: vocabulario {len(vocab):,} palabras")

    def enc(frame):
        X = np.zeros((len(frame), MAXLEN), dtype=np.int64)
        for i, t in enumerate(frame.text):
            ids = [vocab.get(w, 1) for w in tok(t)]
            X[i, :len(ids)] = ids
        return torch.tensor(X)

    Xtr, Xva, Xte = enc(tr).to(dev), enc(va).to(dev), enc(te).to(dev)
    ytr = torch.tensor(np.log(tr.effort_hours.to_numpy()),
                       dtype=torch.float32, device=dev)
    y_va = va.effort_hours.to_numpy()

    class DeepSE(nn.Module):
        def __init__(self, v, d=100, h=100):
            super().__init__()
            self.emb = nn.Embedding(v + 2, d, padding_idx=0)
            self.lstm = nn.LSTM(d, h, batch_first=True)
            self.drop = nn.Dropout(0.4)
            self.out = nn.Linear(h, 1)

        def forward(self, x):
            e = self.drop(self.emb(x))
            o, _ = self.lstm(e)
            mask = (x != 0).unsqueeze(-1).float()
            pooled = (o * mask).sum(1) / mask.sum(1).clamp(min=1)
            return self.out(self.drop(pooled)).squeeze(-1)

    net = DeepSE(len(vocab)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.L1Loss()          # Deep-SE optimises MAE
    best, best_state = np.inf, None
    for ep in range(30):
        net.train()
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            lossf(net(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            lp = net(Xva).cpu().numpy()
        v = float(np.mean(np.abs(np.log(y_va) - lp)))
        if v < best:
            best, best_state = v, {k: x.detach().cpu().clone()
                                   for k, x in net.state_dict().items()}
            best_ep = ep
    net.load_state_dict(best_state)
    net.eval()
    print(f"  Deep-SE: mejor epoca {best_ep+1}/30  val MAE(log) {best:.4f}")
    with torch.no_grad():
        return np.exp(net(Xte).cpu().numpy())


# --------------------------------------------------------------------------
def gpt2sp(tr, va, te, dev):
    import torch
    import torch.nn as nn
    from transformers import GPT2Model, GPT2Tokenizer

    tk = GPT2Tokenizer.from_pretrained("gpt2")
    tk.pad_token = tk.eos_token
    base = GPT2Model.from_pretrained("gpt2").to(dev)

    class GPT2SP(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b
            self.head = nn.Sequential(nn.Dropout(0.1),
                                      nn.Linear(b.config.n_embd, 1))

        def forward(self, ids, mask):
            h = self.b(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            return self.head((h * m).sum(1) / m.sum(1).clamp(min=1e-9)).squeeze(-1)

    net = GPT2SP(base).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    lossf = nn.L1Loss()
    ytr = np.log(tr.effort_hours.to_numpy()).astype(np.float32)
    y_va = va.effort_hours.to_numpy()
    txt = tr.text.to_numpy()

    def infer(frame, bs=32):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(frame), bs):
                b = tk(list(frame.text.to_numpy()[i:i + bs]), padding=True,
                       truncation=True, max_length=128, return_tensors="pt")
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=dev == "cuda"):
                    out.append(net(b["input_ids"].to(dev),
                                   b["attention_mask"].to(dev)).float().cpu().numpy())
        return np.concatenate(out)

    best, best_state, best_ep = np.inf, None, -1
    for ep in range(3):
        net.train()
        perm = np.random.default_rng(ep).permutation(len(txt))
        for i in range(0, len(perm), 8):
            idx = perm[i:i + 8]
            b = tk(list(txt[idx]), padding=True, truncation=True,
                   max_length=128, return_tensors="pt")
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                loss = lossf(net(b["input_ids"].to(dev), b["attention_mask"].to(dev)),
                             torch.tensor(ytr[idx], device=dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        v = float(np.mean(np.abs(np.log(y_va) - infer(va))))
        print(f"  GPT2SP: epoca {ep+1}/3  val MAE(log) {v:.4f}")
        if v < best:
            best, best_ep = v, ep
            best_state = {k: x.detach().cpu().clone()
                          for k, x in net.state_dict().items()}
    net.load_state_dict(best_state)
    print(f"  GPT2SP: mejor epoca {best_ep+1}/3")
    return np.exp(infer(te))


# --------------------------------------------------------------------------
def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = cargar()
    y = te.effort_hours.to_numpy()
    print(f"[{dev}] train {len(tr):,}  val {len(va):,}  test {len(te):,}\n")

    filas = []
    base_p = te.project.map(tr.groupby("project").effort_hours.median())
    base_p = base_p.fillna(tr.effort_hours.median()).to_numpy()
    filas.append(metricas(y, base_p, "linea base (mediana por proyecto)", tr, te))

    emb = np.load("data/processed/embeddings/"
                  "paraphrase-multilingual-mpnet-base-v2.npy")
    ids = pd.read_parquet("data/processed/agix_effort_clean.parquet",
                          columns=["task_id"]).task_id.to_numpy()
    pos = {t: i for i, t in enumerate(ids)}
    from sklearn.linear_model import Ridge
    Xtr = emb[[pos[t] for t in tr.task_id]]
    Xte = emb[[pos[t] for t in te.task_id]]
    r = Ridge(alpha=1.0).fit(Xtr, np.log(tr.effort_hours.to_numpy()))
    filas.append(metricas(y, np.exp(r.predict(Xte)),
                          "nuestro: ridge sobre SBERT congelado", tr, te))

    t0 = time.time()
    filas.append(metricas(y, deep_se(tr, va, te, dev), "Deep-SE (LSTM)", tr, te))
    print(f"  Deep-SE entrenado en {time.time()-t0:.0f}s\n")

    t0 = time.time()
    filas.append(metricas(y, gpt2sp(tr, va, te, dev), "GPT2SP (GPT-2 ajustado)",
                          tr, te))
    print(f"  GPT2SP entrenado en {time.time()-t0:.0f}s\n")

    res = pd.DataFrame(filas)
    print("=" * 100)
    print(res.to_string(index=False))
    for _, x in res.iterrows():
        E.record("literatura", x.modelo.split(" (")[0].replace(" ", "_"),
                 "split_t3", {"MdMRE": x.MdMRE, "PRED25": x.PRED25,
                              "MAE_h": x.MAE_h, "RMSE_log": x.RMSE_log,
                              "spearman": x.spearman})
    res.to_csv("results/baselines_literatura.csv", index=False)
    print("\nescrito results/baselines_literatura.csv")


if __name__ == "__main__":
    main()
