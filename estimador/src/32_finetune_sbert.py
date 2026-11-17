"""Fine-tune the sentence encoder end-to-end for effort regression.

Every previous arm froze the encoder and trained a head on top. This adapts the
representation itself to the task, which is the only remaining lever that could
move the 0.66 ceiling.

The risk is overfitting: 278M parameters against 12k training rows. Mitigated
with a low learning rate, a short schedule, dropout, and early stopping on the
validation split -- and the honest test is the unseen-organisation split, where
memorising the corpus does not help.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

E.DATA = "data/processed/pbe_effort_v2.parquet"
MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MAX_LEN = 192
BATCH = 16
EPOCHS = 4
LR_ENCODER = 2e-5
LR_HEAD = 1e-3


def build(dev):
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = AutoModel.from_pretrained(MODEL).to(dev)

    class Reg(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.enc = encoder
            h = encoder.config.hidden_size
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(h, 256),
                                      nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(256, 1))

        def forward(self, ids, mask):
            out = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return self.head(pooled).squeeze(-1)

    return tok, Reg(enc).to(dev)


def encode_batch(tok, texts, dev):
    import torch
    b = tok(list(texts), padding=True, truncation=True, max_length=MAX_LEN,
            return_tensors="pt")
    return b["input_ids"].to(dev), b["attention_mask"].to(dev)


def predict(model, tok, texts, dev, bs=64):
    import torch
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            ids, mask = encode_batch(tok, texts[i:i + bs], dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                outs.append(model(ids, mask).float().cpu().numpy())
    return np.concatenate(outs)


def run(split_col):
    import torch
    import torch.nn as nn

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te = E.load(split_col)
    print(f"\n{'='*78}\nsplit={split_col}  train {len(tr):,}  val {len(va):,}  "
          f"test {len(te):,}  [{dev}]\n{'='*78}")

    tok, model = build(dev)
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": LR_ENCODER},
        {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=0.01)
    steps = (len(tr) // BATCH + 1) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_ENCODER, LR_HEAD], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    lossf = nn.HuberLoss(delta=1.0)

    txt_tr = tr.text.to_numpy()
    y_tr = np.log(tr.effort_hours.to_numpy()).astype(np.float32)
    y_va, y_te = va.effort_hours.to_numpy(), te.effort_hours.to_numpy()

    best, best_state, best_ep = np.inf, None, -1
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.default_rng(ep).permutation(len(txt_tr))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            ids, mask = encode_batch(tok, txt_tr[idx], dev)
            yb = torch.tensor(y_tr[idx], device=dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                loss = lossf(model(ids, mask), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss) * len(idx)
        lp_va = predict(model, tok, va.text.to_numpy(), dev)
        vmae = float(np.mean(np.abs(np.log(y_va) - lp_va)))
        mdmre = float(np.median(np.abs(y_va - np.exp(lp_va)) / y_va))
        print(f"  epoca {ep+1}/{EPOCHS}  perdida {tot/len(perm):.4f}  "
              f"val MAE(log) {vmae:.4f}  val MdMRE {mdmre:.4f}  "
              f"{time.time()-t0:.0f}s")
        if vmae < best:
            best, best_ep = vmae, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    print(f"  mejor epoca: {best_ep+1}")
    model.load_state_dict(best_state)
    lp_te = predict(model, tok, te.text.to_numpy(), dev)
    lp_va = predict(model, tok, va.text.to_numpy(), dev)

    m = E.point_metrics(y_te, np.exp(lp_te))
    m.update(E.interval_metrics(lp_va, y_va, lp_te, y_te))
    m["compresion"] = float(np.std(lp_te) / np.std(np.log(y_te)))
    m["spearman"] = float(pd.Series(y_te).corr(pd.Series(np.exp(lp_te)),
                                               method="spearman"))
    m["segundos"] = float(time.time() - t0)
    E.show(E.record("finetune", "mpnet_e2e", split_col, m,
                    config={"model": MODEL, "epochs": EPOCHS,
                            "mejor_epoca": best_ep + 1, "lr_enc": LR_ENCODER,
                            "max_len": MAX_LEN, "batch": BATCH}))
    print(f"    compresion {m['compresion']:.3f}  spearman {m['spearman']:.3f}  "
          f"{m['segundos']:.0f}s")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for sc in ["split", "split_project"]:
        run(sc)
