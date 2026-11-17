"""Step 1, arm B: fine-tune the encoder using ONLY the temporal training block.

The earlier fine-tune is unusable here: 70.9% of the temporal test set sat in
its training data. This retrains from the pretrained checkpoint on
split_t3 == 'train', early-stops on 'val', and never reads 'test'.

The resulting encoder then embeds every row, cached separately from the frozen
embeddings so both arms stay available for comparison.
"""
import sys
import time

import numpy as np
import pandas as pd

DATA = "data/processed/pbe_effort_v3.parquet"
MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CACHE = "data/processed/embeddings/mpnet_ft_temporal.npy"
IDS = "data/processed/embeddings/mpnet_ft_temporal.ids.npy"
MAX_LEN, BATCH, EPOCHS = 192, 16, 4
LR_ENC, LR_HEAD = 2e-5, 1e-3


def main():
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1]
    tr = sw[sw.split_t3 == "train"]
    va = sw[sw.split_t3 == "val"]
    print(f"[{dev}] ajuste fino sobre {len(tr):,} filas, validacion {len(va):,}")
    print("  el bloque de prueba no se lee en ningun momento de este script")

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = AutoModel.from_pretrained(MODEL).to(dev)

    class Reg(nn.Module):
        def __init__(self, e):
            super().__init__()
            self.enc = e
            h = e.config.hidden_size
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(h, 256),
                                      nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(256, 1))

        def pool(self, ids, mask):
            out = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            return (out * m).sum(1) / m.sum(1).clamp(min=1e-9)

        def forward(self, ids, mask):
            return self.head(self.pool(ids, mask)).squeeze(-1)

    model = Reg(enc).to(dev)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": LR_ENC},
                             {"params": model.head.parameters(), "lr": LR_HEAD}],
                            weight_decay=0.01)
    steps = (len(tr) // BATCH + 1) * EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[LR_ENC, LR_HEAD], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    lossf = nn.HuberLoss(delta=1.0)

    def batches(texts, bs):
        for i in range(0, len(texts), bs):
            b = tok(list(texts[i:i + bs]), padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
            yield b["input_ids"].to(dev), b["attention_mask"].to(dev), i

    def predict(texts, bs=64):
        model.eval()
        out = []
        with torch.no_grad():
            for ids, mask, _ in batches(texts, bs):
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=dev == "cuda"):
                    out.append(model(ids, mask).float().cpu().numpy())
        return np.concatenate(out)

    txt, y = tr.text.to_numpy(), np.log(tr.effort_hours.to_numpy()).astype(np.float32)
    y_va = va.effort_hours.to_numpy()
    best, best_state, best_ep = np.inf, None, -1
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.default_rng(ep).permutation(len(txt))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            b = tok(list(txt[idx]), padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt")
            ids, mask = b["input_ids"].to(dev), b["attention_mask"].to(dev)
            yb = torch.tensor(y[idx], device=dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                loss = lossf(model(ids, mask), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        lp = predict(va.text.to_numpy())
        vmae = float(np.mean(np.abs(np.log(y_va) - lp)))
        print(f"  epoca {ep+1}/{EPOCHS}  perdida {tot/len(perm):.4f}  "
              f"val MAE(log) {vmae:.4f}  "
              f"val MdMRE {np.median(np.abs(y_va-np.exp(lp))/y_va):.4f}  "
              f"{time.time()-t0:.0f}s")
        if vmae < best:
            best, best_ep = vmae, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    print(f"  mejor epoca segun validacion: {best_ep+1}")
    model.load_state_dict(best_state)

    # embed every row with the selected checkpoint
    model.eval()
    todos = df.text.fillna("").to_numpy()
    embs = []
    with torch.no_grad():
        for ids, mask, _ in batches(todos, 64):
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                embs.append(model.pool(ids, mask).float().cpu().numpy())
    emb = np.vstack(embs)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
    np.save(CACHE, emb.astype(np.float32))
    np.save(IDS, df.task_id.to_numpy())
    print(f"\nguardado {CACHE}  shape={emb.shape}  "
          f"({time.time()-t0:.0f}s en total)")


if __name__ == "__main__":
    main()
