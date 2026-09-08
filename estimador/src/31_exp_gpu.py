"""GPU arm: the learners on this problem that can actually use the RTX 5060.

scikit-learn is CPU-only, so the GPU is reachable here only through XGBoost's
CUDA tree builder, CatBoost's GPU task type, and a PyTorch head on the
embeddings. Timings are reported against the CPU equivalents from the sweep.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import features as F  # noqa: E402

pd.set_option("display.width", 250)
E.DATA = "data/processed/agix_effort_v2.parquet"


def torch_mlp(Xtr, ytr, Xva, Xte, epochs=120, seed=0):
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    to = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32, device=dev)
    xtr, xva, xte = to(Xtr), to(Xva), to(Xte)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev).unsqueeze(1)

    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], 512), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.15),
        nn.Linear(128, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.HuberLoss(delta=1.0)
    n, bs = len(xtr), 256
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(net(xtr[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        return (net(xte).cpu().numpy().ravel(),
                net(xva).cpu().numpy().ravel(), dev)


def main():
    import torch
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor

    print(f"CUDA disponible: {torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    emb, pos = F.load_embeddings_and_pos()

    for split_col in ["split", "split_project"]:
        tr, va, te = E.load(split_col)
        fb = F.FeatureBuilder(use_kind=True, use_source=True, use_priority=False,
                              use_numeric=False, use_project=False)
        Xtr, Xva, Xte = fb.fit_transform(tr, [va, te], emb, pos)
        ytr = np.log(tr.effort_hours.to_numpy())
        y_va, y_te = va.effort_hours.to_numpy(), te.effort_hours.to_numpy()
        print(f"\n{'='*78}\nsplit={split_col}  train {len(tr):,}  "
              f"test {len(te):,}\n{'='*78}")

        runs = []
        for name, mk in [
            ("xgb_gpu", lambda: XGBRegressor(
                n_estimators=600, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0,
                device="cuda", tree_method="hist", random_state=0)),
            ("xgb_cpu", lambda: XGBRegressor(
                n_estimators=600, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.6, reg_lambda=2.0,
                device="cpu", tree_method="hist", n_jobs=-1, random_state=0)),
            ("catboost_gpu", lambda: CatBoostRegressor(
                iterations=800, depth=6, learning_rate=0.06,
                task_type="GPU", devices="0", verbose=0, random_seed=0)),
        ]:
            t0 = time.time()
            mdl = mk().fit(Xtr, ytr)
            lp_te, lp_va = mdl.predict(Xte), mdl.predict(Xva)
            secs = time.time() - t0
            m = E.point_metrics(y_te, np.exp(lp_te))
            m.update(E.interval_metrics(lp_va, y_va, lp_te, y_te))
            m["compresion"] = float(np.std(lp_te) / np.std(np.log(y_te)))
            m["spearman"] = float(pd.Series(y_te).corr(
                pd.Series(np.exp(lp_te)), method="spearman"))
            m["segundos"] = float(secs)
            E.record("gpu", name, split_col, m)
            runs.append(dict(modelo=name, **{k: round(v, 4) for k, v in m.items()}))
            print(f"  {name:14s} MdMRE {m['MdMRE']:.4f}  PRED25 {m['PRED25']:.4f}  "
                  f"comp {m['compresion']:.3f}  rho {m['spearman']:.3f}  {secs:6.1f}s")

        t0 = time.time()
        lp_te, lp_va, dev = torch_mlp(Xtr, ytr, Xva, Xte)
        secs = time.time() - t0
        m = E.point_metrics(y_te, np.exp(lp_te))
        m.update(E.interval_metrics(lp_va, y_va, lp_te, y_te))
        m["compresion"] = float(np.std(lp_te) / np.std(np.log(y_te)))
        m["spearman"] = float(pd.Series(y_te).corr(
            pd.Series(np.exp(lp_te)), method="spearman"))
        m["segundos"] = float(secs)
        E.record("gpu", f"torch_mlp_{dev}", split_col, m)
        runs.append(dict(modelo=f"torch_mlp_{dev}",
                         **{k: round(v, 4) for k, v in m.items()}))
        print(f"  {'torch_mlp':14s} MdMRE {m['MdMRE']:.4f}  "
              f"PRED25 {m['PRED25']:.4f}  comp {m['compresion']:.3f}  "
              f"rho {m['spearman']:.3f}  {secs:6.1f}s  [{dev}]")

        print("\n" + pd.DataFrame(runs)[
            ["modelo", "MdMRE", "PRED25", "RMSE_log", "compresion", "spearman",
             "segundos"]].to_string(index=False))


if __name__ == "__main__":
    main()
