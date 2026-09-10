"""End-to-end fine-tune on the enlarged corpus -- the one lever data volume gates.

Every ceiling test so far kept the encoder frozen and varied the head, so the
representation never had a chance to change. Fine-tuning the encoder itself was
tried on v3 and was hopeless there: 10,597 rows cannot reshape a 278M-parameter
transformer without memorising them. With 63,926 rows the attempt is at least
fair, and it is the only remaining explanation for the 0.68 plateau that we have
not tested -- if the representation is the binding constraint, this is where it
gives.

Trained on v3-train + JiraPublico-train, early-stopped on v3 validation, and
read on the fixed v3 test plus the 150 held-out external projects. L1 loss on
log-hours, matching Deep-SE and GPT2SP so the number is comparable to the
baselines we already reimplemented.
"""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

DATA = "data/processed/agix_effort_v4.parquet"
MODELO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MAXLEN, BATCH, EPOCAS, LR = 128, 32, 3, 2e-5
SALIDA = "data/processed/embeddings/v4_mpnet_ajustado"


def bloques():
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    v3 = sw[sw.source != "JiraPublico"]
    jp = sw[sw.source == "JiraPublico"]
    tr = pd.concat([v3[v3.split_t3 == "train"], jp[jp.split_t3 == "train"]],
                   ignore_index=True)
    return (tr, v3[v3.split_t3 == "val"], v3[v3.split_t3 == "test"],
            jp[jp.split_t3 == "externo"])


def main():
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va, te, ext = bloques()
    print(f"[{dev}] train {len(tr):,}  val {len(va):,}  test {len(te):,}  "
          f"externo {len(ext):,}")
    if dev == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    tk = AutoTokenizer.from_pretrained(MODELO)
    base = AutoModel.from_pretrained(MODELO).to(dev)

    class Regresor(nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b
            h = b.config.hidden_size
            self.cabeza = nn.Sequential(nn.Dropout(0.1), nn.Linear(h, 1))

        def forward(self, ids, mask):
            o = self.b(input_ids=ids, attention_mask=mask).last_hidden_state
            m = mask.unsqueeze(-1).float()
            return self.cabeza((o * m).sum(1) / m.sum(1).clamp(min=1e-9)).squeeze(-1)

    net = Regresor(base).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.01)
    pasos = (len(tr) // BATCH + 1) * EPOCAS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=pasos,
                                                pct_start=0.1)
    escala = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    perdida = nn.L1Loss()

    txt = tr.text.fillna("").to_numpy()
    y = np.log(tr.effort_hours.to_numpy()).astype(np.float32)

    def inferir(frame, bs=128):
        net.eval()
        t = frame.text.fillna("").to_numpy()
        out = []
        with torch.no_grad():
            for i in range(0, len(t), bs):
                b = tk(list(t[i:i + bs]), padding=True, truncation=True,
                       max_length=MAXLEN, return_tensors="pt")
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=dev == "cuda"):
                    out.append(net(b["input_ids"].to(dev),
                                   b["attention_mask"].to(dev)).float().cpu().numpy())
        return np.concatenate(out)

    mejor, mejor_estado, mejor_ep = np.inf, None, -1
    y_va = va.effort_hours.to_numpy()
    for ep in range(EPOCAS):
        net.train()
        t0 = time.time()
        perm = np.random.default_rng(ep).permutation(len(txt))
        acum = 0.0
        for k, i in enumerate(range(0, len(perm), BATCH)):
            idx = perm[i:i + BATCH]
            b = tk(list(txt[idx]), padding=True, truncation=True,
                   max_length=MAXLEN, return_tensors="pt")
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                l = perdida(net(b["input_ids"].to(dev), b["attention_mask"].to(dev)),
                            torch.tensor(y[idx], device=dev))
            escala.scale(l).backward()
            escala.step(opt)
            escala.update()
            sched.step()
            acum += float(l)
            if k % 400 == 0:
                print(f"    ep{ep+1} paso {k:>5}/{len(perm)//BATCH}  "
                      f"perdida {acum/(k+1):.4f}  {time.time()-t0:5.0f}s", flush=True)
        v = float(np.mean(np.abs(np.log(y_va) - inferir(va))))
        mv = E.point_metrics(y_va, np.exp(inferir(va)))
        print(f"  epoca {ep+1}/{EPOCAS}  {time.time()-t0:.0f}s  "
              f"val MAE(log) {v:.4f}  val MdMRE {mv['MdMRE']:.4f}", flush=True)
        if v < mejor:
            mejor, mejor_ep = v, ep
            mejor_estado = {k: x.detach().cpu().clone()
                            for k, x in net.state_dict().items()}
    net.load_state_dict(mejor_estado)
    print(f"\nmejor epoca: {mejor_ep+1}")

    filas = []
    for etiqueta, d in [("prueba fija de v3", te), ("externo (150 proyectos)", ext)]:
        p = np.exp(inferir(d))
        yy = d.effort_hours.to_numpy()
        m = E.point_metrics(yy, p)
        rho = float(pd.Series(yy).corr(pd.Series(p), method="spearman"))
        comp = float(np.std(np.log(np.clip(p, .01, None))) / np.std(np.log(yy)))
        print(f"  {etiqueta:26s} MdMRE {m['MdMRE']:.4f}  PRED25 {m['PRED25']:.4f}  "
              f"MAE {m['MAE_h']:.2f}h  rho {rho:+.3f}  compr {comp:.3f}")
        filas.append(dict(bloque=etiqueta, MdMRE=round(m["MdMRE"], 4),
                          PRED25=round(m["PRED25"], 4),
                          PRED50=round(m["PRED50"], 4),
                          MAE_h=round(m["MAE_h"], 2),
                          RMSE_log=round(m["RMSE_log"], 4),
                          spearman=round(rho, 3), compresion=round(comp, 3)))
        E.record("v4_ajuste_fino", f"mpnet_e2e_{etiqueta[:12]}", "split_t3_v4", m)
    pd.DataFrame(filas).to_csv("results/finetune_v4.csv", index=False)

    # guardar las representaciones para que otros brazos las reutilicen
    todo = pd.read_parquet(DATA, columns=["task_id", "text", "is_software_task"])
    todo = todo[todo.is_software_task == 1]
    net.eval()
    reps = []
    with torch.no_grad():
        t = todo.text.fillna("").to_numpy()
        for i in range(0, len(t), 128):
            b = tk(list(t[i:i + 128]), padding=True, truncation=True,
                   max_length=MAXLEN, return_tensors="pt")
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                o = net.b(input_ids=b["input_ids"].to(dev),
                          attention_mask=b["attention_mask"].to(dev)).last_hidden_state
                mk = b["attention_mask"].to(dev).unsqueeze(-1).float()
                reps.append(((o * mk).sum(1) / mk.sum(1).clamp(min=1e-9))
                            .float().cpu().numpy())
    np.save(f"{SALIDA}.npy", np.concatenate(reps).astype(np.float32))
    np.save(f"{SALIDA}.ids.npy", todo.task_id.to_numpy())
    print(f"\nguardado {SALIDA}.npy")
    print("escrito results/finetune_v4.csv")


if __name__ == "__main__":
    main()
