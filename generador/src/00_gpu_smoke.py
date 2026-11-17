"""Day-0 smoke test on the RTX 5060 (sm_120): bf16 forward + backward + one
Adafactor step on mT5-small, then a quick batch-size probe so the full run
starts from a measured number instead of a guess."""
import json
import time

import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from transformers.optimization import Adafactor

CACHE = r"F:\Software Task generation Model\models\hf_cache"
NAME = "google/mt5-small"

t0 = time.time()
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "cap", torch.cuda.get_device_capability(), "arch", torch.cuda.get_arch_list())
tok = AutoTokenizer.from_pretrained(NAME, cache_dir=CACHE)
model = MT5ForConditionalGeneration.from_pretrained(NAME, cache_dir=CACHE,
                                                    dtype=torch.bfloat16).cuda()
print(f"carga {time.time()-t0:.1f}s  params {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

opt = Adafactor(model.parameters(), lr=1e-3, relative_step=False, scale_parameter=False)
src = tok(["As a user, I want to reset my password so that I can regain access."],
          return_tensors="pt").to("cuda")
tgt = tok(["Add reset endpoint <sep> Send reset email <sep> Build reset form"],
          return_tensors="pt").input_ids.to("cuda")
model.train()
for i in range(3):
    loss = model(**src, labels=tgt).loss
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"paso {i}  loss {loss.item():.4f}  finito {torch.isfinite(loss).item()}")

# batch probe at a representative length (input 256, output 128)
res = {}
for L_in, L_out in [(256, 128), (384, 192)]:
    for bs in [8, 16, 24, 32, 48, 64]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.randint(5, 30000, (bs, L_in), device="cuda")
            y = torch.randint(5, 30000, (bs, L_out), device="cuda")
            torch.cuda.synchronize(); t = time.time()
            for _ in range(3):
                model(input_ids=x, labels=y).loss.backward()
                opt.step(); opt.zero_grad()
            torch.cuda.synchronize()
            dt = (time.time() - t) / 3
            pk = torch.cuda.max_memory_allocated() / 2**30
            res[f"{L_in}/{L_out}/bs{bs}"] = dict(s_paso=round(dt, 3), pico_GB=round(pk, 2),
                                                ej_por_s=round(bs / dt, 1))
            print(f"in{L_in} out{L_out} bs{bs:>3}: {dt:.3f}s/paso  {bs/dt:6.1f} ej/s  pico {pk:.2f} GB")
        except torch.OutOfMemoryError:
            print(f"in{L_in} out{L_out} bs{bs:>3}: OOM")
            res[f"{L_in}/{L_out}/bs{bs}"] = "OOM"
            opt.zero_grad(set_to_none=True)
            break
json.dump(res, open(r"F:\Software Task generation Model\results\00_gpu_smoke.json", "w"), indent=1)
print(f"total {time.time()-t0:.0f}s")
