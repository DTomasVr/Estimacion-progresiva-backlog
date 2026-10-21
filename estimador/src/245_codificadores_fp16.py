"""Pasa los codificadores ajustados a float16 y mide cuanto daria podar el vocabulario.

El .joblib son 110 MB; los codificadores son 6,5 GB:

    models/e5_ajustado       2.153 MB
    models/e5_ajustado_C     2.153 MB
    e5-large base (cache HF) 2.200 MB

De cada uno, el embedding de vocabulario pesa 1.024 MB -el 46 %- y son los
250.002 tokens de XLM-R, de los que un titulo de tarea en ingles usa una
fraccion minima. Es la misma situacion que llevo al mt5-small-pruned del
generador.

Aqui se hace lo seguro -float16, que es exacto para inferencia y no toca la
estructura- y se MIDE la cobertura real del vocabulario, para saber si la poda
merece el riesgo antes de hacerla.

Se verifica que las predicciones no se mueven antes de dar nada por bueno.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, "src")
MODELOS = [Path("models/e5_ajustado"), Path("models/e5_ajustado_C")]
TOL = 2e-3
TEXTOS = [
    "Add a filter to the search bar", "Fix login redirect loop",
    "Migrate the user table to the new schema", "Update the API docs",
    "Refactor the whole payment subsystem and add tests",
    "Fix a typo in the footer", "Cache the dashboard queries",
    "Add pagination to the admin list", "Write integration tests for checkout",
    "Investigate the flaky nightly build",
]


def carga_est():
    import importlib.util
    spec = importlib.util.spec_from_file_location("e6", "src/171_estimador_v6.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.EstimadorV6()


def main():
    t0 = time.time()
    est = carga_est()
    antes = est.estimar_tareas(TEXTOS)[["horas", "valor_probable"]].to_numpy()
    del est
    print(f"referencia tomada · {time.time() - t0:.0f} s", flush=True)

    total_antes = total_desp = 0
    for d in MODELOS:
        f = d / "model.safetensors"
        resp = f.with_suffix(".safetensors.fp32")
        if not resp.exists():
            shutil.copy2(f, resp)
        t = load_file(str(f))
        if next(iter(t.values())).dtype == torch.float16:
            print(f"  {d.name}: ya estaba en float16")
            total_antes += resp.stat().st_size
            total_desp += f.stat().st_size
            continue
        total_antes += f.stat().st_size
        save_file({k: v.half() for k, v in t.items()}, str(f),
                  metadata={"format": "pt"})
        total_desp += f.stat().st_size
        print(f"  {d.name}: {resp.stat().st_size / 1e6:.0f} -> "
              f"{f.stat().st_size / 1e6:.0f} MB · {time.time() - t0:.0f} s",
              flush=True)

    est2 = carga_est()
    desp = est2.estimar_tareas(TEXTOS)[["horas", "valor_probable"]].to_numpy()
    peor = float(np.max(np.abs(desp - antes) / np.maximum(np.abs(antes), 1e-9)))
    print(f"\n  desvio relativo maximo en las predicciones: {peor:.2e}")
    if peor > TOL:
        for d in MODELOS:
            f = d / "model.safetensors"
            if f.with_suffix(".safetensors.fp32").exists():
                shutil.copy2(f.with_suffix(".safetensors.fp32"), f)
        raise SystemExit(f"DESVIO {peor:.2e} > {TOL}: restaurados los fp32")
    print(f"  OK · {total_antes / 1e6:.0f} MB -> {total_desp / 1e6:.0f} MB "
          f"en los dos codificadores ajustados")

    # ---- cuanto daria podar el vocabulario ----
    import pandas as pd
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODELOS[0]))
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    tit = ["query: " + t for t in sw.title.fillna("").astype(str).tolist()]
    usados = set()
    for i in range(0, len(tit), 2000):
        for ids in tok(tit[i:i + 2000], add_special_tokens=True)["input_ids"]:
            usados.update(ids)
    V = len(tok)
    print(f"\n{'=' * 70}\nLA PODA DEL VOCABULARIO\n{'=' * 70}")
    print(f"  vocabulario del modelo       {V:,} tokens")
    print(f"  usados por las {len(tit):,} tareas  {len(usados):,} "
          f"({len(usados) / V:.1%})")
    emb_mb = V * 1024 * 2 / 1e6
    pod_mb = len(usados) * 1024 * 2 / 1e6
    print(f"  embedding en float16         {emb_mb:.0f} MB -> {pod_mb:.0f} MB "
          f"podado")
    print(f"  cada codificador             {total_desp / 2e6:.0f} MB -> "
          f"{total_desp / 2e6 - emb_mb + pod_mb:.0f} MB")
    print(f"  los tres juntos              "
          f"{(total_desp / 2e6) * 3:.0f} MB -> "
          f"{((total_desp / 2e6) - emb_mb + pod_mb) * 3:.0f} MB")
    json.dump({"vocab": V, "usados": len(usados),
               "mb_antes": total_antes / 1e6, "mb_fp16": total_desp / 1e6,
               "desvio": peor}, open("results/codificadores_fp16.json", "w"),
              indent=1)
    print(f"\n· {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
