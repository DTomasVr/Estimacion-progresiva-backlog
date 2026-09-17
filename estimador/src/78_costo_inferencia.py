"""What each option costs to run: our model against the local LLMs.

Accuracy is only half of the feasibility argument for a thesis artefact. The
other half is what it takes to serve a prediction: how long the user waits, how
much memory the box needs, how big the thing is to ship, and whether the same
task always gets the same answer. This script measures those for our pipeline
on this machine and puts them next to what the LLM benchmark already recorded.

Measured here, not quoted:
  - encoder latency (one task at a time, as the product would, and in batches)
    on GPU and on CPU
  - regressor latency, one task and a thousand
  - disk footprint of the encoder weights and of the fitted regressor
  - peak GPU memory of the encoder
  - determinism: the same task encoded and predicted twice

Taken from the LLM benchmark (results/llm_bench/resumen.csv) and from
`ollama list`: seconds per task actually observed, and the weight size of each
local model. The LLM figures are read at run time so this table cannot drift
from the benchmark that produced them.
"""
import json
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import sys
sys.path.insert(0, "src")
import importlib.util

spec = importlib.util.spec_from_file_location("m71", "src/71_estimacion_usuario.py")
M71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M71)

DATA = "data/processed/agix_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
MODELO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OUT = Path("results/costo")
MODELOS = Path("models")


def cronometrar(f, repeticiones=20, calentar=3):
    for _ in range(calentar):
        f()
    t = []
    for _ in range(repeticiones):
        t0 = time.perf_counter()
        f()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def tamano_dir(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MODELOS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    textos = sw.text.head(64).tolist()
    filas = []

    # ------------------------------------------------ codificador
    for disp in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
        if disp == "cuda":
            torch.cuda.reset_peak_memory_stats()
        st = SentenceTransformer(MODELO, device=disp)
        una = cronometrar(lambda: st.encode(textos[:1], show_progress_bar=False))
        lote = cronometrar(lambda: st.encode(textos, batch_size=32, show_progress_bar=False),
                           repeticiones=5)
        pico = (torch.cuda.max_memory_allocated() / 1e6) if disp == "cuda" else float("nan")
        filas.append(dict(componente=f"codificador SBERT ({disp})", seg_una_tarea=una,
                          seg_por_tarea_en_lote=lote / 64, memoria_pico_MB=pico))
        print(f"  codificador {disp}: {una * 1000:.0f} ms una tarea / "
              f"{lote / 64 * 1000:.1f} ms por tarea en lote de 64 / pico {pico:.0f} MB")
        if disp == "cuda":
            del st
            torch.cuda.empty_cache()
        else:
            enc_cpu = st

    # ------------------------------------------------ regresor
    emb = np.load(f"{EMB}.npy")
    ids = np.load(f"{EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fe"] = [pos[t] for t in sw.task_id]
    pool = sw[sw.split_t3 == "train"]
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                       min_frequency=25).fit(pool[M71.CAT].astype(str))
    sc = StandardScaler().fit(np.log1p(pool[M71.NUM].to_numpy(float)))
    base = lambda d: np.hstack([emb[d.fe.to_numpy()], oh.transform(d[M71.CAT].astype(str)),
                                sc.transform(np.log1p(d[M71.NUM].to_numpy(float)))])
    Xp = base(pool)
    t0 = time.perf_counter()
    reg = M71.hgb().fit(Xp, np.log(pool.effort_hours.to_numpy()))
    seg_entren = time.perf_counter() - t0
    joblib.dump({"regresor": reg, "onehot": oh, "escalador": sc}, MODELOS / "frio_v4.joblib",
                compress=3)
    x1, xk = Xp[:1], Xp[:1000]
    filas.append(dict(componente="regresor (una tarea)",
                      seg_una_tarea=cronometrar(lambda: reg.predict(x1), 200),
                      seg_por_tarea_en_lote=cronometrar(lambda: reg.predict(xk), 20) / 1000,
                      memoria_pico_MB=float("nan")))
    print(f"  regresor: entrenado en {seg_entren:.0f} s / "
          f"prediccion {filas[-1]['seg_una_tarea'] * 1000:.2f} ms")

    # ------------------------------------------------ determinismo
    a = enc_cpu.encode(textos[:4], show_progress_bar=False)
    b = enc_cpu.encode(textos[:4], show_progress_bar=False)
    dif_enc = float(np.abs(a - b).max())
    dif_reg = float(np.abs(reg.predict(xk) - reg.predict(xk)).max())

    # ------------------------------------------------ huellas en disco
    caches = [Path.home() / ".cache/huggingface/hub", Path.home() / ".cache/torch/sentence_transformers"]
    peso_enc = 0.0
    for c in caches:
        for d in (c.glob("*mpnet*") if c.exists() else []):
            peso_enc = max(peso_enc, tamano_dir(d))
    peso_reg = (MODELOS / "frio_v4.joblib").stat().st_size / 1e6

    # ------------------------------------------------ lado LLM
    llm = pd.read_csv("results/llm_bench/resumen.csv")
    crudo = llm[llm.version == "crudo"]
    try:
        sal = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=60).stdout
        pesos = {}
        for ln in sal.splitlines()[1:]:
            p = ln.split()
            if len(p) >= 4:
                pesos[p[0]] = f"{p[2]} {p[3]}"
    except Exception as e:                                   # ollama apagado
        pesos, sal = {}, f"no disponible: {e}"

    tabla_llm = []
    for m in sorted(crudo.modelo.unique()):
        v = crudo[crudo.modelo == m]
        clave = m.split(" ")[0]
        tabla_llm.append(dict(modelo=m, peso_en_disco=pesos.get(clave, "?"),
                              seg_tarea_min=float(v.seg_tarea.min()),
                              seg_tarea_max=float(v.seg_tarea.max()),
                              tokens_entrada_mediana=float(v.tokens_entrada.median())))
    t_llm = pd.DataFrame(tabla_llm)

    t = pd.DataFrame(filas)
    t.to_csv(OUT / "latencia_nuestra.csv", index=False)
    t_llm.to_csv(OUT / "latencia_llms.csv", index=False)
    resumen = {"peso_codificador_MB": round(peso_enc, 1), "peso_regresor_MB": round(peso_reg, 2),
               "segundos_entrenamiento_regresor": round(seg_entren, 1),
               "determinismo_codificador_max_dif": dif_enc,
               "determinismo_regresor_max_dif": dif_reg,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sin GPU",
               "pesos_ollama": pesos}
    json.dump(resumen, open(OUT / "resumen.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n" + "=" * 110 + "\nnuestro canal\n" + "=" * 110)
    print(t.round(5).to_string(index=False))
    print("\n" + "=" * 110 + "\nLLMs locales (medido en el banco)\n" + "=" * 110)
    print(t_llm.round(3).to_string(index=False))
    print(f"\ncodificador en disco {peso_enc:,.0f} MB / regresor {peso_reg:,.2f} MB")
    print(f"determinismo: maxima diferencia entre dos corridas identicas "
          f"{dif_enc:.2e} (codificador) y {dif_reg:.2e} (regresor)")
    print(f"\nescrito {OUT}/latencia_nuestra.csv, latencia_llms.csv, resumen.json "
          f"y {MODELOS}/frio_v4.joblib")


if __name__ == "__main__":
    main()
