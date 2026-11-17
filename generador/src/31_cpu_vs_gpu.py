"""Can this be deployed without a GPU, and is it still cheaper than an LLM?

The thesis argument is cost and speed, so the comparison that matters is on the
SAME hardware. If production has no GPU, the LLM alternative has no GPU either.

Measures, per project, wall-clock to a complete backlog:
  nuestro_gpu       full pipeline on the GPU (the number already in the bench)
  nuestro_cpu       same pipeline, CPU only
  nuestro_cpu_ligero  CPU with one generator and beam 1 (the cheap setting)
  <llm>_cpu         the same LLMs with Ollama forced to CPU (num_gpu 0)

LLM runs are the slow part, so they use fewer projects; medians are reported
with the n used for each arm.

usage: python src/31_cpu_vs_gpu.py [N_NUESTRO] [N_LLM]
"""
import importlib.util
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

BASE = r"F:\Software Task generation Model"
OLLAMA = "http://127.0.0.1:11434/api/generate"
SEED = 20260924
K_HISTORIAS = 12
P_HIST = ("You are a product owner. Write exactly {k} user stories for the following project, "
          "each on its own line, in the form 'As a <role>, I want <goal>, so that <benefit>.' "
          "Output only the {k} lines.\n\nProject: {desc}")
P_TAREAS = ("Break this user story into the concrete implementation tasks a development team "
            "would create. One short task title per line, no numbering, no explanations.\n\n"
            "User story: {historia}")


def ollama_cpu(modelo, prompt, num_predict=600):
    body = json.dumps({"model": modelo, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": num_predict,
                                   "num_gpu": 0}}).encode()     # 0 layers on GPU = CPU only
    req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return r["response"].strip(), r.get("eval_count", 0)


def casos(n):
    D = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_dalpiaz_test.parquet")
    T = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_test.parquet")
    T = T[(T.estilo == "historia") & ~T.proj.str.startswith("dalpiaz:")]
    g = pd.concat([D, T]).groupby("parent_uid").entrada.first()
    return g.sample(min(n, len(g)), random_state=SEED)


def main():
    n_nuestro = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n_llm = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    C = casos(max(n_nuestro, n_llm))
    filas = []

    os.environ["PBE_DEVICE"] = "cpu"
    import torch
    torch.set_num_threads(6)                      # i5-12400: 6 performance cores
    spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
    pl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pl)
    print(f"dispositivo del pipeline: {pl.DISPOSITIVO}, hilos {torch.get_num_threads()}", flush=True)

    import gc
    import psutil
    proc = psutil.Process()
    # 4 threads approximates the presentation laptop (Ryzen 3, 8 GB, no GPU)
    for hilos in (6, 4):
        torch.set_num_threads(hilos)
        for etq, conj, beams in [("completo", "dos_modelos", 4), ("ligero", "no", 1)]:
            pl.BEAMS_ABAJO = beams
            gc.collect()
            base_ram = proc.memory_info().rss / 2**30
            p = pl.Pipeline(conjunto=conj)
            ram_cargado = proc.memory_info().rss / 2**30
            p(C.iloc[0], k_historias=3)               # warm-up
            segs, tareas, etapas = [], [], []
            for d in C.head(n_nuestro):
                t = time.time()
                r = p(d, k_historias=K_HISTORIAS)
                segs.append(time.time() - t)
                tareas.append(r["n_tareas"])
                etapas.append(r["tiempos"])
            ram_pico = proc.memory_info().rss / 2**30
            E = pd.DataFrame(etapas).median()
            nombre = f"nuestro_cpu_{etq}_{hilos}h"
            filas.append(dict(sistema=nombre, n=len(segs), s_mediana=float(np.median(segs)),
                              s_p90=float(np.quantile(segs, .9)), tareas=float(np.mean(tareas)),
                              carga_s=p.carga_s, ram_modelos_GB=round(ram_cargado - base_ram, 2),
                              ram_pico_GB=round(ram_pico, 2),
                              etapa_tareas=float(E.get("generar_tareas", np.nan)),
                              etapa_historias=float(E.get("generar_historias", np.nan))))
            print(f"  {nombre:<26} mediana {np.median(segs):6.1f} s  p90 {np.quantile(segs,.9):6.1f} s  "
                  f"tareas {np.mean(tareas):.0f}  carga {p.carga_s}s  RAM {ram_pico:.2f} GB", flush=True)
            del p
            gc.collect()

    for modelo in ["llama3.1:8b"]:   # 7B on CPU is minutes per backlog; one arm is enough
        segs, tareas = [], []
        for d in C.head(n_llm):
            t = time.time()
            txt, _ = ollama_cpu(modelo, P_HIST.format(k=K_HISTORIAS, desc=d))
            hist = [x.strip(" .-*") for x in txt.split("\n") if len(x.strip()) >= 10][:K_HISTORIAS]
            n_t = 0
            for h in hist:
                tt, _ = ollama_cpu(modelo, P_TAREAS.format(historia=h), num_predict=300)
                n_t += len([x for x in tt.split("\n") if len(x.strip()) >= 10])
            segs.append(time.time() - t)
            tareas.append(n_t)
            print(f"    {modelo} proyecto {len(segs)}/{n_llm}: {segs[-1]:.0f} s", flush=True)
        filas.append(dict(sistema=f"{modelo}_cpu", n=len(segs), s_mediana=float(np.median(segs)),
                          s_p90=float(np.quantile(segs, .9)), tareas=float(np.mean(tareas)),
                          carga_s=np.nan, etapa_tareas=np.nan, etapa_historias=np.nan))
        print(f"  {modelo}_cpu  mediana {np.median(segs):6.1f} s  tareas {np.mean(tareas):.0f}", flush=True)

    # the GPU number comes from the bench, same projects family
    B = pd.DataFrame([json.loads(l) for l in open(f"{BASE}/results/20_banco.jsonl", encoding="utf-8")])
    for sis, etq in [("pbe", "nuestro_gpu"), ("llama3.1:8b", "llama3.1:8b_gpu"),
                     ("mistral:7b", "mistral:7b_gpu")]:
        sub = B[(B.sistema == sis) & ((B.condicion == "-") | (B.condicion == "cero"))]
        if len(sub):
            filas.append(dict(sistema=etq, n=len(sub), s_mediana=float(sub.segundos.median()),
                              s_p90=float(sub.segundos.quantile(.9)), tareas=float(sub.n_tareas.mean()),
                              carga_s=np.nan, etapa_tareas=np.nan, etapa_historias=np.nan))

    R = pd.DataFrame(filas).round(2)
    R.to_csv(f"{BASE}/results/31_cpu_vs_gpu.csv", index=False)
    print("\n" + R.to_string(index=False))
    # reference = the setting the presentation laptop would run (light, 4 threads)
    base = R[R.sistema == "nuestro_cpu_ligero_4h"].s_mediana
    if len(base):
        b = float(base.iloc[0])
        print(f"\nveces mas lento que nuestro pipeline ligero en 4 hilos ({b:.1f} s):")
        for _, r in R.iterrows():
            if r.sistema != "nuestro_cpu_ligero_4h":
                print(f"  {r.sistema:<26} {r.s_mediana/b:6.1f}x")


if __name__ == "__main__":
    main()
