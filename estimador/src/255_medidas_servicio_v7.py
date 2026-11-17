"""Medidas de servicio del v7, para que la documentacion las lea de un fichero.

  acuerdo     % de tareas donde los vecinos concuerdan (disp <= umbral) y el
              valor probable sale de la analogia, sobre el corpus sin Apache
  servicio    el estimador SOLO, en CPU y sin cache de Hugging Face: carga,
              primera llamada, llamada en caliente y memoria residente
"""
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

OUT = Path("results/estimador_v7")


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    pr = psutil.Process()
    ram0 = pr.memory_info().rss
    t0 = time.time()
    s = importlib.util.spec_from_file_location("e6", "src/171_estimador_v6.py")
    e6 = importlib.util.module_from_spec(s)
    s.loader.exec_module(e6)
    est = e6.EstimadorV6(paquete=Path("models/estimador_pbe_v7.joblib"), dispositivo="cpu")
    carga = time.time() - t0
    gen = json.load(open("results/prueba_extremo/salidas.json", encoding="utf-8"))
    tg = [t for c in gen for h in c["fp32"]["historias"] for t in h["tareas"]][:42]
    t = time.time()
    est.estimar_tareas(tg, estilo="generador")
    primera = time.time() - t
    t = time.time()
    est.estimar_tareas(tg, estilo="generador")
    caliente = time.time() - t
    t = time.time()
    est.estimar_proyecto(tg)
    proyecto = time.time() - t
    mi = pr.memory_info()

    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[(df.is_software_task == 1) & ~df.fuente_jira.eq("Apache")]
    E5 = np.load("data/processed/embeddings/codificadores/e5-large.npy")
    Vn = E5 / np.maximum(np.linalg.norm(E5, axis=1, keepdims=True), 1e-9)
    BV, By = est.p["vec_V"].astype(np.float32), est.p["vec_y"]
    disp = np.zeros(len(sw))
    for i in range(0, len(sw), 2048):
        j = np.argpartition(-(Vn[i:i + 2048] @ BV.T), 25, axis=1)[:, :25]
        disp[i:i + 2048] = np.std(np.log(np.clip(By[j], 1e-6, None)), axis=1)

    res = {"carga_s": carga, "primera_llamada_42_s": primera,
           "caliente_42_ms": 1000 * caliente, "ms_por_tarea_caliente": 1000 * caliente / 42,
           "estimar_proyecto_42_ms": 1000 * proyecto,
           "ram_proceso_gb": mi.rss / 2**30, "ram_pico_gb": getattr(mi, "peak_wset", 0) / 2**30,
           "ram_inicial_gb": ram0 / 2**30,
           "acuerdo_vecinos": float(np.mean(disp <= est.p["umbral_dispersion"])),
           "umbral_dispersion": float(est.p["umbral_dispersion"]),
           "ram_pico_pipeline_completa_cpu_gb": 5.21,
           "nota_pipeline": "medida en la integracion del 21/10/2026: generador + estimador v7 "
                            "en CPU, tres backlogs, paquete F:/PBE-produccion"}
    json.dump(res, open(OUT / "servicio.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
