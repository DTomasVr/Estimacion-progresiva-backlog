"""Predicciones del estimador v7 DESPLEGADO sobre el banco de comparacion con LLM.

El banco (data/processed/llm_bench/muestra.parquet, 601 tareas) trae como
referencia las predicciones de un modelo de septiembre. Para comparar con los
LLM actuales hay que hacerlo contra lo que hoy esta en produccion: el v7 con su
codificador ONNX int8 podado. Las 601 tareas son del bloque de prueba (301) y de
proyectos externos enteros (300): ninguna entra en el entrenamiento del v7.

El v7 recibe solo el titulo, como en produccion; el LLM recibe texto_llm, que es
el mismo titulo con su tipo. Se guardan las horas (mediana por tarea) y el
valor probable (ajustado a la rejilla).
"""
import importlib.util
from pathlib import Path

import pandas as pd

BENCH = Path("data/processed/llm_bench")


def main():
    s = importlib.util.spec_from_file_location("e6", "src/171_estimador_v6.py")
    e6 = importlib.util.module_from_spec(s)
    s.loader.exec_module(e6)
    est = e6.EstimadorV6(paquete=Path("models/estimador_pbe_v7.joblib"), dispositivo="cpu")
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet", columns=["task_id", "title"])
    for conjunto in ("muestra", "calibracion"):
        d = pd.read_parquet(BENCH / f"{conjunto}.parquet")[["task_id"]]
        d = d.merge(df.drop_duplicates("task_id"), on="task_id", how="left")
        assert d.title.notna().all(), "tareas del banco sin titulo"
        r = est.estimar_tareas(d.title.fillna("").astype(str).tolist())
        d["v7_horas"] = r.horas.to_numpy()
        d["v7_probable"] = r.valor_probable.to_numpy()
        d[["task_id", "v7_horas", "v7_probable"]].to_parquet(BENCH / f"v7_{conjunto}.parquet")
        print(conjunto, len(d))


if __name__ == "__main__":
    main()
