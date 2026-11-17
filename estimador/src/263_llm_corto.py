"""Comparacion corta de un experimento contra un LLM, para cerrar cada experimento.

No llama a ninguna API: reutiliza las respuestas ya guardadas en
results/llm_bench/raw/ del LLM de referencia (Qwen 3.8 27B, condicion A, el
mejor del banco) y su factor de calibracion aprendido en las 130 tareas aparte.
Sale una tabla de 2 filas (experimento y LLM calibrado) y un veredicto.

Uso desde un experimento:

    llm = importlib.util.spec_from_file_location("llm", "src/263_llm_corto.py") ...
    t = llm.tareas_banco()                   # task_id, title de las 601 tareas
    llm.comparar("mi_experimento", t.task_id, horas_predichas, Path("results/mi_exp"))

o por linea de comandos con un csv/parquet de columnas task_id, horas:

    python src/263_llm_corto.py results/mi_exp/banco.csv results/mi_exp

Las 601 tareas son del bloque de prueba y de proyectos externos enteros: el
experimento no debe haberlas visto al entrenar, o la comparacion no vale.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path("data/processed/llm_bench")
RAW = Path("results/llm_bench/raw")
REF = "qwen_qwen3.8-27b"
NOMBRE_REF = "Qwen 3.8 27B calibrado"

_s = importlib.util.spec_from_file_location("ev", "src/69_eval_llm.py")
ev = importlib.util.module_from_spec(_s)
_s.loader.exec_module(ev)


def tareas_banco():
    d = pd.read_parquet(BENCH / "muestra.parquet")[["task_id"]]
    t = pd.read_parquet("data/processed/pbe_effort_v4.parquet", columns=["task_id", "title"])
    return d.merge(t.drop_duplicates("task_id"), on="task_id", how="left")


def _jsonl(p):
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()])


def comparar(nombre, task_ids, horas, salida=None):
    mu = pd.read_parquet(BENCH / "muestra.parquet")
    exp = pd.DataFrame({"task_id": list(task_ids), "exp": np.asarray(horas, float)})
    llm = _jsonl(RAW / f"muestra__groq__{REF}__A.jsonl")
    cal = pd.read_parquet(BENCH / "calibracion.parquet").merge(
        _jsonl(RAW / f"calibracion__groq__{REF}__A.jsonl").query("valido == True")[["task_id", "horas"]],
        on="task_id")
    k = float(np.exp(np.median(np.log(cal.effort_hours / cal.horas))))
    d = mu.merge(exp, on="task_id").merge(llm[["task_id", "valido", "horas"]], on="task_id")
    ok = d.valido.fillna(False).astype(bool).to_numpy()
    p_llm = np.where(ok, d.horas.fillna(0), d.mediana_global) * k
    y = d.effort_hours.to_numpy()
    m_exp, m_llm = ev.metricas(y, d.exp.to_numpy()), ev.metricas(y, p_llm)
    lo, hi = ev.bootstrap_dif(y, p_llm, d.exp.to_numpy())
    ver = "LLM mejor" if hi < 0 else f"{nombre} mejor" if lo > 0 else "empate"
    f = lambda x: f"{x:.3f}".replace(".", ",")
    md = "\n".join([
        f"# {nombre} frente a LLM (comparación corta, {len(d)} tareas del banco)", "",
        "| modelo | MdMRE | SA | Spearman |", "|---|---|---|---|",
        f"| {nombre} | {f(m_exp['MdMRE'])} | {f(m_exp['SA'])} | {f(m_exp['spearman'])} |",
        f"| {NOMBRE_REF} | {f(m_llm['MdMRE'])} | {f(m_llm['SA'])} | {f(m_llm['spearman'])} |", "",
        f"Veredicto: **{ver}** (IC95 de la diferencia de MdMRE, LLM − {nombre}: "
        f"[{lo:+.3f}, {hi:+.3f}]). Generado por `src/263_llm_corto.py`, sin llamadas a API.", ""])
    if salida:
        Path(salida).mkdir(parents=True, exist_ok=True)
        (Path(salida) / "llm_corto.md").write_text(md, encoding="utf-8")
    print(md)
    return md


if __name__ == "__main__":
    src, out = Path(sys.argv[1]), sys.argv[2]
    p = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src)
    comparar(Path(out).name, p.task_id, p.horas, out)
