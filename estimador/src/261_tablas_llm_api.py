"""Tablas de la comparacion del estimador v7 contra LLM actuales por API.

Lee las corridas de results/llm_bench/raw (las de 259_banco_api.py y, como
contexto, las locales de septiembre) y las compara con el v7 DESPLEGADO
(260_v7_en_banco_llm.py) sobre EXACTAMENTE las mismas tareas que respondio cada
modelo. Mismas metricas y pruebas que 69_eval_llm.py:

  crudo      lo que dice el LLM; las respuestas invalidas se sustituyen por la
             mediana global (no se descartan: un modelo que no responde no gana)
  calibrado  el LLM por un factor aprendido SOLO en las 130 tareas de calibracion
  veredicto  sobre el IC95 por remuestreo de la diferencia de MdMRE con el v7

El v7 solo usa el texto de la tarea, asi que se compara en A (solo texto) y en B
(el LLM ademas ve ejemplos con horas reales). C (con historial del proyecto) se
deja fuera: ahi la referencia seria el v7 con anclaje, que es otro experimento.

Salidas: results/llm_bench/api_resumen.csv y api_tablas.md
"""
import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

s = importlib.util.spec_from_file_location("ev", "src/69_eval_llm.py")
ev = importlib.util.module_from_spec(s)
s.loader.exec_module(ev)

BENCH = Path("data/processed/llm_bench")
OUT = Path("results/llm_bench")
NOMBRE = {"openai/gpt-oss-120b": "GPT-OSS 120B (OpenAI)", "openai/gpt-oss-20b": "GPT-OSS 20B (OpenAI)",
          "qwen/qwen3.8-27b": "Qwen 3.8 27B", "gemini-3.5-flash": "Gemini 3.5 Flash (Google)",
          "llama3.1:8b": "Llama 3.1 8B", "qwen2.5:7b": "Qwen 2.5 7B", "gemma2:9b": "Gemma 2 9B",
          "mistral:7b": "Mistral 7B"}
MS_V7 = json.load(open("results/estimador_v7/servicio.json"))["ms_por_tarea_caliente"]


def corridas():
    filas = []
    for p in glob.glob(str(OUT / "raw" / "*.jsonl")):
        conjunto = Path(p).name.split("__")[0]
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r["conjunto"] = conjunto
            filas.append(r)
    return pd.DataFrame(filas)


def main():
    mu = pd.read_parquet(BENCH / "muestra.parquet").merge(
        pd.read_parquet(BENCH / "v7_muestra.parquet"), on="task_id")
    cal = pd.read_parquet(BENCH / "calibracion.parquet")
    runs = corridas()
    runs = runs[runs.condicion.isin(["A", "B"])]
    filas = []
    y_all = mu.effort_hours.to_numpy()
    filas.append(dict(condicion="-", modelo="PBE v7 (nuestro)", n=len(mu), validas=1.0,
                      seg_tarea=MS_V7 / 1000, **ev.metricas(y_all, mu.v7_horas.to_numpy())))
    filas.append(dict(condicion="-", modelo="PBE v7 · valor probable", n=len(mu), validas=1.0,
                      seg_tarea=MS_V7 / 1000, **ev.metricas(y_all, mu.v7_probable.to_numpy())))
    filas.append(dict(condicion="-", modelo="constante: mediana global", n=len(mu), validas=1.0,
                      **ev.metricas(y_all, mu.mediana_global.to_numpy())))

    for (backend, modelo, cond), g in runs[runs.conjunto == "muestra"].groupby(
            ["backend", "modelo", "condicion"]):
        d = mu.merge(g[["task_id", "valido", "horas", "segundos", "entrada", "salida"]],
                     on="task_id", how="inner")
        if len(d) < 30:
            continue
        y = d.effort_hours.to_numpy()
        ok = d.valido.fillna(False).astype(bool).to_numpy()
        crudo = np.where(ok, d.horas.fillna(0).to_numpy(), d.mediana_global.to_numpy())
        gc = runs[(runs.conjunto == "calibracion") & (runs.backend == backend)
                  & (runs.modelo == modelo) & (runs.condicion == cond)]
        cc = cal.merge(gc[gc.valido == True][["task_id", "horas"]], on="task_id")  # noqa: E712
        k = float(np.exp(np.median(np.log(cc.effort_hours / cc.horas)))) if len(cc) >= 20 else np.nan
        nuestro = d.v7_horas.to_numpy()
        base = dict(condicion=cond, modelo=NOMBRE.get(modelo, modelo),
                    proveedor="local (Ollama)" if backend == "ollama" else backend,
                    n=len(d), validas=float(ok.mean()),
                    seg_tarea=float(d.segundos.median()),
                    tokens=float((d.entrada.fillna(0) + d.salida.fillna(0)).median()),
                    v7_mismas_tareas_MdMRE=ev.metricas(y, nuestro)["MdMRE"],
                    v7_mismas_tareas_SA=ev.metricas(y, nuestro)["SA"])
        for version, p in (("crudo", crudo), ("calibrado", crudo * k if k == k else None)):
            if p is None:
                continue
            e_llm = np.abs(np.log(y) - np.log(np.clip(p, .01, None)))
            e_nos = np.abs(np.log(y) - np.log(nuestro))
            lo, hi = ev.bootstrap_dif(y, p, nuestro)
            filas.append(dict(**base, version=version, factor=k if version == "calibrado" else np.nan,
                              **ev.metricas(y, p), p_wilcoxon=wilcoxon(e_llm, e_nos).pvalue,
                              dif_MdMRE=f"[{lo:+.3f}, {hi:+.3f}]",
                              veredicto=("LLM mejor" if hi < 0 else "PBE mejor" if lo > 0 else "empate")))
    r = pd.DataFrame(filas)
    r.to_csv(OUT / "api_resumen.csv", index=False)

    f = lambda x, d=3: "—" if pd.isna(x) else f"{x:.{d}f}".replace(".", ",")
    pc = lambda x: "—" if pd.isna(x) else f"{100 * x:.1f} %".replace(".", ",")
    L = ["# PBE v7 frente a LLM", "",
         "601 tareas con horas reales (301 del bloque de prueba y 300 de proyectos externos), "
         "ninguna vista por el v7 al entrenar. Generado por `src/261_tablas_llm_api.py`.", ""]
    for cond, titulo in (("A", "Condición A · solo el texto de la tarea (como en producción)"),
                         ("B", "Condición B · el LLM ve además ejemplos con horas reales")):
        sub = r[(r.condicion == cond)]
        if sub.empty:
            continue
        L += [f"## {titulo}", "",
              "| modelo | proveedor | versión | n | válidas | MdMRE | PRED(25) | MAE (h) | SA | Spearman | frente a PBE | s/tarea |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        ref = r[r.condicion == "-"]
        for _, x in ref.iterrows():
            L.append(f"| **{x.modelo}** | — | — | {x.n} | 100 % | {f(x.MdMRE)} | {pc(x.PRED25)} | "
                     f"{f(x.MAE_h, 2)} | {f(x.SA)} | {f(x.spearman)} | — | "
                     f"{f(x.seg_tarea, 4) if pd.notna(x.get('seg_tarea')) else '—'} |")
        for _, x in sub.sort_values(["proveedor", "modelo", "version"]).iterrows():
            L.append(f"| {x.modelo} | {x.proveedor} | {x.version} | {x.n} | {pc(x.validas)} | "
                     f"{f(x.MdMRE)} | {pc(x.PRED25)} | {f(x.MAE_h, 2)} | {f(x.SA)} | "
                     f"{f(x.spearman)} | {x.veredicto} {x.dif_MdMRE} | {f(x.seg_tarea, 2)} |")
        L.append("")
    L += ["**Lectura.** «frente a PBE» es el veredicto sobre el IC95 de la diferencia de MdMRE "
          "(LLM − PBE) en las mismas tareas: negativo y sin cruzar el cero, gana el LLM; positivo, "
          "gana PBE. Las respuestas inválidas del LLM cuentan como la mediana global. «calibrado» "
          "multiplica al LLM por un factor aprendido en 130 tareas aparte. PBE responde en "
          f"{f(MS_V7 / 1000, 4)} s por tarea en CPU, sin conexión y sin coste por llamada.", ""]
    (OUT / "api_tablas.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
