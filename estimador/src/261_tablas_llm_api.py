"""Tables of the comparison of the v7 estimator against current LLMs through an API.

Reads the runs in results/llm_bench/raw (those of 259_banco_api.py and, as context, the
local ones from September) and compares them with the DEPLOYED v7
(260_v7_en_banco_llm.py) on EXACTLY the same tasks each model answered. Same metrics
and tests as 69_eval_llm.py:

  raw          what the LLM says; invalid answers are replaced by the global median
               (they are not dropped: a model that does not answer does not win)
  calibrated   the LLM times a factor learnt ONLY on the 130 calibration tasks
  verdict      on the bootstrap CI95 of the MdMRE difference with v7

v7 only uses the task text, so it is compared in A (text only) and in B (the LLM also
sees examples with real hours). C (with the project history) is left out: there the
reference would be v7 with anchoring, which is another experiment.

The raw run files (results/llm_bench/raw/*.jsonl) keep the field names they were
written with (condicion, modelo, valido, horas, segundos, entrada, salida).

Outputs: results/llm_bench/api_resumen.csv and api_tablas.md (Table III of the paper)
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
NAME = {"openai/gpt-oss-120b": "GPT-OSS 120B (OpenAI)", "openai/gpt-oss-20b": "GPT-OSS 20B (OpenAI)",
        "qwen/qwen3.8-27b": "Qwen 3.8 27B", "gemini-3.5-flash": "Gemini 3.5 Flash (Google)",
        "llama3.1:8b": "Llama 3.1 8B", "qwen2.5:7b": "Qwen 2.5 7B", "gemma2:9b": "Gemma 2 9B",
        "mistral:7b": "Mistral 7B"}
MS_V7 = json.load(open("results/estimador_v7/servicio.json"))["ms_por_tarea_caliente"]


def runs_table():
    rows = []
    for p in glob.glob(str(OUT / "raw" / "*.jsonl")):
        subset = Path(p).name.split("__")[0]
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r["conjunto"] = subset
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    mu = pd.read_parquet(BENCH / "muestra.parquet").merge(
        pd.read_parquet(BENCH / "v7_muestra.parquet"), on="task_id")
    cal = pd.read_parquet(BENCH / "calibracion.parquet")
    runs = runs_table()
    runs = runs[runs.condicion.isin(["A", "B"])]
    rows = []
    y_all = mu.effort_hours.to_numpy()
    rows.append(dict(condition="-", model="PBE v7 (ours)", n=len(mu), valid=1.0,
                     sec_per_task=MS_V7 / 1000, **ev.metricas(y_all, mu.v7_horas.to_numpy())))
    rows.append(dict(condition="-", model="PBE v7 · most likely value", n=len(mu), valid=1.0,
                     sec_per_task=MS_V7 / 1000, **ev.metricas(y_all, mu.v7_probable.to_numpy())))
    rows.append(dict(condition="-", model="constant: global median", n=len(mu), valid=1.0,
                     **ev.metricas(y_all, mu.mediana_global.to_numpy())))

    for (backend, model, cond), g in runs[runs.conjunto == "muestra"].groupby(
            ["backend", "modelo", "condicion"]):
        d = mu.merge(g[["task_id", "valido", "horas", "segundos", "entrada", "salida"]],
                     on="task_id", how="inner")
        if len(d) < 30:
            continue
        y = d.effort_hours.to_numpy()
        ok = d.valido.fillna(False).astype(bool).to_numpy()
        raw = np.where(ok, d.horas.fillna(0).to_numpy(), d.mediana_global.to_numpy())
        gc = runs[(runs.conjunto == "calibracion") & (runs.backend == backend)
                  & (runs.modelo == model) & (runs.condicion == cond)]
        cc = cal.merge(gc[gc.valido == True][["task_id", "horas"]], on="task_id")  # noqa: E712
        k = float(np.exp(np.median(np.log(cc.effort_hours / cc.horas)))) if len(cc) >= 20 else np.nan
        ours = d.v7_horas.to_numpy()
        base = dict(condition=cond, model=NAME.get(model, model),
                    provider="local (Ollama)" if backend == "ollama" else backend,
                    n=len(d), valid=float(ok.mean()),
                    sec_per_task=float(d.segundos.median()),
                    tokens=float((d.entrada.fillna(0) + d.salida.fillna(0)).median()),
                    v7_same_tasks_MdMRE=ev.metricas(y, ours)["MdMRE"],
                    v7_same_tasks_SA=ev.metricas(y, ours)["SA"])
        for version, p in (("raw", raw), ("calibrated", raw * k if k == k else None)):
            if p is None:
                continue
            e_llm = np.abs(np.log(y) - np.log(np.clip(p, .01, None)))
            e_ours = np.abs(np.log(y) - np.log(ours))
            lo, hi = ev.bootstrap_dif(y, p, ours)
            rows.append(dict(**base, version=version, factor=k if version == "calibrated" else np.nan,
                             **ev.metricas(y, p), p_wilcoxon=wilcoxon(e_llm, e_ours).pvalue,
                             diff_MdMRE=f"[{lo:+.3f}, {hi:+.3f}]",
                             verdict=("LLM better" if hi < 0 else "PBE better" if lo > 0 else "tie")))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "api_resumen.csv", index=False)

    f = lambda x, d=3: "—" if pd.isna(x) else f"{x:.{d}f}"
    pc = lambda x: "—" if pd.isna(x) else f"{100 * x:.1f} %"
    L = ["# PBE v7 against LLMs", "",
         "601 tasks with real hours (301 from the test block and 300 from external projects), "
         "none seen by v7 in training. Generated by `src/261_tablas_llm_api.py`.", ""]
    for cond, title in (("A", "Condition A · task text only (as in production)"),
                        ("B", "Condition B · the LLM also sees examples with real hours")):
        sub = r[(r.condition == cond)]
        if sub.empty:
            continue
        L += [f"## {title}", "",
              "| model | provider | version | n | valid | MdMRE | PRED(25) | MAE (h) | SA | Spearman | against PBE | s/task |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        ref = r[r.condition == "-"]
        for _, x in ref.iterrows():
            L.append(f"| **{x.model}** | — | — | {x.n} | 100 % | {f(x.MdMRE)} | {pc(x.PRED25)} | "
                     f"{f(x.MAE_h, 2)} | {f(x.SA)} | {f(x.spearman)} | — | "
                     f"{f(x.sec_per_task, 4) if pd.notna(x.get('sec_per_task')) else '—'} |")
        for _, x in sub.sort_values(["provider", "model", "version"]).iterrows():
            L.append(f"| {x.model} | {x.provider} | {x.version} | {x.n} | {pc(x.valid)} | "
                     f"{f(x.MdMRE)} | {pc(x.PRED25)} | {f(x.MAE_h, 2)} | {f(x.SA)} | "
                     f"{f(x.spearman)} | {x.verdict} {x.diff_MdMRE} | {f(x.sec_per_task, 2)} |")
        L.append("")
    L += ["**Reading.** 'against PBE' is the verdict on the CI95 of the MdMRE difference "
          "(LLM − PBE) on the same tasks: negative and not crossing zero, the LLM wins; positive, "
          "PBE wins. Invalid LLM answers count as the global median. 'calibrated' multiplies the "
          "LLM by a factor learnt on 130 separate tasks. PBE answers in "
          f"{f(MS_V7 / 1000, 4)} s per task on CPU, offline and with no cost per call.", ""]
    (OUT / "api_tablas.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
