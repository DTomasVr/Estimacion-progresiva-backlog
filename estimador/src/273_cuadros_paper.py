"""E-273 - The paper tables, generated from the result files.

No number is typed by hand: each table comes from the CSV and JSONL files in results/ and
is written as a LaTeX fragment that the paper includes with \\input. The table captions
and headers are in Spanish because the paper is (they are paper text, not code).

  tab_anclaje          Table I: the project total by closed tasks (E-275, E-161, E-233)
  tab_llm_estimacion   Table III: per task against LLMs, 601 unseen tasks (api_resumen.csv)
  tab_llm_generacion   Table IV: generated backlog against LLMs, 64 held-out projects
  (project total against LLMs: CSV only, the bench has too few projects for a table)

    python src/273_cuadros_paper.py --articulo   only the paper tables, from the CSV files

For the total, the LLM gets a factor learnt on the 130 calibration tasks as a ratio of
SUMS (the one that fits a total), and PBE uses its mean head, at cold start.
"""
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path("data/processed/llm_bench")
RAW = Path("results/llm_bench/raw")
OUT = Path("results/cuadros_paper")
# the LaTeX fragments go next to the CSV files; CUADROS_TEX sends them to the paper folder
TEX = Path(os.environ.get("CUADROS_TEX", OUT / "tex"))
API = {"openai/gpt-oss-120b": "GPT-OSS 120B", "openai/gpt-oss-20b": "GPT-OSS 20B", "qwen/qwen3.8-27b": "Qwen 3.8 27B"}
CARD_NAME = {"Qwen 3.8 27B": "Qwen3.8-27B"}   # name on the model card (guide V2, change 9)
MIN_TASKS = 8

# number formats of the Spanish paper: decimal comma
f = lambda x, d=3: "--" if pd.isna(x) else f"{x:.{d}f}".replace(".", "{,}")
pc = lambda x, d=1: "--" if pd.isna(x) else f"{100 * x:.{d}f}".replace(".", "{,}") + r"\,\%"


def table(name, caption, cols, head, rows, note="", sep="3pt"):
    L = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{tab:{name}}}",
         r"\footnotesize", rf"\setlength{{\tabcolsep}}{{{sep}}}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
         " & ".join(head) + r" \\", r"\midrule"]
    for x in rows:
        L.append(x if isinstance(x, str) else " & ".join(x) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L += [r"\par\vspace{2pt}\raggedright\scriptsize " + note]
    L += [r"\end{table}", ""]
    (TEX / f"tab_{name}.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"written {TEX / f'tab_{name}.tex'}")


def jsonl(p):
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()])


def estimation():
    """Table III."""
    r = pd.read_csv("results/llm_bench/api_resumen.csv")
    r.to_csv(OUT / "estimacion_por_tarea.csv", index=False)
    v7 = r[r.model == "PBE v7 (ours)"].iloc[0]
    ci = lambda s: s.strip("[]").replace(".", "{,}").replace("-", "$-$").replace(", ", ";\\,")
    rows = [[r"\textbf{Propuesta}", f(v7.MdMRE), pc(v7.PRED25), f(v7.spearman, 2), "--"]]
    a = r[(r.condition == "A") & r.provider.eq("groq")]
    n = {int(v7.n)}
    for name in API.values():
        rows.append(r"\midrule")
        for version in ("raw", "calibrated"):
            x = a[a.model.str.startswith(name) & (a.version == version)]
            if x.empty:
                continue
            x = x.iloc[0]
            n.add(int(x.n))
            rows.append([CARD_NAME.get(name, name) if version == "raw" else r"\quad calibrado", f(x.MdMRE),
                         pc(x.PRED25), f(x.spearman, 2), "[" + ci(x.diff_MdMRE) + "]"])
    assert len(n) == 1, n
    table("llm_estimacion",
          f"Estimación por tarea frente a LLM ({n.pop()} tareas no usadas para entrenar). IC95 $\\Delta$MdMRE: "
          "LLM menos propuesta.",
          "lcccc", ["", "MdMRE", "PRED(25)", r"$\rho$", r"IC95 $\Delta$MdMRE"], rows)


def total():
    """Project total against LLMs (CSV only)."""
    mu = pd.read_parquet(BENCH / "muestra.parquet")
    cal = pd.read_parquet(BENCH / "calibracion.parquet")
    cache = BENCH / "v7_media_muestra.parquet"
    if not cache.exists():
        s = importlib.util.spec_from_file_location("e6", "src/171_estimador_v6.py")
        e6 = importlib.util.module_from_spec(s)
        s.loader.exec_module(e6)
        est = e6.EstimadorV6(paquete=Path("models/estimador_pbe_v7.joblib"), dispositivo="cpu")
        t = pd.read_parquet("data/processed/pbe_effort_v4.parquet", columns=["task_id", "title"]).drop_duplicates("task_id")
        d = mu[["task_id"]].merge(t, on="task_id", how="left")
        d["v7_media"] = est._media_tarea(d.title.fillna("").astype(str).tolist())
        d[["task_id", "v7_media"]].to_parquet(cache)
    mu = mu.merge(pd.read_parquet(cache), on="task_id")
    size = mu.groupby("project").size()
    projs = size[size >= MIN_TASKS].index
    # the external block of the bench is Apache projects, which the project-level evaluation
    # always excludes (anomalous logged hours: the model overestimates them x1,4 to x14)
    base = mu[mu.project.isin(projs) & (mu.bloque == "prueba_d1")]
    projs = pd.Index(base.project.unique())
    out = []

    def measure(name, version, d, col):
        g = d.groupby("project")
        t, r = g[col].sum(), g.effort_hours.sum()
        e = np.abs(t - r) / r
        m = dict(system=name, version=version, projects=len(r), tasks=len(d), PRED25=float(np.mean(e <= .25)),
                 MdMRE=float(np.median(e)), factor2=float(np.mean(np.maximum(t / r, r / t) <= 2)))
        out.append(m)
        return m

    measure("PBE", "cold", base, "v7_media")
    for model, name in API.items():
        slug = re.sub(r"[^\w.-]", "_", model)
        pm, pcal = RAW / f"muestra__groq__{slug}__A.jsonl", RAW / f"calibracion__groq__{slug}__A.jsonl"
        if not pm.exists():
            continue
        g = jsonl(pm)
        d = base.merge(g[["task_id", "valido", "horas"]], on="task_id")
        d["llm"] = np.where(d.valido.fillna(False).astype(bool), d.horas.fillna(0), d.mediana_global)
        complete = d.groupby("project").size().reindex(projs).fillna(0) == size[projs]
        d = d[d.project.isin(complete[complete].index)]
        c = cal.merge(jsonl(pcal).query("valido == True")[["task_id", "horas"]], on="task_id")
        k = c.effort_hours.sum() / c.horas.sum()
        d["llm_cal"] = d.llm * k
        for version, col in (("raw", "llm"), ("calibrated", "llm_cal")):
            measure(name, version, d, col)
        measure("PBE (same projects as " + name + ")", "cold", d, "v7_media")
    pd.DataFrame(out).to_csv(OUT / "total_por_proyecto.csv", index=False)
    # with so few projects no table is made for the paper: only the CSV remains
    print(pd.DataFrame(out).round(3).to_string(index=False))


def generation():
    """Table IV."""
    p = Path("results/banco_generacion/resumen.csv")
    if not p.exists():
        print("missing results/banco_generacion/resumen.csv: run src/272_banco_generacion.py tablas first")
        return
    r = pd.read_csv(p)
    red = pd.read_csv("results/redundancia_tareas/resumen.csv").set_index("system").near_duplicates_090
    red.index = [("PBE (current)" if s == "pbe" else s + " zero-shot") for s in red.index]
    r["near_duplicates"] = r.system.map(red)
    r.to_csv(OUT / "generacion.csv", index=False)
    order = [("PBE (current)", r"\textbf{Propuesta}"), ("gemma2:9b zero-shot", "Gemma 2 9B"),
             ("llama3.1:8b zero-shot", "Llama 3.1 8B"), ("mistral:7b zero-shot", "Mistral 7B")]
    rows, ns = [], set()
    for key, name in order:
        x = r[r.system == key]
        if x.empty or x.iloc[0].n < 60:      # a half-finished run does not enter the table
            continue
        x = x.iloc[0]
        ns.add(int(x.n))
        rows.append([name, f(x.sF1), pc(x.coverage), f(x.tasks, 0), pc(x.near_duplicates),
                     f(x.sec_median, 1), "--" if key == "PBE (current)" else pc(x.recognises, 0)])
    assert len(ns) == 1, ns
    table("llm_generacion",
          f"Generación frente a LLM en {ns.pop()} proyectos retenidos. Duplicadas: coseno $\\geq 0{{,}}90$ con otra "
          "tarea; s: segundos por backlog en la misma GPU; Reconoce: proyectos que el LLM dice reconocer, solo un "
          "indicio de contaminación.",
          "lcccccc", ["", "sF1", "Cobertura", "Tareas", "Duplicadas", "s", "Reconoce"], rows, sep="2.5pt")


CASE = "dalpiaz:g02-federalspending"


def side_by_side():
    """One project, the first story and its first three tasks in each system (not in the paper).

    The case was chosen by the generation chat: the most readable Dalpiaz one, where PBE
    and GPT-OSS tie on sF1. The text comes from the JSONL files."""
    esc = lambda s: re.sub(r"([&%$#_])", r"\\\1", s)
    gen = Path("F:/Software Task generation Model/results/20_banco.jsonl")
    R = [json.loads(l) for l in open(gen, encoding="utf-8")]
    R = [r for r in R if r["sistema"] != "pbe"]
    for p in sorted(Path("results/banco_generacion").glob("*.jsonl")):
        R += [json.loads(l) for l in open(p, encoding="utf-8")]
    R = {(r["sistema"], r["condicion"]): r for r in R if r["caso"] == CASE}
    pp = pd.read_csv("results/banco_generacion/por_proyecto.csv")
    pp = pp[pp.case == CASE].set_index("system")
    s = importlib.util.spec_from_file_location("b272", "src/272_banco_generacion.py")
    b272 = importlib.util.module_from_spec(s)
    s.loader.exec_module(b272)
    C = b272.casos()
    inp = " ".join(C[C.parent_uid == CASE].entrada.iloc[0].split())
    order = [(("pbe", "actual"), "PBE (current)", r"\textbf{Propuesta}"),
             (("openai/gpt-oss-120b", "cero"), "openai/gpt-oss-120b zero-shot", "GPT-OSS 120B"),
             (("gemma2:9b", "cero"), "gemma2:9b zero-shot", "Gemma 2 9B"),
             (("llama3.1:8b", "cero"), "llama3.1:8b zero-shot", "Llama 3.1 8B"),
             (("mistral:7b", "cero"), "mistral:7b zero-shot", "Mistral 7B")]
    rows, out = [], []
    for key, sys_, name in order:
        if key not in R or sys_ not in pp.index:
            continue
        r, m = R[key], pp.loc[sys_]
        tasks = "; ".join(r["tareas"][0][:3])
        rows.append([name, esc(r["historias"][0]), esc(tasks), str(int(m.n_tasks)), f(m.sF1), f(m.seconds, 1)])
        out.append(dict(system=sys_, story=r["historias"][0], tasks=tasks, n_tasks=int(m.n_tasks),
                        sF1=float(m.sF1), seconds=float(m.seconds)))
    pd.DataFrame(out).assign(input=inp).to_csv(OUT / "lado_a_lado.csv", index=False)
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{El mismo proyecto retenido (portal de gasto federal, del conjunto de Dalpiaz) en cada sistema: "
         r"la primera historia generada y sus tres primeras tareas. «Tareas» es el total del backlog; el sF1 solo "
         r"puntúa las historias. Todos los sistemas reciben exactamente la misma entrada, que es esta descripción: "
         r"\emph{" + esc(inp) + r"} Las salidas se muestran en inglés, como las produce cada sistema.}",
         r"\label{tab:lado}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l p{0.33\textwidth} p{0.36\textwidth} c c c}", r"\toprule",
         r" & Primera historia & Primeras tareas & Tareas & sF1 & s \\", r"\midrule"]
    L += [" & ".join(x) + r" \\" for x in rows]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TEX / "tab_lado.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"written {TEX / 'tab_lado.tex'}")


# the deployed interval variant for each number of closed tasks
BAND = {1: "D · shrunk size band M=1500", 5: "D · shrunk size band M=200",
        10: "A2 · by k and size band (deployed)", 20: "A2 · by k and size band (deployed)"}


def anchoring():
    """Table I: the total by closed tasks, the curve at the centre of the proposal.

    It comes from E-275 (results/curva_anclaje_dos_criterios). The interval width with
    k >= 1 is that of the deployed variant for each k; at kick-off (k = 0), that of the
    single 80 % interval in results/banda_proyecto."""
    c = pd.read_csv("results/curva_anclaje_dos_criterios/resumen.csv")
    b = pd.read_csv("results/banda_que_cierra_antes/resumen.csv")
    b0 = pd.read_csv("results/banda_proyecto/resumen.csv")
    b0 = b0[(b0.scenario == "cold") & (b0.level == "80%") & (b0.method == "single")].iloc[0]
    v = lambda k, crit, col: float(c[(c.k == k) & (c.criterion == crit)][col].iloc[0])
    rows, out = [], []
    for k in (0, 1, 10, 20):   # guide V5: the k = 1 row is back (the text cites its 37,6 %)
        if k == 0:
            width, cov = float(b0.width_x), float(b0.coverage)
        else:
            x = b[(b.k == k) & (b.band == BAND[k])].iloc[0]
            width, cov = float(x.median_width), float(x.coverage)
        s = dict(closed=k, PRED25=v(k, "all", "PRED25"), factor2=v(k, "all", "factor2"),
                 MdMRE=v(k, "all", "MdMRE"), interval=width, coverage=cov)
        if k:
            s.update(PRED25_larger=v(k, "larger", "PRED25"), projects_larger=int(v(k, "larger", "projects")),
                     cold_larger=v(k, "larger", "PRED25_cold"),
                     PRED25_rule=v(k, "k or 20 % of the backlog", "PRED25"))
        out.append(s)
        cold_all = pc(v(1, "larger", "PRED25_cold"))
        larger = (cold_all + f" (114; {cold_all})" if not k
                  else pc(s["PRED25_larger"]) + f" ({s['projects_larger']}; {pc(s['cold_larger'])})")
        rows.append(["0" if k == 0 else str(k), larger, pc(s["PRED25"]), pc(s["factor2"]), f(s["MdMRE"]),
                     r"$\times$" + f(width, 2), f(cov, 2)])
    pd.DataFrame(out).to_csv(OUT / "anclaje.csv", index=False)
    table("anclaje",
          "El total del proyecto según las tareas cerradas, sobre los 114 "
          "proyectos (incluye los de $k$ o menos tareas, ya cerrados). «Solo mayores»: proyectos con más de $k$ "
          "tareas (cuántos; PRED(25) al arrancar). MdMRE: mediana del error relativo; $\\times$2: dentro de un "
          "factor 2; Cob.: cobertura del intervalo.",
          "lcccccc", ["Cerradas", "Solo mayores", "PRED(25)", r"$\times$2", "MdMRE", "Interv.", "Cob."], rows,
          sep="2pt")


def three_states():
    """How much the user estimates x how many tasks are closed, with ONE configuration (not in the paper)."""
    r = pd.read_csv("results/tres_estados/resumen.csv")
    meta = json.load(open("results/tres_estados/meta.json"))
    ks = (0, 3, 5, 10, 20)

    def cell(est, k):
        x = r[(r.user_estimates == est) & (r.closed == k)]
        if est != "0 %":
            x = x[x.lambda_u == 12]
        if k:
            x = x[x.lambda_c == 30]
        assert len(x) == 1, (est, k, len(x))
        return float(x.PRED25.iloc[0])

    rows, out = [], []
    for est, name in (("0 %", "nada"), ("100 %", "todo el backlog")):
        v = [cell(est, k) for k in ks]
        rows.append([name] + [pc(x) for x in v])
        out.append(dict(user_estimates=est, **{f"k{k}": x for k, x in zip(ks, v)}))
    pd.DataFrame(out).to_csv(OUT / "tres_estados.csv", index=False)
    table("tres_estados",
          f"PRED(25) del total cuando el usuario aporta sus estimaciones al arrancar y luego se cierran tareas "
          f"({meta['projects']} proyectos con estimaciones registradas; $\\lambda_u = 12$, $\\lambda'_c = 30$).",
          "lccccc", ["El usuario estima"] + [f"{k} cerr." for k in ks], rows)


def answers():
    """What the user answers at kick-off, treated as closed tasks (not in the paper).

    results/la_ultima_vez keeps the Spanish column names it was written with."""
    r = pd.read_csv("results/la_ultima_vez/como_cerradas.csv")
    cold = pd.read_csv("results/la_ultima_vez/previo_del_equipo.csv")
    cold = cold[(cold.cerradas == 0) & (cold.w == 0)].iloc[0]
    meta = json.load(open("results/la_ultima_vez/meta.json"))
    x = r[r.respuesta == "propia (acierta)"].set_index("tareas_preguntadas")
    rows = [["0 (arranque)", pc(cold.PRED25), pc(cold.factor2), f(cold.MdMRE)]]
    rows += [[str(m), pc(x.loc[m].PRED25), pc(x.loc[m].factor2), f(x.loc[m].MdMRE)] for m in (1, 3, 5, 10, 20)]
    x.reset_index().to_csv(OUT / "respuestas.csv", index=False)
    table("respuestas",
          f"El total cuando, al arrancar, el usuario da las horas de algunas tareas del backlog y estas se tratan "
          f"como cerradas ({meta['proyectos']} proyectos). Cota superior: el usuario acierta las horas.",
          "lccc", ["Tareas que da el usuario", "PRED(25)", r"$\times$2", "MdMRE"], rows)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    if "--articulo" in sys.argv:          # only the paper tables, from the CSV files in results/
        estimation()
        generation()
        anchoring()
        sys.exit()
    estimation()
    total()
    generation()
    side_by_side()
    anchoring()
    three_states()
    answers()
