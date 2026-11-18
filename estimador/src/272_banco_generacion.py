"""E-272 - GENERATION bench: the current PBE pipeline against LLMs, for the paper.

The bench of the generation repository (results/20_banco.jsonl, 01/10/2026) measured
PBE BEFORE the anchor stories; its local LLM arms are still valid. What is missing is
redone here, without touching that file:

  pbe          the current pipeline (F:/PBE-produccion, 07/11/2026) on the same 64
               held-out projects (4 from Dalpiaz and 60 Jira epics from the test
               block), with the same protocol: wall-clock time to a complete backlog
  groq <model> a current LLM through an API, with no examples: one call for the
               stories and one per story for its tasks, plus the contamination probe
  tablas       scores everything with the same metric (set sF1 over the STORIES
               against the project's real ones) and writes the CSV files

Everything is appended to JSONL: an interrupted run resumes where it stopped. The raw
JSONL files keep the field names they were written with (sistema, condicion, caso,
fuente, segundos, historias, tareas, n_tareas, sonda); the CSV files are in English.

Outputs: results/banco_generacion/por_proyecto.csv and resumen.csv (Table IV of the paper)
"""
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

GEN = Path("F:/Software Task generation Model")
PROD = Path("F:/PBE-produccion/generacion")
OUT = Path("results/banco_generacion")
N_JIRA, K_STORIES, SEED = 60, 12, 20260922
# condition labels of the raw runs, in English for the CSV files
CONDITION = {"actual": "current", "cero": "zero-shot", "ejemplos": "examples",
             "01/10, sin anclas": "01/10, no anchors"}

P_STORIES = ("You are a product owner. Write exactly {k} user stories for the following project, "
             "each on its own line, in the form 'As a <role>, I want <goal>, so that <benefit>.' "
             "Output only the {k} lines.\n\nProject: {desc}")
P_TASKS = ("Break this user story into the concrete implementation tasks a development team "
           "would create. One short task title per line, no numbering, no explanations.\n\n"
           "User story: {historia}")
P_PROBE = ("Do you recognise this specific software project from your training data? "
           "Answer only YES or NO, then its name if yes.\n\nProject: {desc}")


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def casos():
    """The same held-out projects as src/20_banco_llms.py of the generation repository.

    The columns (parent_uid, entrada, refs, proj, fuente) come from its parquet files."""
    D = pd.read_parquet(GEN / "data/processed/s2s_o2o_dalpiaz_test.parquet")
    T = pd.read_parquet(GEN / "data/processed/s2s_o2o_test.parquet")
    T = T[(T.estilo == "historia") & ~T.proj.str.startswith("dalpiaz:")]
    rows = []
    for d, source in [(D, "dalpiaz"), (T, "jira")]:
        g = d.groupby("parent_uid").agg(entrada=("entrada", "first"), refs=("objetivo", list),
                                        proj=("proj", "first")).reset_index()
        if source == "jira":
            g = g[g.refs.str.len() >= 3]
            g = g.sample(min(N_JIRA, len(g)), random_state=SEED)
        rows.append(g.assign(fuente=source))
    return pd.concat(rows, ignore_index=True)


def lines(text, n=None):
    out = []
    for l in text.split("\n"):
        l = re.sub(r"^\s*[-*\d.)\s]+", "", l).strip(" .")
        if len(l) >= 10:
            out.append(l)
    return out[:n] if n else out


def done(path):
    return {json.loads(l)["caso"] for l in open(path, encoding="utf-8")} if path.exists() else set()


def arm_pbe():
    import torch
    C = casos()
    dest = OUT / "pbe_actual.jsonl"
    already = done(dest)
    print(f"{len(C)} projects ({C.fuente.value_counts().to_dict()}) · {len(already)} already done", flush=True)
    t0 = time.time()
    os.environ.setdefault("PBE_BASE", str(PROD))
    pl = module("pl", PROD / "src" / "19_pipeline.py")
    p = pl.Pipeline()
    p(C.entrada.iloc[0], k_historias=4)          # warm-up: so that CUDA does not land on case 1
    print(f"pipeline loaded and warm on {pl.DISPOSITIVO} · {time.time() - t0:.0f} s", flush=True)
    with open(dest, "a", encoding="utf-8") as fh:
        for i, c in enumerate(C.itertuples(), 1):
            if c.parent_uid in already:
                continue
            torch.cuda.synchronize()
            t = time.time()
            r = p(c.entrada, k_historias=K_STORIES)
            torch.cuda.synchronize()
            dt = time.time() - t
            fh.write(json.dumps(dict(sistema="pbe", condicion="actual", caso=c.parent_uid, fuente=c.fuente,
                                     segundos=round(dt, 2), historias=[h["historia"] for h in r["historias"]],
                                     tareas=[h["tareas"] for h in r["historias"]], n_tareas=r["n_tareas"]),
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i % 8 == 0 or i == len(C):
                print(f"  {i}/{len(C)} · last {dt:.1f} s · total {time.time() - t0:.0f} s", flush=True)
    print("done")


def arm_groq(model):
    run = module("run", "src/68_llm_runner.py")
    url = "https://api.groq.com/openai/v1/chat/completions"

    def call(prompt):
        # after a long rate-limit wait the open connection expires and the next call
        # died with RemoteDisconnected, losing the whole project
        import requests
        for attempt in range(6):
            try:
                return run._compatible_openai(url, "GROQ_API_KEY", model, "You are a helpful assistant.",
                                              prompt, {"seed": 0})
            except requests.exceptions.ConnectionError:
                run.SESION = requests.Session()
                time.sleep(5 * (attempt + 1))
        raise run.CuotaAgotada("the connection drops again and again")

    C = casos()
    dest = OUT / f"groq__{re.sub(r'[^\w.-]', '_', model)}.jsonl"
    already = done(dest)
    print(f"{model}: {len(C)} projects · {len(already)} already done", flush=True)
    t0 = time.time()
    with open(dest, "a", encoding="utf-8") as fh:
        for i, c in enumerate(C.itertuples(), 1):
            if c.parent_uid in already:
                continue
            try:
                t = time.time()
                txt, u = call(P_STORIES.format(k=K_STORIES, desc=c.entrada))
                stories = lines(txt, K_STORIES)
                tok = (u.get("entrada") or 0) + (u.get("salida") or 0)
                tasks = []
                for h in stories:
                    tt, u = call(P_TASKS.format(historia=h))
                    tasks.append(lines(tt))
                    tok += (u.get("entrada") or 0) + (u.get("salida") or 0)
                dt = time.time() - t
                probe, _ = call(P_PROBE.format(desc=c.entrada))
            except run.CuotaAgotada as e:
                print(f"stopped at {i - 1}/{len(C)}: {e}; it resumes when relaunched", flush=True)
                sys.exit(3)
            fh.write(json.dumps(dict(sistema=model, condicion="cero", caso=c.parent_uid, fuente=c.fuente,
                                     segundos=round(dt, 2), historias=stories, tareas=tasks,
                                     n_tareas=sum(len(x) for x in tasks), tokens=tok, sonda=probe[:120]),
                                ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  {i}/{len(C)} · {dt:.0f} s · {tok} tokens · total {time.time() - t0:.0f} s", flush=True)
    print("done")


def tables():
    from sentence_transformers import SentenceTransformer
    ev = module("ev", GEN / "src" / "11_eval_gen.py")
    pat = re.compile(r"as\s+(?:an?|the)\s+[^,.]{2,60}[,;]?\s+i\s*(?:'d|\s+would)?\s*"
                     r"(?:want|would\s+like|need|wish|should|must|am\s+able)", re.I)
    R = [json.loads(l) for l in open(GEN / "results/20_banco.jsonl", encoding="utf-8")]
    R = [r for r in R if not (r["sistema"] == "pbe")]
    for f in sorted(OUT.glob("*.jsonl")):
        R += [json.loads(l) for l in open(f, encoding="utf-8")]
    R = pd.DataFrame(R)
    before = [json.loads(l) for l in open(GEN / "results/20_banco.jsonl", encoding="utf-8")]
    before = pd.DataFrame([r for r in before if r["sistema"] == "pbe"]).assign(condicion="01/10, sin anclas")
    R = pd.concat([R, before], ignore_index=True)
    # raw field names -> English column names
    R = R.rename(columns={"sistema": "system", "caso": "case", "fuente": "source", "segundos": "seconds",
                          "n_tareas": "n_tasks", "historias": "stories", "sonda": "probe"})
    cond = R.condicion.map(CONDITION).fillna(R.condicion)
    R["system"] = np.where(R.system == "pbe", "PBE (" + cond + ")", R.system + " " + cond)
    C = casos()
    ref = dict(zip(C.parent_uid, C.refs))
    R = R[R.case.isin(ref)].reset_index(drop=True)
    sb = SentenceTransformer(ev.SBERT, device="cuda")
    s = ev.puntuar(sb, [list(x) for x in R.stories], [list(ref[c]) for c in R.case])
    R["sF1"] = s["mat"][:, 0]
    R["coverage"] = s["mat"][:, 5]
    R["format"] = [np.mean([bool(pat.match(t)) for t in h]) if len(h) else 0 for h in R.stories]
    R["recognises"] = R.get("probe", pd.Series([""] * len(R))).fillna("").str.strip().str.lower().str.startswith(("yes", "si"))
    R[["system", "case", "source", "seconds", "n_tasks", "sF1", "coverage", "format", "recognises"]].to_csv(
        OUT / "por_proyecto.csv", index=False)
    g = R.groupby("system").agg(n=("case", "size"), sec_median=("seconds", "median"),
                                stories=("stories", lambda x: np.mean([len(i) for i in x])),
                                tasks=("n_tasks", "mean"), sF1=("sF1", "mean"), coverage=("coverage", "mean"),
                                format=("format", "mean"), recognises=("recognises", "mean")).reset_index()
    ours = "PBE (current)"
    P = R.pivot_table(index="case", columns="system", values="sF1")
    rng = np.random.default_rng(SEED)
    pair = {}
    for sys_ in P.columns:
        if sys_ == ours or ours not in P:
            continue
        d = P[[ours, sys_]].dropna()
        diff = (d[sys_] - d[ours]).to_numpy()
        b = rng.choice(diff, (4000, len(diff))).mean(1)
        lo, hi = np.quantile(b, [.025, .975])
        pair[sys_] = dict(n_paired=len(d), diff_sF1=float(diff.mean()), ci95_lo=float(lo), ci95_hi=float(hi),
                          verdict="other wins" if lo > 0 else "PBE wins" if hi < 0 else "tie")
    g = g.merge(pd.DataFrame(pair).T.rename_axis("system").reset_index(), on="system", how="left")
    g.to_csv(OUT / "resumen.csv", index=False)
    pd.set_option("display.width", 250)
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    {"pbe": arm_pbe, "tablas": tables}.get(sys.argv[1], lambda: arm_groq(sys.argv[2]))()
