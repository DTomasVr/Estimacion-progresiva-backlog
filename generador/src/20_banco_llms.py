"""Bench: our pipeline vs local LLMs, on the same held-out projects, same GPU.

The thesis claim is speed with acceptable fidelity, so the headline number is
WALL-CLOCK TIME TO A COMPLETE BACKLOG (stories + their tasks), measured per
project, plus the story-set quality under the same metric.

LLM arms (Ollama, same machine): one call for the stories, then one call per
story for its tasks -- exactly what Agix does today. Two conditions:
  cero      zero-shot, what Agix does now
  ejemplos  same retrieved neighbours our model gets (the hard baseline)

Contamination probe: public Jira/Dalpiaz text may be in the LLMs' training
data, so each project is also asked "do you recognise this project?" and the
answer is recorded next to its score.

Results are appended as JSON lines, so the run resumes after an interruption.
usage: python src/20_banco_llms.py [N_PROYECTOS]
"""
import importlib.util
import json
import os
import re
import sys
import time
import urllib.request

import numpy as np
import pandas as pd
import torch

BASE = r"F:\Software Task generation Model"
OUT = f"{BASE}/results/20_banco.jsonl"
OLLAMA = "http://127.0.0.1:11434/api/generate"
MODELOS_LLM = ["llama3.1:8b", "mistral:7b", "gemma2:9b"]
K_HISTORIAS = 12
SEED = 20260922

spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
spec2 = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ev)

P_HIST = ("You are a product owner. Write exactly {k} user stories for the following project, "
          "each on its own line, in the form 'As a <role>, I want <goal>, so that <benefit>.' "
          "Output only the {k} lines.\n\n{ctx}Project: {desc}")
P_TAREAS = ("Break this user story into the concrete implementation tasks a development team "
            "would create. One short task title per line, no numbering, no explanations.\n\n"
            "User story: {historia}")
P_SONDA = ("Do you recognise this specific software project from your training data? "
           "Answer only YES or NO, then its name if yes.\n\nProject: {desc}")


def ollama(modelo, prompt, num_predict=600, intentos=3):
    """retries: a transient HTTP 500 while Ollama loads a model killed a 2 h run once"""
    body = json.dumps({"model": modelo, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": num_predict}}).encode()
    for k in range(intentos):
        try:
            req = urllib.request.Request(OLLAMA, body, {"Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=900))
            return r["response"].strip(), r.get("eval_count", 0)
        except Exception as e:
            if k == intentos - 1:
                raise
            print(f"  aviso: {modelo} fallo ({e}); reintento {k+1}/{intentos-1} en 20 s", flush=True)
            time.sleep(20)


def lineas(texto, n=None):
    out = []
    for l in texto.split("\n"):
        l = re.sub(r"^\s*[-*\d.)\s]+", "", l).strip(" .")
        if len(l) >= 10:
            out.append(l)
    return out[:n] if n else out


def casos(n):
    """held-out projects: Dalpiaz (4, never trained on) + epics from the test split"""
    D = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_dalpiaz_test.parquet")
    T = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_test.parquet")
    T = T[(T.estilo == "historia") & ~T.proj.str.startswith("dalpiaz:")]   # they are the dalpiaz arm
    filas = []
    for d, fuente in [(D, "dalpiaz"), (T, "jira")]:
        g = d.groupby("parent_uid").agg(entrada=("entrada", "first"), refs=("objetivo", list),
                                        proj=("proj", "first")).reset_index()
        if fuente == "jira":
            # >=3 reference stories: with >=4 only 44 test epics qualify, with >=2 the
            # references get too thin to compare against
            g = g[g.refs.str.len() >= 3]
            g = g.sample(min(n, len(g)), random_state=SEED)
        filas.append(g.assign(fuente=fuente))
    return pd.concat(filas, ignore_index=True)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    C = casos(n)
    hechos = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            r = json.loads(l)
            hechos.add((r["sistema"], r["condicion"], r["caso"]))
        print(f"reanudando: {len(hechos)} corridas ya hechas")
    print(f"{len(C)} proyectos ({C.fuente.value_counts().to_dict()})", flush=True)
    fh = open(OUT, "a", encoding="utf-8")

    for mm in MODELOS_LLM:            # our arm must not share the GPU with a loaded LLM
        try:
            urllib.request.urlopen(urllib.request.Request(
                OLLAMA, json.dumps({"model": mm, "keep_alive": 0}).encode(),
                {"Content-Type": "application/json"}), timeout=120).read()
        except Exception as e:
            print(f"  aviso: no pude descargar {mm}: {e}", flush=True)
    time.sleep(5)
    libre = torch.cuda.mem_get_info()[0] / 2**30
    print(f"VRAM libre antes de medir: {libre:.2f} GB", flush=True)

    # ---------------------------------------------------------- our system
    p = pl.Pipeline()
    print(f"pipeline cargado en {p.carga_s}s", flush=True)
    p(C.entrada.iloc[0], k_historias=4)      # warm-up: CUDA init must not land on case 1
    for _, c in C.iterrows():
        if ("agix", "-", c.parent_uid) in hechos:
            continue
        torch.cuda.synchronize()
        t = time.time()
        r = p(c.entrada, k_historias=K_HISTORIAS)
        torch.cuda.synchronize()
        dt = time.time() - t
        fh.write(json.dumps(dict(sistema="agix", condicion="-", caso=c.parent_uid, fuente=c.fuente,
                                 segundos=round(dt, 2), historias=[h["historia"] for h in r["historias"]],
                                 tareas=[h["tareas"] for h in r["historias"]],
                                 n_tareas=r["n_tareas"], tiempos=r["tiempos"]), ensure_ascii=False) + "\n")
        fh.flush()
    # the LLMs get the SAME retrieved neighbours, computed with the pipeline that is
    # already in memory (re-loading SBERT after freeing the pipeline crashes the process)
    ctx, _ = p._vecinos(C.entrada.tolist(), p.EA, p.A, "hijos")
    ctxs = dict(zip(C.parent_uid, ctx))
    print(f"[{time.strftime('%H:%M')}] brazo agix listo; contextos para los LLM listos", flush=True)
    del p
    torch.cuda.empty_cache()

    for modelo in MODELOS_LLM:
        for cond in ["cero", "ejemplos"]:
            for _, c in C.iterrows():
                if (modelo, cond, c.parent_uid) in hechos:
                    continue
                pre = "" if cond == "cero" else f"Similar projects and their stories:\n{ctxs[c.parent_uid]}\n\n"
                t = time.time()
                txt, tk1 = ollama(modelo, P_HIST.format(k=K_HISTORIAS, ctx=pre, desc=c.entrada))
                hist = lineas(txt, K_HISTORIAS)
                tareas, tk2 = [], 0
                for h in hist:
                    tt, k = ollama(modelo, P_TAREAS.format(historia=h), num_predict=300)
                    tareas.append(lineas(tt))
                    tk2 += k
                dt = time.time() - t
                sonda, _ = ollama(modelo, P_SONDA.format(desc=c.entrada), num_predict=30)
                fh.write(json.dumps(dict(sistema=modelo, condicion=cond, caso=c.parent_uid,
                                         fuente=c.fuente, segundos=round(dt, 2), historias=hist,
                                         tareas=tareas, n_tareas=sum(len(x) for x in tareas),
                                         tokens=tk1 + tk2, sonda=sonda[:120]), ensure_ascii=False) + "\n")
                fh.flush()
            print(f"[{time.strftime('%H:%M')}] {modelo} {cond} listo", flush=True)
        ollama(modelo, "", 1) if False else None
        urllib.request.urlopen(urllib.request.Request(  # free the VRAM between models
            OLLAMA, json.dumps({"model": modelo, "keep_alive": 0}).encode(),
            {"Content-Type": "application/json"}), timeout=120).read()
    fh.close()
    print("banco completo", flush=True)


if __name__ == "__main__":
    main()
