"""E-272 - Banco de GENERACION: la tuberia actual de Agix frente a LLM, para el paper.

El banco del repositorio de generacion (results/20_banco.jsonl, 01/10/2026) midio a
Agix ANTES de las historias ancla; sus brazos de LLM locales siguen valiendo. Aqui
se rehace lo que falta, sin tocar aquel archivo:

  agix          la tuberia actual (F:/Agix-produccion, 07/11/2026) sobre los mismos
                64 proyectos retenidos (4 de Dalpiaz y 60 epicas de Jira del bloque
                de prueba), con el mismo protocolo: tiempo de reloj a backlog completo
  groq <modelo> un LLM actual por API, sin ejemplos: una llamada para las historias
                y una por historia para sus tareas, mas la sonda de contaminacion
  tablas        puntua todo con la misma metrica (sF1 de conjunto sobre las HISTORIAS
                frente a las reales del proyecto) y deja los CSV

Todo se anexa en JSONL: una corrida cortada se retoma donde quedo.
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
PROD = Path("F:/Agix-produccion/generacion")
OUT = Path("results/banco_generacion")
N_JIRA, K_HISTORIAS, SEED = 60, 12, 20260922

P_HIST = ("You are a product owner. Write exactly {k} user stories for the following project, "
          "each on its own line, in the form 'As a <role>, I want <goal>, so that <benefit>.' "
          "Output only the {k} lines.\n\nProject: {desc}")
P_TAREAS = ("Break this user story into the concrete implementation tasks a development team "
            "would create. One short task title per line, no numbering, no explanations.\n\n"
            "User story: {historia}")
P_SONDA = ("Do you recognise this specific software project from your training data? "
           "Answer only YES or NO, then its name if yes.\n\nProject: {desc}")


def modulo(nombre, ruta):
    s = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def casos():
    """Los mismos proyectos retenidos que src/20_banco_llms.py del repositorio de generacion."""
    D = pd.read_parquet(GEN / "data/processed/s2s_o2o_dalpiaz_test.parquet")
    T = pd.read_parquet(GEN / "data/processed/s2s_o2o_test.parquet")
    T = T[(T.estilo == "historia") & ~T.proj.str.startswith("dalpiaz:")]
    filas = []
    for d, fuente in [(D, "dalpiaz"), (T, "jira")]:
        g = d.groupby("parent_uid").agg(entrada=("entrada", "first"), refs=("objetivo", list),
                                        proj=("proj", "first")).reset_index()
        if fuente == "jira":
            g = g[g.refs.str.len() >= 3]
            g = g.sample(min(N_JIRA, len(g)), random_state=SEED)
        filas.append(g.assign(fuente=fuente))
    return pd.concat(filas, ignore_index=True)


def lineas(texto, n=None):
    out = []
    for l in texto.split("\n"):
        l = re.sub(r"^\s*[-*\d.)\s]+", "", l).strip(" .")
        if len(l) >= 10:
            out.append(l)
    return out[:n] if n else out


def hechos(ruta):
    return {json.loads(l)["caso"] for l in open(ruta, encoding="utf-8")} if ruta.exists() else set()


def brazo_agix():
    import torch
    C = casos()
    destino = OUT / "agix_actual.jsonl"
    ya = hechos(destino)
    print(f"{len(C)} proyectos ({C.fuente.value_counts().to_dict()}) · {len(ya)} ya hechos", flush=True)
    t0 = time.time()
    os.environ.setdefault("AGIX_BASE", str(PROD))
    pl = modulo("pl", PROD / "src" / "19_pipeline.py")
    p = pl.Pipeline()
    p(C.entrada.iloc[0], k_historias=4)          # calentamiento: que CUDA no caiga en el caso 1
    print(f"tuberia cargada y caliente en {pl.DISPOSITIVO} · {time.time() - t0:.0f} s", flush=True)
    with open(destino, "a", encoding="utf-8") as fh:
        for i, c in enumerate(C.itertuples(), 1):
            if c.parent_uid in ya:
                continue
            torch.cuda.synchronize()
            t = time.time()
            r = p(c.entrada, k_historias=K_HISTORIAS)
            torch.cuda.synchronize()
            dt = time.time() - t
            fh.write(json.dumps(dict(sistema="agix", condicion="actual", caso=c.parent_uid, fuente=c.fuente,
                                     segundos=round(dt, 2), historias=[h["historia"] for h in r["historias"]],
                                     tareas=[h["tareas"] for h in r["historias"]], n_tareas=r["n_tareas"]),
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i % 8 == 0 or i == len(C):
                print(f"  {i}/{len(C)} · ultimo {dt:.1f} s · total {time.time() - t0:.0f} s", flush=True)
    print("listo")


def brazo_groq(modelo):
    run = modulo("run", "src/68_llm_runner.py")
    url = "https://api.groq.com/openai/v1/chat/completions"

    def llama(prompt):
        # tras una espera larga por limite de tasa la conexion abierta caduca y la
        # siguiente llamada moria con RemoteDisconnected, perdiendo el proyecto entero
        import requests
        for intento in range(6):
            try:
                return run._compatible_openai(url, "GROQ_API_KEY", modelo, "You are a helpful assistant.",
                                              prompt, {"seed": 0})
            except requests.exceptions.ConnectionError:
                run.SESION = requests.Session()
                time.sleep(5 * (intento + 1))
        raise run.CuotaAgotada("la conexion se corta una y otra vez")

    C = casos()
    destino = OUT / f"groq__{re.sub(r'[^\w.-]', '_', modelo)}.jsonl"
    ya = hechos(destino)
    print(f"{modelo}: {len(C)} proyectos · {len(ya)} ya hechos", flush=True)
    t0 = time.time()
    with open(destino, "a", encoding="utf-8") as fh:
        for i, c in enumerate(C.itertuples(), 1):
            if c.parent_uid in ya:
                continue
            try:
                t = time.time()
                txt, u = llama(P_HIST.format(k=K_HISTORIAS, desc=c.entrada))
                hist = lineas(txt, K_HISTORIAS)
                tok = (u.get("entrada") or 0) + (u.get("salida") or 0)
                tareas = []
                for h in hist:
                    tt, u = llama(P_TAREAS.format(historia=h))
                    tareas.append(lineas(tt))
                    tok += (u.get("entrada") or 0) + (u.get("salida") or 0)
                dt = time.time() - t
                sonda, _ = llama(P_SONDA.format(desc=c.entrada))
            except run.CuotaAgotada as e:
                print(f"cortado en {i - 1}/{len(C)}: {e}; se retoma al relanzar", flush=True)
                sys.exit(3)
            fh.write(json.dumps(dict(sistema=modelo, condicion="cero", caso=c.parent_uid, fuente=c.fuente,
                                     segundos=round(dt, 2), historias=hist, tareas=tareas,
                                     n_tareas=sum(len(x) for x in tareas), tokens=tok, sonda=sonda[:120]),
                                ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  {i}/{len(C)} · {dt:.0f} s · {tok} tokens · total {time.time() - t0:.0f} s", flush=True)
    print("listo")


def tablas():
    from sentence_transformers import SentenceTransformer
    ev = modulo("ev", GEN / "src" / "11_eval_gen.py")
    pat = re.compile(r"as\s+(?:an?|the)\s+[^,.]{2,60}[,;]?\s+i\s*(?:'d|\s+would)?\s*"
                     r"(?:want|would\s+like|need|wish|should|must|am\s+able)", re.I)
    R = [json.loads(l) for l in open(GEN / "results/20_banco.jsonl", encoding="utf-8")]
    R = [r for r in R if not (r["sistema"] == "agix")]
    for f in sorted(OUT.glob("*.jsonl")):
        R += [json.loads(l) for l in open(f, encoding="utf-8")]
    R = pd.DataFrame(R)
    antes = [json.loads(l) for l in open(GEN / "results/20_banco.jsonl", encoding="utf-8")]
    antes = pd.DataFrame([r for r in antes if r["sistema"] == "agix"]).assign(condicion="01/10, sin anclas")
    R = pd.concat([R, antes], ignore_index=True)
    R["sistema"] = np.where(R.sistema == "agix", "Agix (" + R.condicion + ")", R.sistema + " " + R.condicion)
    C = casos()
    ref = dict(zip(C.parent_uid, C.refs))
    R = R[R.caso.isin(ref)].reset_index(drop=True)
    sb = SentenceTransformer(ev.SBERT, device="cuda")
    s = ev.puntuar(sb, [list(x) for x in R.historias], [list(ref[c]) for c in R.caso])
    R["sF1"] = s["mat"][:, 0]
    R["cobertura"] = s["mat"][:, 5]
    R["formato"] = [np.mean([bool(pat.match(t)) for t in h]) if len(h) else 0 for h in R.historias]
    R["reconoce"] = R.get("sonda", pd.Series([""] * len(R))).fillna("").str.strip().str.lower().str.startswith(("yes", "si"))
    R[["sistema", "caso", "fuente", "segundos", "n_tareas", "sF1", "cobertura", "formato", "reconoce"]].to_csv(
        OUT / "por_proyecto.csv", index=False)
    g = R.groupby("sistema").agg(n=("caso", "size"), seg_mediana=("segundos", "median"),
                                 historias=("historias", lambda x: np.mean([len(i) for i in x])),
                                 tareas=("n_tareas", "mean"), sF1=("sF1", "mean"), cobertura=("cobertura", "mean"),
                                 formato=("formato", "mean"), reconoce=("reconoce", "mean")).reset_index()
    nuestro = "Agix (actual)"
    P = R.pivot_table(index="caso", columns="sistema", values="sF1")
    rng = np.random.default_rng(SEED)
    par = {}
    for sis in P.columns:
        if sis == nuestro or nuestro not in P:
            continue
        d = P[[nuestro, sis]].dropna()
        dif = (d[sis] - d[nuestro]).to_numpy()
        b = rng.choice(dif, (4000, len(dif))).mean(1)
        lo, hi = np.quantile(b, [.025, .975])
        par[sis] = dict(n_pareado=len(d), dif_sF1=float(dif.mean()), ic95_lo=float(lo), ic95_hi=float(hi),
                        veredicto="gana el otro" if lo > 0 else "gana Agix" if hi < 0 else "empate")
    g = g.merge(pd.DataFrame(par).T.rename_axis("sistema").reset_index(), on="sistema", how="left")
    g.to_csv(OUT / "resumen.csv", index=False)
    pd.set_option("display.width", 250)
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    {"agix": brazo_agix, "tablas": tablas}.get(sys.argv[1], lambda: brazo_groq(sys.argv[2]))()
