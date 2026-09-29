"""Does the style gap cost anything? Rewrite REAL test tasks in the generator's style.

The corpus is written as Jira tickets ("[C++] ArrayFromJSON doesn't work for
float value dictionary type"); the module that will feed the estimator writes
short product imperatives ("Store payment data in the member profile"). The
obvious treatment is to rewrite the training corpus, which costs ~2 h of GPU.
Before paying it, this measures whether the gap costs anything at all -- and it
can be measured exactly, because rewriting tasks whose real hours we already
know keeps the ground truth.

  original    the real title, as the model was trained
  reescrito   the same task restyled by a local LLM, hours untouched

If the model loses little on the rewritten version, the style gap is small and
rewriting the training corpus cannot buy much. If it loses a lot, the rewrite of
the corpus is worth running.

Guards: the LLM never sees the hours, so it cannot leak the target; the cosine
between the original and the rewrite is reported so a rewrite that changed the
meaning shows up instead of hiding inside the error.

    python src/87_reescritura.py            # reescribe y evalua
    python src/87_reescritura.py --seguir   # reanuda una corrida a medias
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import wilcoxon

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
# 127.0.0.1 y no localhost: en Windows localhost resuelve primero a IPv6 y cada
# conexion nueva pierde ~2 s en el fallback. La sesion reutiliza la conexion.
URL = "http://127.0.0.1:11434/api/generate"
SESION = requests.Session()
MODELO_LLM = "qwen2.5:7b"
DATA = "data/processed/agix_effort_v4.parquet"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
ART = "models/candidato_titulos.joblib"
SALIDA = Path("results/reescritura")
N_POR_BLOQUE = 400

INSTRUCCION = """Rewrite this software issue title as a task in a product backlog.

Rules:
- imperative, starting with a verb: Store, Display, Send, Create, Add, Implement
- at most 12 words
- plain product language: no ticket keys, no version numbers, no class or file
  names, no project names, no stack traces
- keep WHAT the work is; do not invent scope and do not say how long it takes

Issue title: {titulo}

Answer with the rewritten task only, nothing else."""


def reescribir(titulo, intentos=3):
    for i in range(intentos):
        try:
            r = SESION.post(URL, json={
                "model": MODELO_LLM, "prompt": INSTRUCCION.format(titulo=titulo),
                "stream": False, "options": {"temperature": 0, "num_ctx": 2048,
                                             "num_predict": 40}}, timeout=120)
            t = r.json().get("response", "").strip().strip('"').split("\n")[0].strip()
            if 2 <= len(t.split()) <= 25:
                return t
        except Exception as e:                       # red o modelo caido
            if i == intentos - 1:
                print(f"    fallo tras {intentos} intentos: {e}", flush=True)
            time.sleep(1)
    return None


def muestra():
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    bloques = {"prueba SiP": sw[(sw.split_t3 == "test") & (sw.source == "SiP")],
               "externo no Apache": sw[(sw.split_t3 == "externo") & (~sw.apache)]}
    rng = np.random.default_rng(20260929)
    partes = []
    for nombre, d in bloques.items():
        k = min(N_POR_BLOQUE, len(d))
        m = d.iloc[rng.choice(len(d), k, replace=False)].copy()
        m["bloque"] = nombre
        partes.append(m)
    return pd.concat(partes)[["task_id", "bloque", "title", "task_kind", "effort_hours"]]


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    crudo = SALIDA / "reescrituras.jsonl"
    m = muestra()
    print(f"muestra: {len(m)} tareas ({m.bloque.value_counts().to_dict()})")

    hechas = {}
    if crudo.exists() and "--seguir" in sys.argv:
        for ln in crudo.open(encoding="utf-8"):
            try:
                d = json.loads(ln)
                hechas[d["task_id"]] = d["reescrito"]
            except json.JSONDecodeError:             # linea a medias
                pass
        print(f"  reanudando: {len(hechas)} ya hechas")
    else:
        crudo.write_text("", encoding="utf-8")

    pend = m[~m.task_id.isin(hechas)]
    t0 = time.time()
    with crudo.open("a", encoding="utf-8") as fh:
        for i, fila in enumerate(pend.to_dict("records"), 1):
            r = reescribir(str(fila["title"]))
            hechas[fila["task_id"]] = r
            fh.write(json.dumps({"task_id": fila["task_id"], "original": fila["title"],
                                 "reescrito": r}, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 50 == 0:
                s = (time.time() - t0) / i
                print(f"  {i}/{len(pend)} · {s:.2f} s/tarea · "
                      f"faltan {(len(pend) - i) * s / 60:.0f} min", flush=True)
    m["reescrito"] = m.task_id.map(hechas)
    validas = m.reescrito.notna()
    print(f"\nreescritas {int(validas.sum())}/{len(m)} "
          f"({time.time() - t0:.0f} s, {(time.time() - t0) / max(1, len(pend)):.2f} s/tarea)")
    m = m[validas].copy()

    # ------------------------------------------------ evaluar
    import torch
    from sentence_transformers import SentenceTransformer

    a = joblib.load(ART)
    reg, oh = a["regresor"], a["onehot"]
    st = SentenceTransformer(a["codificador"],
                             device="cuda" if torch.cuda.is_available() else "cpu")

    def predecir(textos, kinds):
        X = st.encode(list(textos), normalize_embeddings=False, show_progress_bar=False)
        if oh is not None:
            X = np.hstack([X, oh.transform(pd.DataFrame({"task_kind": list(kinds)}).astype(str))])
        return np.exp(reg.predict(X)), X

    y = m.effort_hours.to_numpy()
    p_ori, X_ori = predecir(m.title.astype(str), m.task_kind.astype(str))
    p_re, X_re = predecir(m.reescrito.astype(str), m.task_kind.astype(str))
    n_o = X_ori[:, :768] / np.linalg.norm(X_ori[:, :768], axis=1, keepdims=True)
    n_r = X_re[:, :768] / np.linalg.norm(X_re[:, :768], axis=1, keepdims=True)
    m["cos_sentido"] = (n_o * n_r).sum(1)
    m["pred_original"], m["pred_reescrito"] = p_ori, p_re

    def met(p, yy=None):
        yy = y if yy is None else yy
        p = np.clip(p, .01, None)
        q = E.point_metrics(yy, p)
        lr = np.log(p / yy)
        return dict(MdMRE=q["MdMRE"], MdAElog=float(np.median(np.abs(lr))),
                    PRED25=q["PRED25"], MAE_h=q["MAE_h"],
                    sesgo=float(np.exp(np.median(lr))),
                    spearman=float(pd.Series(yy).corr(pd.Series(p), method="spearman")))

    filas = [dict(version="original (estilo Jira)", n=len(m), **met(p_ori)),
             dict(version="reescrito (estilo del generador)", n=len(m), **met(p_re))]
    for b in m.bloque.unique():
        k = (m.bloque == b).to_numpy()
        filas += [dict(version=f"{b} · original", n=int(k.sum()), **met(p_ori[k], y[k])),
                  dict(version=f"{b} · reescrito", n=int(k.sum()), **met(p_re[k], y[k]))]
    t = pd.DataFrame(filas)
    e_o, e_r = np.abs(np.log(y / np.clip(p_ori, .01, None))), np.abs(np.log(y / np.clip(p_re, .01, None)))
    p_val = wilcoxon(e_o, e_r).pvalue if np.any(e_o != e_r) else 1.0
    rng = np.random.default_rng(1)
    d = [np.median(np.abs(y - p_re)[i] / y[i]) - np.median(np.abs(y - p_ori)[i] / y[i])
         for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    lo, hi = np.percentile(d, [2.5, 97.5])

    print("\n" + "=" * 118)
    print(t.round(4).to_string(index=False))
    print("=" * 118)
    print(f"reescribir cuesta {t.MdMRE.iloc[1] - t.MdMRE.iloc[0]:+.4f} de MdMRE "
          f"(IC95 [{lo:+.3f}, {hi:+.3f}], Wilcoxon p={p_val:.3g}) y "
          f"{t.MdAElog.iloc[1] - t.MdAElog.iloc[0]:+.4f} de error logarítmico")
    print(f"sentido conservado: coseno mediano {m.cos_sentido.median():.3f} "
          f"(p10 {m.cos_sentido.quantile(.1):.3f})")
    print("\nejemplos:")
    for _, r in m.sample(8, random_state=5).iterrows():
        print(f"  real {r.effort_hours:6.2f} h | {r.pred_original:5.2f} -> {r.pred_reescrito:5.2f} h"
              f" | cos {r.cos_sentido:.2f}")
        print(f"     ori: {str(r.title)[:88]}")
        print(f"     re : {str(r.reescrito)[:88]}")
    m.to_csv(SALIDA / "comparacion.csv", index=False)
    t.to_csv(SALIDA / "resumen.csv", index=False)
    print(f"\nescrito {SALIDA}/comparacion.csv y resumen.csv")


if __name__ == "__main__":
    main()
