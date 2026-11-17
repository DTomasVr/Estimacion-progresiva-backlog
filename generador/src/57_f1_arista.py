"""F1 de ARISTA: la metrica que faltaba.

El F1 suave de conjunto que usamos se calcula por nivel y por separado, asi que
tiene un punto ciego: si el modelo produce las tareas correctas del proyecto
pero las cuelga de la historia equivocada, el F1 de tareas NO baja. Y eso es
justo el defecto que se ve al leer la salida.

La unidad de evaluacion pasa a ser la ARISTA (historia, tarea) en lugar de la
tarea sola. Es la Edge F1 de la induccion de taxonomias y la relation
generation de Wiseman, Shieber y Rush (EMNLP 2017); ver entrega/METRICAS.md.

Procedimiento:
  1. emparejamiento OPTIMO de historias generadas con las de referencia
     (asignacion hungara sobre la similitud del SBERT propio, umbral 0,5)
  2. para cada par de historias emparejadas, se emparejan sus tareas
  3. P = aristas acertadas / aristas generadas,  R = aristas acertadas /
     aristas de referencia
  4. aparte, MALA ATRIBUCION sin referencia: tareas que se parecen mas a otra
     historia del mismo backlog que a la suya

Comprobacion de que la metrica mide lo que dice (--sanidad): se toma un backlog
de referencia y se reparten sus tareas entre historias equivocadas. El F1 de
NODO casi no se mueve, el de ARISTA se hunde. Si eso no pasa, la metrica no
sirve.

  python src/57_f1_arista.py --sanidad          valida la metrica, solo CPU
  python src/57_f1_arista.py --modelo           evalua el pipeline actual

out: results/57_arista.csv
"""
import json
import os
import random
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UMBRAL = 0.5          # el mismo del F1 suave de conjunto que ya usamos
SEED = 20260926


def cargar_sbert():
    return SentenceTransformer(f"{BASE}/models/sbert_agix_v1",
                               device=os.environ.get("AGIX_DEVICE", "cpu"))


def emparejar(sb, A, B, umbral=UMBRAL):
    """asignacion optima entre dos listas de textos; devuelve [(i, j, sim)]"""
    if not A or not B:
        return []
    Ea = sb.encode(A, normalize_embeddings=True, show_progress_bar=False)
    Eb = sb.encode(B, normalize_embeddings=True, show_progress_bar=False)
    S = Ea @ Eb.T
    fi, co = linear_sum_assignment(-S)
    return [(int(i), int(j), float(S[i, j])) for i, j in zip(fi, co) if S[i, j] >= umbral]


def f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def evaluar(sb, gen, ref):
    """gen y ref: [(historia, [tareas])]. Devuelve nodo y arista."""
    hg = [h for h, _ in gen]
    hr = [h for h, _ in ref]
    pares = emparejar(sb, hg, hr)

    # --- nivel NODO: historias
    p_h = len(pares) / max(1, len(hg))
    r_h = len(pares) / max(1, len(hr))

    # --- nivel NODO: tareas, todas contra todas SIN mirar de quien cuelgan
    tg = [t for _, ts in gen for t in ts]
    tr = [t for _, ts in ref for t in ts]
    pt = emparejar(sb, tg, tr)
    p_t = len(pt) / max(1, len(tg))
    r_t = len(pt) / max(1, len(tr))

    # --- nivel ARISTA: la tarea cuenta solo si esta bajo la historia que toca
    ok = 0
    for i, j, _ in pares:
        ok += len(emparejar(sb, gen[i][1], ref[j][1]))
    p_a = ok / max(1, len(tg))
    r_a = ok / max(1, len(tr))

    return dict(hist_P=p_h, hist_R=r_h, hist_F1=f1(p_h, r_h),
                nodo_P=p_t, nodo_R=r_t, nodo_F1=f1(p_t, r_t),
                arista_P=p_a, arista_R=r_a, arista_F1=f1(p_a, r_a),
                n_gen=len(tg), n_ref=len(tr))


def mala_atribucion(sb, bl):
    """sin referencia: tareas mas cercanas a una hermana que a su propia historia"""
    hs = [h for h, ts in bl if ts]
    if len(hs) < 3:
        return None
    idx = [i for i, (_, ts) in enumerate([(h, t) for h, t in bl if t]) for _ in ts]
    tareas = [t for _, ts in bl for t in ts]
    if not tareas:
        return None
    Eh = sb.encode(hs, normalize_embeddings=True, show_progress_bar=False)
    Et = sb.encode(tareas, normalize_embeddings=True, show_progress_bar=False)
    S = Et @ Eh.T
    mal = sum(1 for k, i in enumerate(idx) if S[k, i] < np.delete(S[k], i).max())
    return mal / len(idx)


# AGIX_REF elige contra que referencia se mide. Por omision v1, que es la
# poblacion sobre la que ya esta medido el modelo anterior: la comparacion tiene
# que ser pareada. Aviso para leer los numeros: las referencias v1 las escribio
# el profesor v1, que usaba UN solo rol por backlog, asi que un modelo que
# reparte bien los actores puede parecer peor en el nivel de historia sin serlo.
# Lo que no sufre ese sesgo es el F1 de ARISTA, que mide estructura, y el test de
# las 328 historias humanas.
REF = os.environ.get("AGIX_REF", "v1")
SUF = os.environ.get("AGIX_SUFIJO", "")


def referencia_reservada():
    """backlogs del profesor que nunca se entrenaron"""
    pref, fich = (("mixa", "backlogs_qwen2.5_7b.jsonl") if REF == "v1"
                  else ("v3a", "backlogs_v3_qwen2.5_7b.jsonl"))
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_{pref}_test.parquet")
    ids = {str(x).split("#")[0] for x in d.parent_uid.dropna()
           if not str(x).startswith("Apache")}
    B = [json.loads(l) for l in open(f"{BASE}/data/synth/{fich}", encoding="utf-8")]
    return [b for b in B if b["id"] in ids]


# --------------------------------------------------------------------- sanidad
def sanidad():
    """valida la metrica: al desordenar las tareas entre historias, el F1 de
    NODO casi no se mueve y el de ARISTA se hunde"""
    sb = cargar_sbert()
    B = referencia_reservada()
    if not B:
        print("no se encontraron backlogs reservados; se usan los 30 primeros")
        B = [json.loads(l) for l in open(f"{BASE}/data/synth/backlogs_qwen2.5_7b.jsonl",
                                         encoding="utf-8")][:30]
    B = B[:25]
    print(f"{len(B)} backlogs de referencia\n")
    rng = random.Random(SEED)
    filas = []
    for etiqueta, transformar in [
        ("identico (techo)", lambda bl: bl),
        ("tareas desordenadas entre historias", None),
        ("mitad de las tareas desordenadas", None),
    ]:
        acc = []
        for b in B:
            ref = [(h["historia"], list(h["tareas"])) for h in b["historias"]]
            if etiqueta.startswith("identico"):
                gen = [(h, list(ts)) for h, ts in ref]
            else:
                todas = [t for _, ts in ref for t in ts]
                n = len(todas)
                orden = list(range(n))
                k = n if etiqueta.startswith("tareas") else n // 2
                sub = orden[:k]
                rng.shuffle(sub)
                orden[:k] = sub
                mezcladas = [todas[i] for i in orden]
                gen, p = [], 0
                for h, ts in ref:
                    gen.append((h, mezcladas[p:p + len(ts)]))
                    p += len(ts)
            r = evaluar(sb, gen, ref)
            r["mala_atrib"] = mala_atribucion(sb, gen)
            acc.append(r)
        m = {k: float(np.mean([a[k] for a in acc if a[k] is not None])) for k in acc[0]}
        m["caso"] = etiqueta
        filas.append(m)
        print(f"{etiqueta:38s}  historias F1 {m['hist_F1']:.3f} | "
              f"NODO F1 {m['nodo_F1']:.3f} | ARISTA F1 {m['arista_F1']:.3f} | "
              f"mala atribucion {100*m['mala_atrib']:.1f} %")
    D = pd.DataFrame(filas)
    os.makedirs(f"{BASE}/results", exist_ok=True)
    D.to_csv(f"{BASE}/results/57_sanidad.csv", index=False)
    print(f"\n-> results/57_sanidad.csv")
    d = filas[0]["nodo_F1"] - filas[1]["nodo_F1"]
    a = filas[0]["arista_F1"] - filas[1]["arista_F1"]
    print(f"\nal desordenar: el F1 de NODO cae {d:.3f} y el de ARISTA cae {a:.3f} "
          f"({a/max(d,1e-6):.0f} veces mas)")
    print("la metrica distingue 'las tareas correctas' de 'las tareas en su sitio'"
          if a > 3 * d else "ATENCION: la metrica no discrimina, revisar")


# ---------------------------------------------------------------------- modelo
def modelo():
    import importlib.util
    os.environ.setdefault("AGIX_DEVICE", "cuda")
    spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
    pl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pl)
    for v, at in [("AGIX_ARRIBA", "M_ARRIBA"), ("AGIX_ABAJO", "M_ABAJO"),
                  ("AGIX_ABAJO2", "M_ABAJO2")]:
        if os.environ.get(v):
            setattr(pl, at, os.environ[v])
    print(f"arriba {os.path.basename(pl.M_ARRIBA)} | abajo {os.path.basename(pl.M_ABAJO)}")
    P = pl.Pipeline(traducir=False)
    sb = P.sb
    B = referencia_reservada()
    print(f"{len(B)} backlogs reservados (referencia del profesor, nunca entrenados)")
    acc = []
    for i, b in enumerate(B):
        ref = [(h["historia"], list(h["tareas"])) for h in b["historias"]]
        r = P(b["descripcion"], k_historias=len(ref), n_muestras=max(30, 3 * len(ref)))
        gen = [(h["historia"], list(h["tareas"])) for h in r["historias"]]
        if not gen:
            continue
        m = evaluar(sb, gen, ref)
        m["mala_atrib"] = mala_atribucion(sb, gen)
        m["sector"] = b["sector"]
        acc.append(m)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(B)}", flush=True)
    D = pd.DataFrame(acc)
    D.to_csv(f"{BASE}/results/57_arista{SUF}.csv", index=False)
    print("\n" + "=" * 70)
    for c in ["hist_F1", "nodo_F1", "arista_F1", "mala_atrib"]:
        print(f"  {c:12s} {D[c].mean():.4f}")
    print(f"-> results/57_arista{SUF}.csv  (n={len(D)}, referencia {REF})")


if __name__ == "__main__":
    if "--modelo" in sys.argv:
        modelo()
    else:
        sanidad()
