"""Evalua las TAREAS que produce el pipeline completo contra las tareas humanas
del corpus de Jira reservado.

Por que hace falta ademas de src/11_eval_gen.py. El 11 evalua el MODELO crudo con
una configuracion de generacion fija, que es lo que reporta la tesis (0,468). Pero
lo que se despliega no es el modelo crudo: es el conjunto de dos generadores, la
segunda pasada con arranques vetados, el filtro de tareas degeneradas, el
deduplicador y el tope de arranques por backlog. Ninguna de esas piezas aparece en
el 11, asi que su efecto sobre la fidelidad a referencias HUMANAS estaba sin medir.

Aqui se reproduce exactamente la etapa de tareas del pipeline sobre las historias
de prueba y se puntua con la misma metrica (F1 suave de conjunto con el SBERT
propio), mas la concentracion de arranques y el ratio de conteo, para ver las tres
cosas juntas.

  python src/75_tareas_vs_humanas.py [N]

out: results/75_tareas_humanas.csv
"""
import collections
import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = int(next((a for a in sys.argv[1:] if a.isdigit()), 400))
SEED = 20260922
os.environ.setdefault("AGIX_DEVICE", "cuda")

spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
spec2 = importlib.util.spec_from_file_location("ev", f"{BASE}/src/11_eval_gen.py")
ev = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(ev)

d = pd.read_parquet(f"{BASE}/data/processed/s2s_pruned_test.parquet").reset_index(drop=True)
d = d[d.mitad == "tasks"].reset_index(drop=True)
rng = np.random.default_rng(SEED)
if N < len(d):
    d = d.iloc[rng.permutation(len(d))[:N]].reset_index(drop=True)
print(f"{len(d)} historias de prueba con tareas escritas por personas", flush=True)

P = pl.Pipeline(traducir=False)
# la entrada de la mitad de abajo, tal y como la arma el pipeline
historias = [str(s).split("decompose story into tasks: ")[-1].split(" | examples:")[0]
             for s in d.src]
refs = [list(x) for x in d.tareas]


def etapa_tareas(hs, con_extras=True):
    """reproduce la etapa de tareas del pipeline sobre una lista de historias"""
    gen = dict(max_new_tokens=192, num_beams=pl.BEAMS_ABAJO,
               no_repeat_ngram_size=4, repetition_penalty=1.3)
    srcs = [f"decompose story into tasks: {h} | examples: " for h in hs]
    listas = [P._generar(P.tok_b, P.m_b, srcs, **gen)]
    if con_extras and P.m_b2 is not None:
        listas.append(P._generar(P.tok_b, P.m_b2, srcs, **gen))
    if con_extras and pl.VETAR_ARRANQUES and listas:
        cuenta = collections.Counter(
            a for lst in listas for s_ in lst for x in pl.separar_tareas(s_)
            if (a := pl.arranque(x)))
        top = [a for a, n in cuenta.most_common(3) if n >= 3]
        mal = []
        for a in top:
            for var in (a, a[0].upper() + a[1:]):
                ids = P.tok_b(var, add_special_tokens=False).input_ids
                if ids:
                    mal.append(ids)
        if mal:
            listas.append(P._generar(P.tok_b, P.m_b, srcs, bad_words_ids=mal, **gen))
    salidas = ([" ; ".join(x) for x in zip(*listas)] if len(listas) > 1 else listas[0])

    fuera = []
    for h, s_ in zip(hs, salidas):
        ts = [t for t in pl.separar_tareas(s_)
              if not (con_extras and pl.tarea_degenerada(t, h))]
        if con_extras and len(ts) > 1:
            E = P.sb.encode(ts, convert_to_tensor=True, normalize_embeddings=True,
                            show_progress_bar=False)
            S = (E @ E.T).float().cpu().numpy()
            keep, arr = [], collections.Counter()
            for i, t in enumerate(ts):
                if any(S[i, j] >= pl.UMBRAL_TAREA_DUP_INTRA for j in range(i) if ts[j] in keep):
                    continue
                a = pl.arranque(t)
                if pl.TOPE_ARRANQUE_SALIDA and a and arr[a] >= pl.TOPE_ARRANQUE_SALIDA:
                    continue
                if a:
                    arr[a] += 1
                keep.append(t)
            ts = keep
        fuera.append(ts)
    return fuera


def concentracion(listas, n=5):
    a = collections.Counter()
    for ts in listas:
        for t in ts:
            x = pl.arranque(t)
            if x:
                a[x] += 1
    tot = sum(a.values())
    return sum(c for _, c in a.most_common(n)) / max(1, tot)


filas, puntos = [], {}
for etq, extras in [("modelo crudo", False), ("pipeline completo", True)]:
    gens = []
    for i in range(0, len(historias), 24):
        gens += etapa_tareas(historias[i:i + 24], con_extras=extras)
        if (i // 24) % 4 == 0:
            print(f"  {etq}: {min(i+24, len(historias))}/{len(historias)}", flush=True)
    s = ev.puntuar(P.sb, gens, refs)
    puntos[etq] = s["por_ejemplo"]
    filas.append(dict(variante=etq,
                      **{c: round(float(s["mat"][:, i].mean()), 4)
                         for i, c in enumerate(s["cols"])},
                      concentracion=round(concentracion(gens), 4),
                      tareas=round(float(np.mean([len(g) for g in gens])), 2)))
    print(f"{etq}: sF1 {filas[-1]['sF1']:.4f}  tareas/historia {filas[-1]['tareas']}  "
          f"concentracion {100*filas[-1]['concentracion']:.1f} %", flush=True)

R = pd.DataFrame(filas)
R.to_csv(f"{BASE}/results/75_tareas_humanas.csv", index=False)
print("\n" + R.to_string(index=False))
if len(puntos) == 2:
    a, b = puntos["pipeline completo"], puntos["modelo crudo"]
    dd = a - b
    bs = dd[rng.integers(0, len(dd), (2000, len(dd)))].mean(1)
    print(f"\npipeline - modelo crudo: {dd.mean():+.4f} "
          f"IC95 [{np.quantile(bs,.025):+.4f}, {np.quantile(bs,.975):+.4f}] (n={len(dd)})")
print("-> results/75_tareas_humanas.csv")
