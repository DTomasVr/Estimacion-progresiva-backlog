"""E106 - Descomponer la tarea en pasos y contarlos: el principio del WBS.

Todo lo probado hasta ahora reprocesa el MISMO texto con otro codificador u
otros rasgos, y por eso se estrella siempre en el mismo sitio: no se puede
sacar informacion que no esta. Lo que sigue es distinto.

El modulo de generacion tiene un descomponedor -mT5 "decompose story into
tasks"- entrenado en OTRO corpus, el de jerarquias padre-hijo de Jira. Ese
modelo sabe algo que nuestro corpus de esfuerzo no contiene: en cuantos pasos
se suele partir un trabajo de un tipo dado. Es conocimiento del mundo
destilado, no una recodificacion del titulo.

La hipotesis es la de la descomposicion del trabajo, que es como estima una
persona: esfuerzo ~ numero de pasos x coste por paso. Si "migrar la tabla de
usuarios" se descompone en seis pasos y "corregir una errata" en uno, el
CONTEADOR lleva senal aunque el texto por si solo no la lleve.

Se mide lo unico que importa, que es si anade algo SOBRE el vector del texto:

  A  solo el vector e5                     la referencia
  B  solo los rasgos de descomposicion     ¿lleva senal por si mismo?
  C  vector + descomposicion               ¿anade sobre lo que ya hay?

Rasgos que salen del descomponedor: cuantos pasos, cuantas palabras en total,
la longitud del paso mas largo y cuantos pasos distintos entre si.

Sobre una muestra, porque generar es caro. Si el rho de B supera 0,15 o C
mejora a A, la via esta viva y se corre entera.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
CACHE = Path("data/processed/embeddings/codificadores")
GEN = Path(r"F:\Software Task generation Model")
M_ABAJO = GEN / "models/mt5-small-pruned_hist_pbe_v1_v5_pbe_v1"
OUT = Path("results/descomponer")
S = "=" * 100
N_MUESTRA, LOTE, PLIEGUES = 3000, 48, 3
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def met(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    return dict(err_log=float(np.median(np.abs(np.log(p) - np.log(y)))),
                rho=float(spearmanr(p, y).statistic),
                PRED25=float(np.mean(r <= .25)))


def separar(texto):
    """Los pasos vienen separados por ';' o por saltos, como en el generador."""
    br = [x.strip(" .;·-") for x in texto.replace("\n", ";").split(";")]
    return [x for x in br if len(x) > 3]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import torch
    from transformers import AutoTokenizer, MT5ForConditionalGeneration

    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy").astype(np.float32)
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    rng = np.random.default_rng(31)
    sel = rng.choice(len(sw), min(N_MUESTRA, len(sw)), replace=False)
    d = sw.iloc[sel].reset_index(drop=True)
    Vs = V[sel]
    y = d.effort_hours.to_numpy(float)
    ly = np.log(y)
    proy = d.project.to_numpy()
    tit = d.title.fillna("").astype(str).tolist()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(d):,} tareas · {d.project.nunique()} proyectos · {dev} · "
          f"{time.time() - t0:.0f} s", flush=True)

    tok = AutoTokenizer.from_pretrained(str(M_ABAJO))
    mo = MT5ForConditionalGeneration.from_pretrained(
        str(M_ABAJO), dtype=torch.float32).to(dev).eval()
    salidas = []
    with torch.inference_mode():
        for i in range(0, len(tit), LOTE):
            srcs = [f"decompose story into tasks: {t} | examples: "
                    for t in tit[i:i + LOTE]]
            enc = tok(srcs, return_tensors="pt", padding=True, truncation=True,
                      max_length=96).to(dev)
            out = mo.generate(**enc, max_new_tokens=128, num_beams=4,
                              no_repeat_ngram_size=4, repetition_penalty=1.3)
            salidas.extend(tok.batch_decode(out, skip_special_tokens=True))
            if i % (LOTE * 10) == 0:
                print(f"  {i:,}/{len(tit):,} · {time.time() - t0:.0f} s",
                      flush=True)

    pasos = [separar(s) for s in salidas]
    R = np.column_stack([
        np.array([len(p) for p in pasos], float),
        np.array([sum(len(x.split()) for x in p) for p in pasos], float),
        np.array([max([len(x.split()) for x in p], default=0) for p in pasos], float),
        np.array([len({x.lower() for x in p}) for p in pasos], float),
    ]).astype(np.float32)
    nom = ["n_pasos", "palabras_totales", "paso_mas_largo", "pasos_distintos"]
    print(f"\n{S}\nLO QUE SALE DEL DESCOMPONEDOR\n{S}")
    print(f"  pasos por tarea: mediana {np.median(R[:, 0]):.0f} · "
          f"p10 {np.quantile(R[:, 0], .1):.0f} · p90 {np.quantile(R[:, 0], .9):.0f}")
    for j, n_ in enumerate(nom):
        print(f"  rho({n_:17s}, horas reales) = "
              f"{spearmanr(R[:, j], y).statistic:+.4f}")
    print("\n  tres ejemplos:")
    for j in rng.choice(len(tit), 3, replace=False):
        print(f"    [{y[j]:6.2f} h] {tit[j][:60]}")
        print(f"             -> {len(pasos[j])} pasos: "
              f"{' | '.join(pasos[j][:3])[:90]}")

    filas = []
    for etq, X in (("A · solo el vector e5", Vs),
                   ("B · solo la descomposicion", R),
                   ("C · vector + descomposicion", np.hstack([Vs, R]))):
        p_ = np.zeros(len(y))
        for a, b in GroupKFold(PLIEGUES).split(X, y, proy):
            p_[b] = np.exp(HistGradientBoostingRegressor(**PAR)
                           .fit(X[a], ly[a]).predict(X[b]))
        filas.append(dict(brazo=etq, **met(y, p_)))
    t = pd.DataFrame(filas)
    print(f"\n{S}\n¿ANADE ALGO SOBRE EL TEXTO? ({len(y):,} tareas)\n{S}")
    print(t.round(4).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    a, b_, c = t.iloc[0], t.iloc[1], t.iloc[2]
    print(f"  solo texto          rho {a.rho:.4f} · error {a.err_log:.4f}")
    print(f"  solo descomposicion rho {b_.rho:.4f}")
    print(f"  las dos             rho {c.rho:.4f} ({c.rho - a.rho:+.4f})")
    if c.rho - a.rho > .02 or b_.rho > .15:
        print("\n  LA VIA ESTA VIVA. El descomponedor aporta informacion que el")
        print("  codificador no tiene, y merece correrse sobre el corpus entero.")
    else:
        print("\n  No aporta. El descomponedor solo reexpresa el titulo: si el")
        print("  texto no sabe cuanto cuesta, su descomposicion tampoco, porque")
        print("  sale del mismo texto.")
    pd.DataFrame(dict(titulo=tit, horas=y, n_pasos=R[:, 0],
                      salida=salidas)).to_csv(OUT / "descomposiciones.csv",
                                              index=False)
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n": int(len(y)),
               "rho_n_pasos": float(spearmanr(R[:, 0], y).statistic),
               "rho_texto": float(a.rho), "rho_juntos": float(c.rho)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/ · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
