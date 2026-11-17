"""E19 · Afinar el techo de ruido, que hoy descansa en 101 grupos.

El techo del trabajo -- "un modelo de solo texto no puede pasar de aqui" -- se
estimo con las 275 tareas cuyo titulo es byte a byte identico. Son 101 grupos:
pocos, y ademas sesgados, porque los titulos que se repiten literalmente son los
genericos ("documentation changes") y esos son mas ruidosos que la media. Si el
techo esta mal estimado, esta mal justo en la direccion que nos conviene, asi
que hay que apretarlo.

La forma de tener muchos mas grupos es relajar "identico" a "casi identico" en
el espacio del codificador: pares de tareas con coseno por encima de un umbral.
Cuanto mas alto el umbral mas se parece a la identidad literal, y la curva de
dispersion frente al umbral dice hacia donde tiende el ruido cuando el texto es
el mismo.

Se hace DENTRO del mismo proyecto y entre proyectos por separado, porque la
diferencia entre las dos curvas es la parte del ruido que es de organizacion y
no de tarea.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/techo")
SEP = "=" * 112
UMBRALES = [.90, .93, .95, .97, .98, .99]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    X = np.load(CACHE / "e5-large.npy")
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    ly = np.log(sw.effort_hours.to_numpy())
    proy = sw.project.to_numpy()
    total = float(ly.var())
    print(f"{len(sw):,} tareas · varianza total de log-horas {total:.3f}")

    # todos los pares por encima del umbral mas bajo, por bloques para la memoria
    pares_i, pares_j, pares_s = [], [], []
    paso = 2000
    for i in range(0, len(X), paso):
        s = X[i:i + paso] @ X.T
        for fila in range(s.shape[0]):
            g = i + fila
            j = np.flatnonzero(s[fila] >= UMBRALES[0])
            j = j[j > g]
            if len(j):
                pares_i.append(np.full(len(j), g))
                pares_j.append(j)
                pares_s.append(s[fila][j])
    pi = np.concatenate(pares_i)
    pj = np.concatenate(pares_j)
    ps = np.concatenate(pares_s)
    print(f"pares con coseno ≥ {UMBRALES[0]}: {len(pi):,}")

    mismo = proy[pi] == proy[pj]
    d = np.abs(ly[pi] - ly[pj])
    filas = []
    for u in UMBRALES:
        k = ps >= u
        for etq, m in (("todos", np.ones(len(k), bool)), ("mismo proyecto", mismo),
                       ("proyectos distintos", ~mismo)):
            kk = k & m
            if kk.sum() < 30:
                continue
            # la diferencia entre dos observaciones con el mismo texto tiene
            # varianza 2*sigma^2, asi que sigma = sd(diferencia)/sqrt(2)
            sd = float(np.std(ly[pi][kk] - ly[pj][kk]) / np.sqrt(2))
            filas.append(dict(umbral=u, conjunto=etq, pares=int(kk.sum()),
                              dif_mediana_h=float(np.exp(np.median(d[kk]))),
                              sigma_log=sd, var_irreducible=sd ** 2,
                              fraccion=sd ** 2 / total))
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nDISPERSIÓN ENTRE TAREAS CASI IDÉNTICAS\n{SEP}")
    for etq in t.conjunto.unique():
        print(f"\n{etq}")
        print(t[t.conjunto == etq].drop(columns=["conjunto"]).round(4).to_string(index=False))

    # techo implicado por cada sigma
    rng = np.random.default_rng(0)
    techos = []
    for _, r in t[t.conjunto == "todos"].iterrows():
        e = rng.normal(0, r.sigma_log, 300_000)
        rr = np.abs(np.exp(e) - 1)
        techos.append(dict(umbral=r.umbral, sigma_log=r.sigma_log,
                           MdMRE=float(np.median(rr)), PRED25=float(np.mean(rr <= .25)),
                           R2_max=float(1 - r.fraccion)))
    tt = pd.DataFrame(techos)
    print(f"\n{SEP}\nTECHO IMPLICADO POR CADA NIVEL DE PARECIDO\n{SEP}")
    print(tt.round(4).to_string(index=False))
    print("\n  el modelo final está en MdMRE 0,679 · PRED25 0,168 · R² 0,181")
    print("  (prueba fija). La columna R²_max dice cuánto podría explicar como")
    print("  máximo un modelo que solo viera el texto.")

    t.to_csv(OUT / "afinado.csv", index=False)
    tt.to_csv(OUT / "techo_por_umbral.csv", index=False)
    json.dump({"varianza_total": total, "umbrales": UMBRALES,
               "n_pares": int(len(pi))},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/afinado.csv y techo_por_umbral.csv")


if __name__ == "__main__":
    main()
