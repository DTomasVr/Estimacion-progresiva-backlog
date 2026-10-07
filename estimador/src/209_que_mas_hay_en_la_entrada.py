"""E94 - Pasar el muro por donde no hemos mirado: la ENTRADA.

E90-E92 dejaron el muro con un numero: la mezcla de expertos necesita una
compuerta de regimen con AUC >= 0,80 y el titulo solo da 0,646, con tres
senales distintas convergiendo ahi. La conclusion que saque es que es un techo
del TEXTO. Pero "el texto" en este proyecto quiere decir SOLO EL TITULO, con
una mediana de 51 caracteres, y eso fue una decision de diseno -el generador de
tareas solo produce titulos- no una limitacion del dato.

Hay tres entradas disponibles que el modelo nunca ha visto, y las tres son
legitimas: existen cuando la tarea se crea, que es la regla de HU-49-4.

  task_kind     bug / task / improvement / feature / docs / support
                100 % de cobertura. Un bug y una migracion no cuestan igual, y
                el usuario de Agix elige el tipo al crear la tarea
  priority      100 % de cobertura
  description   56,9 % de cobertura, mediana 87 caracteres. Pedir una frase de
                contexto es poco rozamiento comparado con estimar horas

Se mide lo que suben las dos cosas que importan:

  la COMPUERTA   AUC de distinguir <=2 h de >2 h. Si pasa de 0,646 a 0,80, la
                 mezcla de expertos de E90 revive y vale +5,8 puntos
  el REGRESOR    el +-25 % por tarea directamente

  A  solo titulo                        lo actual
  B  titulo + tipo y prioridad          gratis, 100 % de cobertura
  C  titulo + descripcion (TF-IDF+SVD)  requiere pedir una frase
  D  todo junto

La descripcion entra por TF-IDF con SVD ajustado DENTRO del pliegue. Si asoma
senal, vale la pena pagar un codificador neuronal para ella; si no, no.
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
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              HistGradientBoostingRegressor)
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/que_mas_hay_en_la_entrada")
S = "=" * 100
PLIEGUES, CORTE, DIMS = 5, 2.0, 150
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def mide(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                MAE_h=float(np.abs(p - y).mean()),
                spearman=float(spearmanr(p, y).statistic))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap].astype(np.float32)
    y = sw.effort_hours.to_numpy(float)
    ly = np.log(y)
    proy = sw.project.to_numpy()
    corto = y <= CORTE
    desc = sw.description.fillna("").astype(str).to_numpy()
    tit = sw.title.fillna("").astype(str)

    # estructurados: tipo, prioridad y dos medidas de forma del titulo
    kind = pd.get_dummies(sw.task_kind.fillna("?").astype(str), prefix="k")
    prio = pd.get_dummies(sw.priority.fillna("?").astype(str), prefix="p")
    extra = np.hstack([kind.to_numpy(float), prio.to_numpy(float),
                       tit.str.len().to_numpy(float)[:, None],
                       tit.str.split().str.len().to_numpy(float)[:, None],
                       (sw.has_desc.to_numpy(float))[:, None]]).astype(np.float32)
    print(f"{len(y):,} tareas · {sw.project.nunique()} proyectos · "
          f"{kind.shape[1]} tipos · {prio.shape[1]} prioridades · "
          f"descripcion en {np.mean([len(d) > 0 for d in desc]):.1%} · "
          f"{time.time() - t0:.0f} s", flush=True)
    print("  horas medianas por tipo: " + " · ".join(
        f"{k} {v:.1f}" for k, v in
        sw.groupby("task_kind").effort_hours.median().sort_values().items()),
        flush=True)

    brazos = ("A · solo titulo", "B · + tipo y prioridad",
              "C · + descripcion", "D · todo junto")
    pr = {b: np.zeros(len(y)) for b in brazos}
    ga = {b: np.zeros(len(y)) for b in brazos}
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(V, y, proy)):
        # la descripcion, reducida DENTRO del pliegue
        tf = TfidfVectorizer(min_df=5, max_features=60000, ngram_range=(1, 2),
                             sublinear_tf=True)
        Da = tf.fit_transform(desc[a])
        svd = TruncatedSVD(DIMS, random_state=0).fit(Da)
        Ea = svd.transform(Da).astype(np.float32)
        Eb = svd.transform(tf.transform(desc[b])).astype(np.float32)
        X = {
            "A · solo titulo": (V[a], V[b]),
            "B · + tipo y prioridad": (np.hstack([V[a], extra[a]]),
                                       np.hstack([V[b], extra[b]])),
            "C · + descripcion": (np.hstack([V[a], Ea]), np.hstack([V[b], Eb])),
            "D · todo junto": (np.hstack([V[a], extra[a], Ea]),
                               np.hstack([V[b], extra[b], Eb])),
        }
        for k, (Xa, Xb) in X.items():
            pr[k][b] = np.exp(HistGradientBoostingRegressor(**PAR)
                              .fit(Xa, ly[a]).predict(Xb))
            ga[k][b] = (HistGradientBoostingClassifier(**PAR)
                        .fit(Xa, corto[a]).predict_proba(Xb)[:, 1])
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s",
              flush=True)

    filas = []
    for k in brazos:
        auc = float(roc_auc_score(corto, ga[k]))
        filas.append(dict(brazo=k, AUC_compuerta=auc, **mide(y, pr[k])))
    t = pd.DataFrame(filas)
    print(f"\n{S}\n¿SUBE LA COMPUERTA? ¿SUBE EL REGRESOR?\n{S}")
    print(t[["brazo", "AUC_compuerta", "PRED25", "MdMRE", "factor2", "MAE_h",
             "spearman"]].round(4).to_string(index=False))

    # solo donde SI hay descripcion, que es donde C y D pueden lucir
    con = np.array([len(d) > 0 for d in desc])
    print(f"\n{S}\nSOLO LAS {con.sum():,} TAREAS QUE TRAEN DESCRIPCION\n{S}")
    f2 = []
    for k in brazos:
        f2.append(dict(brazo=k, AUC_compuerta=float(roc_auc_score(corto[con],
                                                                 ga[k][con])),
                       **mide(y[con], pr[k][con])))
    t2 = pd.DataFrame(f2)
    print(t2[["brazo", "AUC_compuerta", "PRED25", "MdMRE", "spearman"]]
          .round(4).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    a0 = t[t.brazo.str.startswith("A")].iloc[0]
    mejor = t.loc[t.AUC_compuerta.idxmax()]
    print(f"  compuerta: {a0.AUC_compuerta:.4f} (solo titulo) -> "
          f"{mejor.AUC_compuerta:.4f} ({mejor.brazo})")
    print(f"  umbral para que la mezcla de expertos compense: 0,80")
    if mejor.AUC_compuerta >= .80:
        print("  LLEGA: la mezcla de expertos de E90 revive, hay que remedirla.")
    elif mejor.AUC_compuerta - a0.AUC_compuerta > .03:
        print(f"  No llega, pero sube {mejor.AUC_compuerta - a0.AUC_compuerta:+.3f}. "
              "Vale la pena pagar un codificador neuronal para la descripcion y")
        print("  volver a medir.")
    else:
        print("  No se mueve. El techo no es del titulo: es de la tarea aislada.")
    mp = t.loc[t.PRED25.idxmax()]
    print(f"\n  regresor: {a0.PRED25:.1%} (solo titulo) -> {mp.PRED25:.1%} "
          f"({mp.brazo}) · {(mp.PRED25 - a0.PRED25) * 100:+.1f} puntos")
    t.to_csv(OUT / "resumen.csv", index=False)
    t2.to_csv(OUT / "con_descripcion.csv", index=False)
    json.dump({"n": int(len(y)), "auc_titulo": float(a0.AUC_compuerta),
               "auc_mejor": float(mejor.AUC_compuerta), "mejor": mejor.brazo,
               "pred25_titulo": float(a0.PRED25), "pred25_mejor": float(mp.PRED25)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
