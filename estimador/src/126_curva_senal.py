"""E26 · La curva de señal: ¿cuánto sabe el texto, de verdad, sobre las horas?

E19 midio la dispersion entre tareas CASI IDENTICAS y salio muy baja dentro del
mismo proyecto (x1,00, sigma 0,36). De ahi se podria concluir que el texto
determina casi por completo las horas dentro de un proyecto y que al modelo le
queda muchisimo margen. Pero las tareas que tienen un gemelo son las
formularias, que probablemente sean tambien las mas predecibles: el numero puede
estar diciendo mas sobre ESAS tareas que sobre el texto en general.

La forma de comprobarlo es no quedarse en el extremo. Se toman pares de tareas
DEL MISMO PROYECTO en todo el rango de parecido, y se mide como crece la
diferencia de horas segun baja el coseno. Si la curva sube deprisa y se aplana
enseguida en el nivel del ruido total, el texto solo informa de las tareas
practicamente iguales y el techo real es mucho mas bajo de lo que sugeria E19.

De la curva sale una estimacion honesta del techo: la dispersion que le queda a
un modelo que solo puede distinguir tareas por su parecido semantico.
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
SEP = "=" * 108
BORDES = [.2, .4, .5, .6, .7, .75, .8, .85, .9, .93, .95, .97, 1.01]
MAX_PARES = 3_000_000


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

    rng = np.random.default_rng(0)
    cos, dif = [], []
    for p in np.unique(proy):
        i = np.flatnonzero(proy == p)
        if len(i) < 30:
            continue
        if len(i) > 700:                      # muestreo para no explotar
            i = rng.choice(i, 700, replace=False)
        s = X[i] @ X[i].T
        a, b = np.triu_indices(len(i), 1)
        cos.append(s[a, b])
        dif.append(np.abs(ly[i][a] - ly[i][b]))
    cos = np.concatenate(cos)
    dif = np.concatenate(dif)
    if len(cos) > MAX_PARES:
        k = rng.choice(len(cos), MAX_PARES, replace=False)
        cos, dif = cos[k], dif[k]
    print(f"pares dentro del mismo proyecto: {len(cos):,}")

    filas = []
    for lo, hi in zip(BORDES[:-1], BORDES[1:]):
        k = (cos >= lo) & (cos < hi)
        if k.sum() < 200:
            continue
        # sigma de una observacion = sd de la diferencia / raiz(2)
        sd = float(np.std(dif[k]) / np.sqrt(2))
        filas.append(dict(coseno=f"{lo:.2f}–{hi:.2f}", pares=int(k.sum()),
                          dif_mediana_h=float(np.exp(np.median(dif[k]))),
                          sigma_log=sd, fraccion_ruido=sd ** 2 / total))
    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nCÓMO CRECE LA DIFERENCIA DE HORAS AL BAJAR EL PARECIDO\n{SEP}")
    print(t.round(4).to_string(index=False))

    # techo realista: media de sigma ponderada por como se reparten los pares
    peso = t.pares / t.pares.sum()
    sigma_tip = float((t.sigma_log * peso).sum())
    rng2 = np.random.default_rng(1)
    e = rng2.normal(0, sigma_tip, 300_000)
    r = np.abs(np.exp(e) - 1)
    print(f"\n{SEP}\nLECTURA\n{SEP}")
    print(f"  σ en el tramo más parecido (≥ 0,97): "
          f"{t.sigma_log.iloc[-1]:.3f}")
    print(f"  σ típico, ponderando como se reparten los pares de verdad: {sigma_tip:.3f}")
    print(f"  fracción de la varianza que ese σ representa: "
          f"{sigma_tip ** 2 / total:.1%}")
    print(f"  techo implicado si el modelo solo pudiera distinguir por parecido:")
    print(f"    MdMRE {np.median(r):.3f} · PRED25 {np.mean(r <= .25):.3f} · "
          f"R² máx {1 - sigma_tip ** 2 / total:.3f}")
    print(f"\n  el modelo final está en MdMRE 0,679 · PRED25 0,168 · R² 0,181")
    print("  (el techo de arriba es optimista: supone que el modelo aprovecha TODO")
    print("   el parecido semántico, y ademas los pares se reparten muy desigual)")

    t.to_csv(OUT / "curva_senal.csv", index=False)
    json.dump({"sigma_ponderado": sigma_tip, "varianza_total": total,
               "n_pares": int(len(cos))},
              open(OUT / "curva_senal.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/curva_senal.csv")


if __name__ == "__main__":
    main()
