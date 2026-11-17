"""E73 - Cuanto se gana al sumar, con las MISMAS predicciones.

El informe afirmaba que "el MdMRE pasa de 0,741 por tarea a 0,355 por proyecto", y
ese 0,741 no salia de ningun CSV. Es ademas una comparacion mal planteada si los dos
numeros no vienen del mismo modelo: el 0,355 lo produce la cabeza de la MEDIA (la de
Poisson, la que se suma) y el 0,741 parecia venir de la cabeza de la MEDIANA medida
en otro bloque.

La comparacion que sostiene el argumento de la cancelacion de errores es la de las
MISMAS predicciones medidas en las dos unidades: tarea y proyecto. Eso es lo que se
calcula aqui, sobre los mismos 114 proyectos y las mismas predicciones fuera de
pliegue que producen la metrica publicada.

Que la cabeza de la media salga mal por tarea es lo esperado y no es un defecto: con
un objetivo asimetrico la media esta por encima de la mediana, asi que sobreestima la
tarea tipica. Es el precio de poder sumarla, y es justo lo que justifica que el
artefacto lleve DOS cabezas y no una.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
OUT = Path("results/cancelacion")
S = "=" * 100
MIN_TAREAS = 8


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    assert len(pred) == len(sw), f"pred_total {len(pred)} no cuadra con {len(sw)}"
    y = sw.effort_hours.to_numpy()
    tam = sw.groupby("project").size()
    k = sw.project.isin(tam[tam >= MIN_TAREAS].index).to_numpy()
    sw, pred, y = sw[k].reset_index(drop=True), pred[k], y[k]

    r_tarea = np.abs(pred - y) / y
    g = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": pred}).groupby("p")
    real_t, est_t = g.real.sum().to_numpy(), g.est.sum().to_numpy()
    r_proy = np.abs(est_t - real_t) / real_t

    filas = [dict(unidad="por tarea", n=int(len(y)), MdMRE=float(np.median(r_tarea)),
                  PRED25=float(np.mean(r_tarea <= .25)),
                  factor2=float(np.mean(np.maximum(pred / y, y / pred) <= 2))),
             dict(unidad="por proyecto", n=int(len(real_t)),
                  MdMRE=float(np.median(r_proy)),
                  PRED25=float(np.mean(r_proy <= .25)),
                  factor2=float(np.mean(np.maximum(est_t / real_t,
                                                   real_t / est_t) <= 2)))]
    t = pd.DataFrame(filas)
    print(f"{S}\nLAS MISMAS PREDICCIONES, EN LAS DOS UNIDADES\n{S}")
    print(t.round(4).to_string(index=False))
    f = filas[0]["MdMRE"] / filas[1]["MdMRE"]
    print(f"\n  sumar divide el MdMRE por {f:.2f}")
    print("  (la cabeza de la media sobreestima la tarea suelta a proposito: es el "
          "precio de poder sumarla)")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": int(len(real_t)), "tareas": int(len(y)),
               "factor_de_mejora": float(f)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
