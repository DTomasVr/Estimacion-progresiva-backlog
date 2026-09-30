"""Rehacer la cadena entera del total del proyecto, en orden y de una vez.

Cuando cambia el estimador por tarea cambia todo lo que cuelga de el: la curva de
tareas cerradas, los intervalos, los ejemplos, las figuras y el deck. Hacerlo a
mano invita a que una tabla del deck quede con cifras de hace dos horas y otra
con las nuevas, que es la forma mas facil de llevar a una exposicion dos numeros
que no cuadran.

    python src/166_rehacer_cadena.py            rehace todo
    python src/166_rehacer_cadena.py --sin-modelo   salta el modelo y rehace lo demas

El estimador por tarea que usa todo lo de abajo vive en
results/modelo_total/pred_total.npy. Este script elige el mejor brazo de E59 en
las metricas que se reportan y lo escribe ahi, de modo que los pasos siguientes
no tienen que enterarse de nada.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(".")
PY = sys.executable
PASOS = [
    ("165_total_ampliado.py", "modelo del total, con todos los canales"),
    (None, "elegir el mejor brazo y publicarlo como pred_total.npy"),
    ("159_curva_final.py", "curva de tareas cerradas"),
    ("161_banda_proyecto.py", "intervalos"),
    ("162_ejemplos_proyecto.py", "ejemplos concretos"),
    ("164_estimacion_usuario_proyecto.py", "estimacion del usuario"),
    ("141_figuras_metricas.py", "figuras"),
    ("152_pptx_exposicion.py", "deck"),
]
COLS = ["MdMRE", "R2_log", "PRED25", "factor2"]


def elegir():
    """El brazo que gana en mas metricas de las que se reportan.

    No se elige por una sola: en este problema casi cada metrica prefiere un
    brazo distinto, y quedarse con la que mejor sale seria elegir el resultado."""
    t = pd.read_csv("results/total_ampliado/resumen.csv")
    P = np.load("results/total_ampliado/pred.npy")
    orden = ["base", "mas_lineal", "mas_desc", "todo"]
    puntos = np.zeros(len(t))
    for c in COLS:
        v = t[c].to_numpy()
        puntos += (v == (v.min() if c == "MdMRE" else v.max())).astype(float)
    j = int(np.argmax(puntos))
    print(f"  brazo elegido: {t.brazo.iloc[j]}  (gana {int(puntos[j])} de "
          f"{len(COLS)} metricas)")
    print("  " + t.iloc[j][["MdMRE", "R2_log", "PRED25", "factor2",
                            "MAE_h"]].round(4).to_string().replace("\n", " · "))
    np.save("results/modelo_total/pred_total.npy", P[:, j])
    return t.brazo.iloc[j]


def main():
    t0 = time.time()
    saltar = "--sin-modelo" in sys.argv
    for guion, que in PASOS:
        if guion is None:
            print(f"\n>>> {que}", flush=True)
            elegir()
            continue
        if saltar and guion == "165_total_ampliado.py":
            print(f"\n>>> saltado: {que}", flush=True)
            continue
        print(f"\n>>> {guion} · {que}", flush=True)
        r = subprocess.run([PY, f"src/{guion}"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode:
            print(r.stdout[-1500:])
            print(r.stderr[-1500:])
            print(f"!!! {guion} fallo con codigo {r.returncode}")
            return
        ult = [l for l in (r.stdout or "").splitlines() if l.strip()][-1:]
        print("    " + (ult[0] if ult else "sin salida"))
    print(f"\ncadena rehecha en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
