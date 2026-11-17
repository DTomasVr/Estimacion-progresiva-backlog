"""E74 - ¿Hace falta el factor de estilo tambien para el TOTAL del proyecto?

Al enganchar el generador de backlog con el estimador aparecio un hueco: la cabeza
de la MEDIANA (estimar_tareas) multiplica su salida por el factor de estilo 0,831,
porque el texto del generador es mas escueto que un titulo de Jira e infla las
predicciones; pero la cabeza de la MEDIA, que es de donde sale el total del
proyecto, NO lo aplica. Y el total es justo el numero que se presenta.

El factor 0,831 se midio sobre la cabeza de la mediana (E87/E89). Transferirlo sin
mas a la otra cabeza seria suponer que el efecto es el mismo, y no tiene por que:
son dos modelos con perdidas distintas. Asi que se mide.

Los datos ya existen: results/reescritura/reescrituras.jsonl tiene 800 tareas reales
con su titulo ORIGINAL y el mismo contenido REESCRITO al estilo del generador. Se
pasan las dos versiones por la cabeza de la media y se compara.

El estadistico es el COCIENTE DE SUMAS, no la mediana de cocientes, porque lo que
se quiere corregir es un total.

    factor = (suma de predicciones del ORIGINAL) / (suma de predicciones del REESCRITO)

Se evalua solo sobre tareas que NO estan en el entrenamiento del modelo.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "8"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
OUT = Path("results/estilo_total")
S = "=" * 100


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    filas = [json.loads(l) for l in
             open("results/reescritura/reescrituras.jsonl", encoding="utf-8") if l.strip()]
    re_ = pd.DataFrame(filas).drop_duplicates("task_id")
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1][["task_id", "effort_hours", "split_t3",
                                       "fuente_jira"]]
    d = re_.merge(sw, on="task_id", how="inner")
    print(f"{len(re_)} reescrituras · {len(d)} con horas reales · "
          f"{time.time() - t0:.0f} s")
    print("  reparto por particion:", d.split_t3.value_counts().to_dict())
    fuera = d[d.split_t3 != "train"].reset_index(drop=True)
    if len(fuera) < 50:
        print(f"\n  OJO: solo {len(fuera)} fuera de entrenamiento; se informan las dos")
    print(f"  se evalua sobre {len(fuera)} tareas fuera del entrenamiento", flush=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location("e6", "src/171_estimador_v6.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    est = m.EstimadorV6()
    print(f"  estimador cargado · {time.time() - t0:.0f} s", flush=True)

    def mide(sub, etq):
        real = sub.effort_hours.to_numpy(float)
        po = np.asarray(est._media_tarea(sub.original.tolist()), float)
        pr = np.asarray(est._media_tarea(sub.reescrito.tolist()), float)
        f = float(po.sum() / pr.sum())
        return dict(bloque=etq, n=len(sub),
                    sesgo_original=float(po.sum() / real.sum()),
                    sesgo_reescrito=float(pr.sum() / real.sum()),
                    factor=f,
                    mediana_cociente=float(np.median(po / pr)),
                    horas_reales=float(real.sum()),
                    horas_original=float(po.sum()), horas_reescrito=float(pr.sum()))

    filas_r = [mide(fuera, "fuera de entrenamiento")]
    print(f"  bloque 1 listo · {time.time() - t0:.0f} s", flush=True)
    if (d.split_t3 == "train").any():
        filas_r.append(mide(d, "todas (incluye entrenamiento)"))
    t = pd.DataFrame(filas_r)
    print(f"\n{S}\nEL FACTOR DE ESTILO PARA LA CABEZA DE LA MEDIA\n{S}")
    print(t.round(4).to_string(index=False))

    f = float(t.iloc[0].factor)
    so, sr = float(t.iloc[0].sesgo_original), float(t.iloc[0].sesgo_reescrito)
    print(f"\n{S}\nLECTURA\n{S}")
    print(f"  sobre el texto ORIGINAL   la suma predicha es el {so:.3f} de la real")
    print(f"  sobre el texto REESCRITO  la suma predicha es el {sr:.3f} de la real")
    print(f"  -> el estilo del generador multiplica el total por {sr / so:.3f}")
    print(f"  -> el factor que lo corrige es {f:.4f}")
    print(f"\n  el de la cabeza de la MEDIANA, que ya esta en el artefacto, es "
          f"{est.p['factor_estilo']:.4f}")
    dif = abs(f - float(est.p["factor_estilo"]))
    print("  " + ("los dos factores son practicamente el mismo: transferirlo estaba "
                  "justificado" if dif < .06 else
                  f"NO son el mismo (difieren {dif:.3f}): hay que usar el medido aqui "
                  f"para el total"))
    if f > 0.98:
        print("  AVISO: el factor es ~1, asi que para el TOTAL el estilo no infla nada "
              "y no hay que corregir")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"factor_total": f, "factor_mediana": float(est.p["factor_estilo"]),
               "n": int(t.iloc[0].n)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
