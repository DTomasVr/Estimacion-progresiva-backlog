"""Corrector POR TAREA con la estimacion del usuario, y su banda, para el v6.

E108 midio que la banda por tarea no se puede estrechar con solo el texto -por
acuerdo de vecinos, con cuantiles aprendidos (CQR) o por valor predicho, todas
en x31-33- pero que con la estimacion del usuario baja de x32 a x10,4 cubriendo
el 80,4 %, medido SIN las filas en que el real es la estimacion copiada.

El corrector por tarea que hay en el paquete viene del v3 y espera como entrada
la prediccion del v3, asi que no encaja con el v6. Este es propio:

    log(horas) ~ HGB( vector e5 normalizado , log(estimacion) )

entrenado con todas las tareas no Apache que traen estimacion. La banda se
calibra con el residuo FUERA DE PLIEGUE y solo en las filas limpias: si se
calibrara con las copiadas, cuyo residuo es cero, la banda saldria x6,6 y no
cubriria lo que promete en un uso real.
"""
import os
import shutil
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
V6 = Path("models/estimador_pbe_v6.joblib")
CACHE = Path("data/processed/embeddings/codificadores")
PLIEGUES, NIVEL = 3, .8
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def qconf(x, p, arriba):
    s = np.sort(np.asarray(x, float))
    m = len(s)
    j = (int(np.ceil((m + 1) * p)) if arriba else int(np.floor((m + 1) * p))) - 1
    return float(s[int(np.clip(j, 0, m - 1))])


def main():
    t0 = time.time()
    resp = V6.with_suffix(".joblib.antes_de_corrector_tarea")
    if not resp.exists():
        shutil.copy2(V6, resp)
        print(f"respaldo en {resp.name}")
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy").astype(np.float32)
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    est = sw.HoursEstimate.where(sw.HoursEstimate > 0).to_numpy(float)
    hay = ~np.isnan(est)
    sw, V, est = sw[hay].reset_index(drop=True), V[hay], est[hay]
    y = sw.effort_hours.to_numpy(float)
    ly = np.log(y)
    proy = sw.project.to_numpy()
    X = np.hstack([V, np.log(est)[:, None]]).astype(np.float32)
    limpio = np.abs(est - y) / y > .01
    print(f"{len(y):,} tareas con estimacion · limpias {limpio.mean():.1%} · "
          f"{time.time() - t0:.0f} s", flush=True)

    oof = np.zeros(len(y))
    for a, b in GroupKFold(PLIEGUES).split(X, y, proy):
        oof[b] = HistGradientBoostingRegressor(**PAR).fit(X[a], ly[a]).predict(X[b])
    r = (ly - oof)[limpio]
    a_ = (1 - NIVEL) / 2
    lo, hi = qconf(r, a_, False), qconf(r, 1 - a_, True)
    dentro = ((ly - oof)[limpio] >= lo) & ((ly - oof)[limpio] <= hi)
    print(f"  banda con estimacion: x{np.exp(lo):.3f} a x{np.exp(hi):.3f} · anchura "
          f"x{np.exp(hi - lo):.2f} · cubre {dentro.mean():.1%} en las limpias",
          flush=True)

    final = HistGradientBoostingRegressor(**PAR).fit(X, ly)
    p = dict(joblib.load(V6))
    p["corrector_tarea_v6"] = final
    p["banda_con_estimacion"] = (lo, hi)
    joblib.dump(p, V6, compress=3)
    print(f"escrito {V6} ({V6.stat().st_size / 1e6:.0f} MB) · "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
