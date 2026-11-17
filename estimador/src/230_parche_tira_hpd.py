"""Anade al artefacto la tira de probabilidad (region HPD) como salida opcional.

E112 midio que la region de densidad predictiva maxima es un 68 % mas estrecha
EN MEDIDA que la banda al 80 %, con la cobertura por tramo mucho mas pareja
(0,45-0,86 contra 0,25-1,00). El precio es que sale fragmentada -43 trozos de
mediana- asi que no sustituye a la banda: se ensena como una tira de
probabilidad en el detalle de la tarea.

Lo que hay que guardar son datos, no modelo:

  bordes_tira    la rejilla en log donde se cuenta la densidad
  marginal_tira  la distribucion del corpus, para suavizar la de los vecinos
  tau_tira       la masa que hay que acumular para cubrir el nivel, calibrada
                 FUERA DE MUESTRA contra el bloque de validacion

El banco de vecinos (vec_V, vec_y) ya esta en el paquete para el canal de
analogia, asi que no hace falta nada mas.
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

sys.path.insert(0, "src")
V6 = Path("models/estimador_pbe_v6.joblib")
CACHE = Path("data/processed/embeddings/codificadores")
N_BIN, SUAVE, K_DENS, NIVEL, LOTE = 240, .25, 200, .8, 1024


def main():
    t0 = time.time()
    resp = V6.with_suffix(".joblib.antes_de_la_tira")
    if not resp.exists():
        shutil.copy2(V6, resp)
        print(f"respaldo en {resp.name}")
    p = dict(joblib.load(V6))
    BV = np.asarray(p["vec_V"], np.float32)
    BV = BV / np.maximum(np.linalg.norm(BV, axis=1, keepdims=True), 1e-9)
    by = np.asarray(p["vec_y"], float)
    ly_b = np.log(np.clip(by, 1e-6, None))
    lo_g, hi_g = np.quantile(ly_b, .001) - .05, np.quantile(ly_b, .999) + .05
    borde = np.linspace(lo_g, hi_g, N_BIN + 1)
    b_b = np.clip(np.digitize(ly_b, borde) - 1, 0, N_BIN - 1)
    marg = np.bincount(b_b, minlength=N_BIN).astype(np.float32)
    marg /= marg.sum()
    print(f"banco {len(by):,} tareas · {N_BIN} bins de "
          f"x{np.exp(borde[1] - borde[0]):.3f} · {time.time() - t0:.0f} s",
          flush=True)

    # calibracion de tau contra el bloque de validacion, que el banco no contiene
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy").astype(np.float32)
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    va = (sw.split_t3 == "val").to_numpy()
    Vv = V[va]
    Vv = Vv / np.maximum(np.linalg.norm(Vv, axis=1, keepdims=True), 1e-9)
    yv = sw.effort_hours.to_numpy(float)[va]
    bv = np.clip(np.digitize(np.log(yv), borde) - 1, 0, N_BIN - 1)
    dens = np.zeros((len(yv), N_BIN), np.float32)
    for i in range(0, len(yv), LOTE):
        sim = Vv[i:i + LOTE] @ BV.T
        vec = np.argpartition(-sim, K_DENS, axis=1)[:, :K_DENS]
        d = np.zeros((vec.shape[0], N_BIN), np.float32)
        fil = np.repeat(np.arange(vec.shape[0]), K_DENS)
        np.add.at(d, (fil, b_b[vec].ravel()), 1.0)
        d /= d.sum(1, keepdims=True)
        dens[i:i + LOTE] = (1 - SUAVE) * d + SUAVE * marg
    print(f"  validacion {len(yv):,} tareas · {time.time() - t0:.0f} s", flush=True)

    ordn = np.argsort(-dens, axis=1)
    acum = np.cumsum(np.take_along_axis(dens, ordn, 1), axis=1)
    rank = np.empty_like(ordn)
    np.put_along_axis(rank, ordn, np.arange(N_BIN)[None, :].repeat(len(yv), 0), 1)
    # la masa que hace falta para incluir el bin verdadero de cada tarea
    necesaria = acum[np.arange(len(yv)), rank[np.arange(len(yv)), bv]]
    tau = float(np.quantile(necesaria, NIVEL))
    cob = float((necesaria <= tau).mean())
    inc = acum <= tau
    anchura = np.exp(inc.sum(1) * (borde[1] - borde[0]))
    print(f"  tau = {tau:.4f} · cubre {cob:.1%} en validacion · "
          f"anchura mediana x{np.median(anchura):.2f}", flush=True)

    p["bordes_tira"] = borde
    p["marginal_tira"] = marg
    p["tau_tira"] = tau
    p["k_dens_tira"] = K_DENS
    p["suave_tira"] = SUAVE
    joblib.dump(p, V6, compress=3)
    print(f"escrito {V6} ({V6.stat().st_size / 1e6:.0f} MB) · "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
