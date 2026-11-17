"""Calibra la ventana adaptativa contra el artefacto y la compara con su banda.

E113 midio sobre el modelo limpio que la ventana contigua mas corta de la
densidad condicional estrecha un 7 % en la mediana, pero sobre todo ADAPTA: el
percentil 10 de anchura baja de x29,9 a x16,7-20,5. Para las tareas que el
modelo entiende, la banda casi se parte por la mitad.

Aqui se calibra contra el artefacto de verdad, no contra el modelo limpio, y se
compara con su banda actual -el factor simetrico x4,3157, o sea x18,62- sobre
el bloque de VALIDACION, que el banco de vecinos no contiene. Se reutiliza todo
lo que ya guardo la tira (bordes, marginal, K, suavizado), asi que solo se
anade tau_ventana.

Se mide tambien sobre PRUEBA, para que el numero que se reporte no sea el de
calibracion.
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
NIVEL, LOTE = .8, 1024


def ventanas(dens, tau):
    n, B = dens.shape
    ini, fin = np.zeros(n, int), np.full(n, B - 1)
    c = np.cumsum(dens, axis=1)
    ar = np.arange(B)
    for i in range(n):
        ci = c[i]
        j = np.searchsorted(ci, np.r_[0.0, ci[:-1]] + tau, side="left")
        anch = np.where(j < B, j - ar, B + 1)
        k = int(np.argmin(anch))
        ini[i], fin[i] = k, min(j[k], B - 1)
    return ini, fin


def main():
    t0 = time.time()
    resp = V6.with_suffix(".joblib.antes_de_la_ventana")
    if not resp.exists():
        shutil.copy2(V6, resp)
        print(f"respaldo en {resp.name}")
    p = dict(joblib.load(V6))
    borde = p["bordes_tira"]
    marg = p["marginal_tira"]
    nb = len(marg)
    paso = float(borde[1] - borde[0])
    K, SU = p["k_dens_tira"], p["suave_tira"]
    BV = np.asarray(p["vec_V"], np.float32)
    BV = BV / np.maximum(np.linalg.norm(BV, axis=1, keepdims=True), 1e-9)
    b_b = np.clip(np.digitize(np.log(np.clip(
        np.asarray(p["vec_y"], float), 1e-6, None)), borde) - 1, 0, nb - 1)

    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy").astype(np.float32)
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]

    def densidad(mask):
        Vv = V[mask]
        Vv = Vv / np.maximum(np.linalg.norm(Vv, axis=1, keepdims=True), 1e-9)
        d_ = np.zeros((len(Vv), nb), np.float32)
        for i in range(0, len(Vv), LOTE):
            sim = Vv[i:i + LOTE] @ BV.T
            vec = np.argpartition(-sim, K, axis=1)[:, :K]
            dd = np.zeros((vec.shape[0], nb), np.float32)
            fil = np.repeat(np.arange(vec.shape[0]), K)
            np.add.at(dd, (fil, b_b[vec].ravel()), 1.0)
            dd /= dd.sum(1, keepdims=True)
            d_[i:i + LOTE] = (1 - SU) * dd + SU * marg
        return d_

    out = {}
    for blo in ("val", "test"):
        m = (sw.split_t3 == blo).to_numpy()
        yb = sw.effort_hours.to_numpy(float)[m]
        out[blo] = (densidad(m), yb,
                    np.clip(np.digitize(np.log(yb), borde) - 1, 0, nb - 1))
        print(f"  {blo}: {m.sum():,} tareas · {time.time() - t0:.0f} s", flush=True)

    dv, yv, bv = out["val"]
    lo_t, hi_t = .30, .9995
    for _ in range(14):
        md = (lo_t + hi_t) / 2
        i0, i1 = ventanas(dv, md)
        if ((bv >= i0) & (bv <= i1)).mean() < NIVEL:
            lo_t = md
        else:
            hi_t = md
    tau = hi_t
    dt, yt, bt = out["test"]
    i0, i1 = ventanas(dt, tau)
    cob = float(((bt >= i0) & (bt <= i1)).mean())
    anch = np.exp((i1 - i0 + 1) * paso)
    k80 = float(p["factor_banda80"])
    print(f"\n  tau = {tau:.4f} (calibrado en validacion)")
    print(f"  EN PRUEBA ({len(yt):,} tareas):")
    print(f"    banda de hoy      x{k80 ** 2:6.2f} fija")
    print(f"    ventana           x{np.median(anch):6.2f} mediana · "
          f"p10 x{np.quantile(anch, .1):.2f} · p90 x{np.quantile(anch, .9):.2f} · "
          f"cubre {cob:.1%}")
    print(f"    tareas con banda MAS ESTRECHA que hoy: "
          f"{float((anch < k80 ** 2).mean()):.1%}")

    p["tau_ventana"] = float(tau)
    joblib.dump(p, V6, compress=3)
    print(f"\nescrito {V6} ({V6.stat().st_size / 1e6:.0f} MB) · "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
