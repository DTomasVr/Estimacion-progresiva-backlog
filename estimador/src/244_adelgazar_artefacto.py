"""Adelgaza el artefacto sin tocar el modelo: 222 MB -> lo que se pueda.

Los codificadores no estan dentro del paquete -se cargan de la cache de HF- asi
que el peso viene de otra parte. Medido:

    svds      114 MB   las dos TruncatedSVD del TF-IDF, en float64
    centros    68 MB   (16648, 1024) float32
    vec_V      34 MB   el banco de vecinos, ya en float16
    bolsa      26 MB   los 5 HGB

Dos cosas sobran:

  centros  no aparece en estimar_tareas, ni en _media_tarea, ni en
           estimar_proyecto, ni en el adaptador del generador. Es el mismo
           banco que vec_V en float32: residuo de una version anterior
  float64  en los componentes del SVD. La transformacion es un producto de
           matrices y float32 sobra; el error relativo que mete esta por debajo
           de 1e-6, varios ordenes por debajo del ruido del propio modelo

Se comprueba que las predicciones NO cambian antes de guardar: si alguna se
mueve mas de 1e-4 en relativo, no se escribe nada.
"""
import io
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, "src")
V6 = Path("models/estimador_pbe_v6.joblib")
TOL = 1e-4
TEXTOS = [
    "Add a filter to the search bar", "Fix login redirect loop",
    "Migrate the user table to the new schema", "Update the API docs",
    "Refactor the whole payment subsystem and add tests",
    "Fix a typo in the footer", "Cache the dashboard queries",
    "Add pagination to the admin list", "Write integration tests for checkout",
    "Investigate the flaky nightly build",
]


def carga():
    import importlib.util
    spec = importlib.util.spec_from_file_location("e6", "src/171_estimador_v6.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    t0 = time.time()
    m = carga()
    est = m.EstimadorV6()
    antes_t = est.estimar_tareas(TEXTOS)
    antes_p = est.estimar_proyecto(TEXTOS, cerradas=[2.0] * 4 + [None] * 6)
    del est
    print(f"referencia tomada · {time.time() - t0:.0f} s", flush=True)

    resp = V6.with_suffix(".joblib.antes_de_adelgazar")
    if not resp.exists():
        shutil.copy2(V6, resp)
        print(f"respaldo en {resp.name}")
    p = dict(joblib.load(V6))
    antes_mb = V6.stat().st_size / 1e6

    quitadas = []
    if "centros" in p:
        quitadas.append(f"centros ({p['centros'].nbytes / 1e6:.0f} MB)")
        del p["centros"]
    ahorro = 0.0
    for sv in p.get("svds", []):
        for a in ("components_", "explained_variance_",
                  "explained_variance_ratio_", "singular_values_"):
            v = getattr(sv, a, None)
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                ahorro += v.nbytes / 2e6
                setattr(sv, a, v.astype(np.float32))
    if ahorro:
        quitadas.append(f"SVD a float32 ({ahorro:.0f} MB)")
    joblib.dump(p, V6, compress=3)
    print("  " + " · ".join(quitadas))
    print(f"  {antes_mb:.0f} MB -> {V6.stat().st_size / 1e6:.0f} MB · "
          f"{time.time() - t0:.0f} s", flush=True)

    # ---- comprobacion: las predicciones no se mueven ----
    m2 = carga()
    est2 = m2.EstimadorV6()
    desp_t = est2.estimar_tareas(TEXTOS)
    desp_p = est2.estimar_proyecto(TEXTOS, cerradas=[2.0] * 4 + [None] * 6)
    peor = 0.0
    for c in ("horas", "valor_probable", "banda_baja", "banda_alta"):
        d = np.max(np.abs(desp_t[c].to_numpy() - antes_t[c].to_numpy()) /
                   np.maximum(np.abs(antes_t[c].to_numpy()), 1e-9))
        peor = max(peor, float(d))
        print(f"  {c:16s} desvio relativo max {d:.2e}")
    for c in ("total_horas", "banda_baja", "banda_alta"):
        d = abs(desp_p[c] - antes_p[c]) / max(abs(antes_p[c]), 1e-9)
        peor = max(peor, float(d))
        print(f"  proyecto.{c:12s} desvio relativo {d:.2e}")

    if peor > TOL:
        shutil.copy2(resp, V6)
        raise SystemExit(f"DESVIO {peor:.2e} > {TOL}: restaurado el respaldo, "
                         "no se ha cambiado nada")
    print(f"\nOK · desvio maximo {peor:.2e}, por debajo de {TOL} · "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
