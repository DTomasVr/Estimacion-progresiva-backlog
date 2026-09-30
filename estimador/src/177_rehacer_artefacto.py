"""Rehacer el artefacto de servicio despues de reentrenar el codificador ajustado.

Si cambian los pesos del codificador C, cambian sus vectores, y la cabeza que los
consume dentro del apilado (hgb_ajustado_C del v5) quedo ajustada sobre los
ANTIGUOS. Servir con un codificador que no case con su cabeza da predicciones
silenciosamente mal: no falla, solo acierta menos. Asi que la cadena es

    147 (reentrenar C)  ->  163 (rehacer el v5)  ->  171 (rehacer el v6)

y este guion hace los dos ultimos pasos y despues prueba el artefacto de extremo
a extremo, que es lo unico que demuestra que sirve de verdad. La primera vez que
se construyo el v6 parecia correcto y no podia codificar texto nuevo, porque los
pesos del codificador nunca se habian guardado; lo destapo justamente la prueba.

    python src/177_rehacer_artefacto.py
"""
import subprocess
import sys
import time
from pathlib import Path

PASOS = [("163_modelo_v5.py", "apilado v5 con los dos codificadores ajustados"),
         ("171_estimador_v6.py", "estimador v6 de servicio")]
REQUIS = [Path("data/processed/embeddings/codificadores/e5-ajustado-C.npy"),
          Path("models/e5_ajustado_C/config.json")]


def prueba():
    """Cargar el artefacto y estimar de verdad, con y sin tareas cerradas."""
    import importlib.util
    import warnings
    warnings.filterwarnings("ignore")
    s = importlib.util.spec_from_file_location("e", "src/171_estimador_v6.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    est = m.EstimadorV6()
    T = ["Add OAuth2 login with Google", "Fix typo in the login page footer",
         "Migrate the reporting service to the new event bus",
         "Add pagination to the backup report",
         "Write integration tests for the payment module",
         "Update README with deployment steps"]
    import pandas as pd
    pd.set_option("display.width", 200)
    r = est.estimar_tareas(T)
    print(r[["texto", "horas", "valor_probable", "banda_baja",
             "banda_alta"]].round(2).to_string(index=False))
    print()
    for etq, kw in (("en frio", {}),
                    ("con 2 tareas cerradas",
                     dict(cerradas=[12.0, 0.5, None, None, None, None])),
                    ("con estimacion del usuario",
                     dict(estimaciones=[16, 1, 24, 4, 8, 2]))):
        d = est.estimar_proyecto(T, **kw)
        print(f"  {etq:26s} total {d['total_horas']:7.1f} h · "
              f"banda al {d['nivel']:.0%} de {d['banda_baja']:.0f} a "
              f"{d['banda_alta']:.0f} h · cerradas {d['cerradas']}")


def main():
    t0 = time.time()
    falta = [r for r in REQUIS if not r.exists()]
    if falta:
        print("faltan requisitos, no se puede rehacer:")
        for r in falta:
            print("  ", r)
        print("\nhay que correr antes: python src/147_finetune_rapido.py C lote=16")
        return
    for guion, que in PASOS:
        print(f"\n>>> {guion} · {que}", flush=True)
        r = subprocess.run([sys.executable, f"src/{guion}"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            print(r.stdout[-1200:])
            print(r.stderr[-1200:])
            print(f"!!! {guion} fallo con codigo {r.returncode}")
            return
        ult = [l for l in (r.stdout or "").splitlines() if l.strip()][-1:]
        print("    " + (ult[0] if ult else "sin salida"))
    print("\n>>> prueba de extremo a extremo", flush=True)
    prueba()
    print(f"\nartefacto rehecho y probado en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
