"""Regenera las tablas del artículo.

    python articulo/reproducir.py           tablas I, III y IV desde los CSV guardados en estimador/results/
    python articulo/reproducir.py --todo    además, vuelve a correr antes los experimentos del total

La Tabla II se lee directamente de results/baselines_literatura.csv y
results/rejilla_valores/contra_literatura.csv. Las cifras del generador están en
generador/results/. El mapa completo de cada cifra a su script está en el README.
"""
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
EST = RAIZ / "estimador"

EXPERIMENTOS = [
    ("275_curva_anclaje_dos_criterios.py", "Tabla I: el total según las tareas cerradas"),
    ("277_extrapolacion_simple.py", "ablación de la contracción (lambda_c = 0)"),
    ("265_avance_orden_real.py", "orden real de cierre"),
    ("268_tres_estados.py", "entradas del usuario"),
    ("278_orden_real_con_estimaciones.py", "orden real con estimaciones (meta del trabajo futuro)"),
]


def corre(script, que, *args):
    print(f"\n== {script} · {que}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, str(EST / "src" / script), *args], cwd=EST, check=True)
    print(f"   listo en {time.time() - t0:.0f} s", flush=True)


def main():
    if "--todo" in sys.argv:
        datos = EST / "data/processed/agix_effort_v4.parquet"
        if not datos.exists():
            sys.exit(f"falta {datos}: constrúyelo desde las fuentes públicas (ver README, «Datos»)")
        for script, que in EXPERIMENTOS:
            corre(script, que)
    corre("273_cuadros_paper.py", "tablas del artículo en LaTeX", "--articulo")
    print(f"\nTablas en {EST / 'results/cuadros_paper/tex'}")


if __name__ == "__main__":
    main()
