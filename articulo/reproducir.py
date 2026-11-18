"""Regenerates the paper tables.

    python articulo/reproducir.py           Tables I, III and IV from the CSV files saved in estimador/results/
    python articulo/reproducir.py --todo    also reruns the experiments on the total first

Table II is read directly from results/baselines_literatura.csv and
results/rejilla_valores/contra_literatura.csv. The generator figures are in
generador/results/. The full map from each figure to its script is in the README.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EST = ROOT / "estimador"

EXPERIMENTS = [
    ("275_curva_anclaje_dos_criterios.py", "Table I: the total by closed tasks"),
    ("277_extrapolacion_simple.py", "ablation of the shrinkage (lambda_c = 0)"),
    ("265_avance_orden_real.py", "real closing order"),
    ("268_tres_estados.py", "user inputs"),
    ("278_orden_real_con_estimaciones.py", "real order with estimates (future-work goal)"),
]


def run(script, what, *args):
    print(f"\n== {script} · {what}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, str(EST / "src" / script), *args], cwd=EST, check=True)
    print(f"   done in {time.time() - t0:.0f} s", flush=True)


def main():
    if "--todo" in sys.argv:
        data = EST / "data/processed/pbe_effort_v4.parquet"
        if not data.exists():
            sys.exit(f"{data} is missing: build it from the public sources (see README, 'Data')")
        for script, what in EXPERIMENTS:
            run(script, what)
    run("273_cuadros_paper.py", "paper tables in LaTeX", "--articulo")
    print(f"\nTables in {EST / 'results/cuadros_paper/tex'}")


if __name__ == "__main__":
    main()
