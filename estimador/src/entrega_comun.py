"""Shared helpers for building the two delivery notebooks.

The notebooks are generated rather than hand-written so that the narrative and
the numbers cannot drift apart: every figure and every table in them is produced
by code that runs against the actual files being delivered, and the build fails
loudly if a step no longer reproduces.
"""
import nbformat as nbf

# Una sola definicion de estilo, inyectada al principio de cada cuaderno, para
# que las dos entregas se lean como un mismo documento.
ESTILO = '''
import warnings, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

RAIZ = Path("..") if Path("../datos").exists() else Path(".")
DATOS = RAIZ / "datos"
INTER = DATOS / "intermedios"
FIG = RAIZ / "figuras"
TAB = RAIZ / "tablas"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

TINTA, SUAVE = "#12645d", "#dcebe9"
AVISO, AMBAR, GRIS = "#a4442a", "#7d5a17", "#6b7a8a"
PALETA = [TINTA, AVISO, AMBAR, "#3d4a58", "#54b7ac", "#b9c5ce"]

matplotlib.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#b9c5ce",
    "axes.labelcolor": "#3d4a58", "axes.titlesize": 11, "axes.titleweight": "600",
    "axes.titlelocation": "left", "axes.titlepad": 10, "axes.labelsize": 9,
    "axes.grid": True, "grid.color": "#e3e9ee", "grid.linewidth": .8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": "#6b7a8a", "ytick.color": "#6b7a8a",
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "font.family": "DejaVu Sans", "figure.autolayout": True,
})


def guardar(fig, nombre):
    """Save a figure next to the notebook and report where it went."""
    p = FIG / f"{nombre}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    print(f"figura guardada: {p}")
    return p


def tabla(df, nombre, titulo=None):
    """Persist a table as CSV and echo it, so the notebook and disk agree."""
    p = TAB / f"{nombre}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    if titulo:
        print(titulo)
        print("-" * len(titulo))
    print(df.to_string(index=False))
    print(f"\\ntabla guardada: {p}")
    return df


def sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:n]


def miles(x, _=None):
    return f"{int(x):,}".replace(",", " ")


MILES = FuncFormatter(miles)
'''


def md(texto):
    return nbf.v4.new_markdown_cell(texto.strip("\n"))


def code(texto):
    return nbf.v4.new_code_cell(texto.strip("\n"))


def construir(celdas, titulo):
    nb = nbf.v4.new_notebook()
    nb.cells = celdas
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "title": titulo,
    }
    return nb
