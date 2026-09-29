"""Figuras de la noche: de donde salimos, adonde llegamos y que palanca vale cuanto.

Todas las cifras se leen de los CSV de resultados en tiempo de ejecucion.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
SAL = "reports/figuras_pptx"
os.makedirs(SAL, exist_ok=True)
TINTA, AVISO, AMBAR, GRIS, NEGRO = "#12645d", "#a4442a", "#7d5a17", "#6b7a8a", "#111820"
matplotlib.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#b9c5ce", "axes.labelcolor": "#3d4a58",
    "axes.titlesize": 14, "axes.titleweight": "600", "axes.titlelocation": "left",
    "axes.titlepad": 10, "axes.labelsize": 12, "axes.grid": True, "grid.color": "#e6ebef",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": GRIS, "ytick.color": GRIS, "xtick.labelsize": 11,
    "ytick.labelsize": 11, "legend.frameon": False, "legend.fontsize": 11,
    "font.family": "DejaVu Sans", "figure.autolayout": True,
})


def guardar(fig, nombre):
    p = f"{SAL}/{nombre}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", p)


def palancas():
    t = pd.read_csv("results/palancas/resumen.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.0))
    for a, b, titulo in ((ax[0], "externo no Apache", "Proyectos nuevos (externo no Apache)"),
                         (ax[1], "prueba fija", "Proyectos ya vistos (prueba fija)")):
        v = t[t.bloque == b]
        etiquetas, r2, mdm, col = [], [], [], []
        for _, r in v.iterrows():
            if r.anclas not in (0, 20):
                continue
            nom = ("en frío" if not r.estimacion and r.anclas == 0 else
                   "20 anclas" if not r.estimacion else
                   "estimación del usuario" if r.anclas == 0 else
                   "anclas + estimación")
            etiquetas.append(nom)
            r2.append(r.R2)
            mdm.append(r.MdMRE)
            col.append({"en frío": GRIS, "20 anclas": AMBAR,
                        "estimación del usuario": TINTA,
                        "anclas + estimación": AVISO}[nom])
        x = np.arange(len(etiquetas))
        a.bar(x, r2, .6, color=col)
        for i, (q, m) in enumerate(zip(r2, mdm)):
            a.text(i, q + .012, f"R² {q:.3f}\nMdMRE {m:.3f}", ha="center",
                   fontsize=10.5, weight="600", color=NEGRO)
        a.set_xticks(x, etiquetas, fontsize=10.5)
        a.set_ylim(0, max(r2) * 1.38)
        a.set_ylabel("R² sobre log-horas · más es mejor")
        a.set_title(titulo, fontsize=13)
    fig.suptitle("Qué aporta cada palanca, sobre el mismo modelo y los mismos bloques",
                 x=.01, ha="left", fontsize=15, weight="600")
    guardar(fig, "noche_palancas")


def anclas():
    t = pd.read_csv("results/anclas/contraidas.csv")
    v = t[t.bloque == "externo no Apache"]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    base = float(v[v.anclas == 0].MdMRE.iloc[0])
    for etq, col, est in (("sin contraer", AVISO, "--o"),):
        s = v[v.variante == etq].sort_values("anclas")
        ax.plot(s.anclas, s.MdMRE, est, color=col, lw=2.2, ms=6, label="sin contraer")
    s = v[v.variante.str.startswith("contraído")].sort_values("anclas")
    ax.plot(s.anclas, s.MdMRE, "-o", color=TINTA, lw=2.6, ms=6.5,
            label="con contracción James-Stein (λ=12)")
    ax.axhline(base, color=GRIS, ls=":", lw=1.8)
    ax.text(19.6, base + .004, f"sin anclar  {base:.4f}", fontsize=11, color=GRIS,
            weight="700", ha="right")
    ax.set_xlabel("tareas del proyecto cuyas horas conoce el usuario")
    ax.set_ylabel("MdMRE · menos es mejor")
    ax.legend(loc="upper right")
    ax.set_title("Anclar sin contraer empeora; contrayendo, nunca")
    guardar(fig, "noche_anclas")


def avance():
    c = pd.read_csv("results/codificadores/resumen.csv")
    c = c[c.bloque == "prueba fija"].sort_values("R2")
    e = pd.read_csv("results/ensamblado/resumen.csv")
    e = e[e.bloque == "prueba fija"]
    v2 = pd.read_csv("results/estimador_v2/verificacion.csv")
    v2 = v2[(v2.bloque == "prueba fija") & (v2.subconjunto == "todas")]
    filas = [("punto de partida\nmpnet + HGB", float(c[c.codificador.str.startswith("mpnet")].R2.iloc[0]), GRIS)]
    filas.append(("codificador e5-large", float(c[c.codificador == "e5-large"].R2.iloc[0]), AMBAR))
    filas.append(("+ tf-idf de caracteres\ny de palabras",
                  float(e[e.modelo.str.startswith("1 ·")].R2.iloc[0]), AMBAR))
    filas.append(("+ bolsa de 7 con train+val",
                  float(e[e.modelo.str.startswith("7 ·")].R2.iloc[0]), TINTA))
    filas.append(("estimador v2 desplegado", float(v2.R2.iloc[0]), TINTA))
    d = pd.DataFrame(filas, columns=["etapa", "R2", "col"])
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    x = np.arange(len(d))
    ax.bar(x, d.R2, .62, color=d.col)
    for i, v in enumerate(d.R2):
        ax.text(i, v + .004, f"{v:.3f}", ha="center", fontsize=12, weight="700",
                color=NEGRO)
    ax.set_xticks(x, d.etapa, fontsize=10.5)
    ax.set_ylim(0, max(d.R2) * 1.25)
    ax.set_ylabel("R² sobre log-horas, prueba fija")
    ax.set_title("Lo que ganó la predicción en frío durante la noche")
    guardar(fig, "noche_avance")


if __name__ == "__main__":
    print("figuras:")
    palancas()
    anclas()
    avance()
    print("listo")
