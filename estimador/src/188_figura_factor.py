"""Las dos figuras que explican el factor de anclaje: como se calcula y que consigue.

    python src/188_figura_factor.py

Todas las cifras salen de results/: ejemplo_anclaje (E75), anclas_escala (E67) y
horas_del_equipo (E76).
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
SAL = "reports/figuras_metricas"
os.makedirs(SAL, exist_ok=True)
TINTA, AVISO, AMBAR, GRIS, NEGRO = "#12645d", "#a4442a", "#7d5a17", "#6b7a8a", "#111820"
matplotlib.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#b9c5ce", "axes.labelcolor": "#3d4a58",
    "axes.titlesize": 13, "axes.titleweight": "600", "axes.titlelocation": "left",
    "axes.titlepad": 9, "axes.labelsize": 11.5, "axes.grid": True, "grid.color": "#e6ebef",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": GRIS, "ytick.color": GRIS, "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5, "legend.frameon": False, "legend.fontsize": 10.5,
    "font.family": "DejaVu Sans", "figure.autolayout": True,
})
NL = chr(10)
LAM = 12


def guardar(fig, nombre):
    p = f"{SAL}/{nombre}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", p)


def como_se_calcula():
    """De donde sale el factor y como la contraccion lo frena."""
    e = pd.read_csv("results/ejemplo_anclaje/resumen.csv")
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.6, 4.3))

    # 1 - la contraccion: que fraccion de la correccion se aplica segun k
    k = np.arange(0, 31)
    a1.plot(k, k / (k + LAM), "-", color=TINTA, lw=2.6)
    for kk in (1, 3, 5, 10, 20):
        f = kk / (kk + LAM)
        a1.plot([kk], [f], "o", color=AMBAR, ms=8, zorder=3)
        a1.annotate(f"{f:.0%}", (kk, f), textcoords="offset points", xytext=(6, -11),
                    fontsize=10, color=AMBAR)
    a1.set_xlabel("tareas cerradas (k)")
    a1.set_ylabel("fraccion de la correccion que se aplica")
    a1.set_ylim(0, 1)
    a1.set_title(f"1 · La contraccion k/(k+{LAM}){NL}"
                 "con una tarea solo se cree el 8 %", fontsize=12)

    # 2 - el factor en el proyecto de ejemplo
    a2.axhline(1.0, color=GRIS, lw=1.4, ls="--")
    a2.text(len(e) - 1, 1.004, "sin corregir", color=GRIS, ha="right", fontsize=10)
    x = np.arange(len(e))
    a2.plot(x, e.factor, "o-", color=AVISO, lw=2.6, ms=8)
    for xi, f in zip(x, e.factor):
        a2.annotate(f"{f:.3f}".replace(".", ","), (xi, f), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9.5, color=AVISO)
    a2.set_xticks(x)
    a2.set_xticklabels([str(int(c)) for c in e.cerradas])
    a2.set_xlabel("tareas cerradas")
    a2.set_ylabel("factor que multiplica a las abiertas")
    a2.set_ylim(.68, 1.06)
    a2.set_title(f"2 · El factor en un proyecto real{NL}"
                 "este equipo va mas rapido de lo previsto", fontsize=12)

    # 3 - la prediccion de UNA tarea: cruda contra corregida
    cruda = float(e.horas_tarea.iloc[0])
    a3.axhline(cruda, color=TINTA, lw=2.2, ls="--")
    a3.text(0, cruda + .06, f"lo que el modelo predice: {cruda:.2f} h".replace(".", ","),
            color=TINTA, fontsize=10.5, va="bottom")
    a3.plot(x, e.horas_tarea, "o-", color=AMBAR, lw=2.6, ms=8)
    a3.set_xticks(x)
    a3.set_xticklabels([str(int(c)) for c in e.cerradas])
    a3.set_xlabel("tareas cerradas")
    a3.set_ylabel("horas estimadas de esa tarea")
    a3.set_ylim(min(e.horas_tarea) - .35, cruda + .45)
    a3.set_title(f"3 · Para una tarea concreta{NL}"
                 "el modelo no cambia; cambia el factor", fontsize=12)
    fig.suptitle("Como se calcula el factor de anclaje y que le hace a una tarea",
                 x=.004, ha="left", fontsize=14.5, weight="600")
    guardar(fig, "12_factor_como_se_calcula")


def que_consigue():
    """Que gana el total, y de donde TIENEN que salir esas horas."""
    an = pd.read_csv("results/anclas_escala/resumen.csv").set_index("k")
    eq = pd.read_csv("results/horas_del_equipo/resumen.csv")
    e = pd.read_csv("results/ejemplo_anclaje/resumen.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.0, 4.6))

    x = np.arange(len(e))
    a1.plot(x, e.error_pc, "o-", color=TINTA, lw=2.6, ms=8)
    for xi, v in zip(x, e.error_pc):
        a1.annotate(f"{v:.1f} %".replace(".", ","), (xi, v),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=10, color=TINTA)
    a1.set_xticks(x)
    a1.set_xticklabels([str(int(c)) for c in e.cerradas])
    a1.set_xlabel("tareas cerradas")
    a1.set_ylabel("error del total del proyecto (%)")
    a1.set_ylim(0, max(e.error_pc) * 1.22)
    a1.set_title(f"El total del proyecto de ejemplo{NL}"
                 f"del {e.error_pc.iloc[0]:.0f} % al {e.error_pc.iloc[-1]:.1f} % "
                 "de error".replace(".", ","))

    ks = [1, 3, 5, 10, 20]
    frio = float(eq[eq.k == 0].PRED25.iloc[0])
    brazos = [("B · del PROPIO proyecto", TINTA, "o-"),
              ("C · de otro proyecto del MISMO equipo", AMBAR, "s--"),
              ("D · de otra organizacion", AVISO, "v:")]
    a2.axhline(frio, color=GRIS, lw=1.8, ls="--")
    a2.text(len(ks) - 1, frio + .006, "sin pedir nada (en frio)", color=GRIS,
            ha="right", fontsize=10)
    for pref, col, est in brazos:
        v = [float(eq[(eq.k == k) & eq.brazo.str.startswith(pref[0])].PRED25.iloc[0])
             for k in ks]
        a2.plot(range(len(ks)), v, est, color=col, lw=2.4, ms=7, label=pref)
    a2.set_xticks(range(len(ks)))
    a2.set_xticklabels([str(k) for k in ks])
    a2.set_xlabel("horas reales que se aportan")
    a2.set_ylabel("proyectos dentro del ±25 %")
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:.0%}"))
    a2.legend(loc="upper left", fontsize=9.8)
    a2.set_title(f"Pero solo valen las del PROPIO proyecto{NL}"
                 "las de otros proyectos dejan PEOR que no pedir nada")
    fig.suptitle("Que consigue el factor, y de donde tienen que salir las horas",
                 x=.004, ha="left", fontsize=14.5, weight="600")
    guardar(fig, "13_factor_que_consigue")


if __name__ == "__main__":
    for f in (como_se_calcula, que_consigue):
        try:
            f()
        except Exception as e:                       # noqa: BLE001
            print(f"  {f.__name__} FALLO: {type(e).__name__}: {e}")
