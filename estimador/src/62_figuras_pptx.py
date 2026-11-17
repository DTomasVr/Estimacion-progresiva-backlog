"""Slide-sized figures for the two decks.

Deliberately not the notebook figures: those are dense because a reader can
lean in, while a slide is read from the back of a room in about eight seconds.
Bigger type, fewer series, one idea per panel.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
SAL = "reports/figuras_pptx"
os.makedirs(SAL, exist_ok=True)

TINTA, SUAVE = "#12645d", "#dcebe9"
AVISO, AMBAR, GRIS, NEGRO = "#a4442a", "#7d5a17", "#6b7a8a", "#111820"
CLARO = "#f1f4f6"

matplotlib.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": "#b9c5ce",
    "axes.labelcolor": "#3d4a58", "axes.titlesize": 17, "axes.titleweight": "600",
    "axes.titlelocation": "left", "axes.titlepad": 12, "axes.labelsize": 14,
    "axes.grid": True, "grid.color": "#e6ebef", "grid.linewidth": 1.0,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": GRIS, "ytick.color": GRIS,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.frameon": False, "legend.fontsize": 13,
    "font.family": "DejaVu Sans", "figure.autolayout": True,
})


def guardar(fig, nombre):
    p = f"{SAL}/{nombre}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ", p)


# ------------------------------------------------------------------ limpieza
def limpieza():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))

    etapas = ["bruto\n20 248", "tras D\n18 593", "tras E\n18 474"]
    vals = [20248, 18593, 18474]
    b = ax[0].bar(etapas, vals, color=[GRIS, TINTA, TINTA], width=.62)
    for r, v in zip(b, vals):
        ax[0].text(r.get_x() + r.get_width() / 2, v + 320, f"{v:,}".replace(",", " "),
                   ha="center", fontsize=14, weight="600", color=NEGRO)
    ax[0].annotate("−1 655", xy=(0.5, 19400), fontsize=15, color=AVISO,
                   weight="700", ha="center")
    ax[0].annotate("−119", xy=(1.5, 18900), fontsize=15, color=AVISO,
                   weight="700", ha="center")
    ax[0].set_ylim(0, 23000)
    ax[0].set_ylabel("filas")
    ax[0].set_title("Dataset 1 · se descarta el 8,8 %")
    ax[0].set_yticks([])

    etapas2 = ["con tiempo\n78 634", "tras D1-D4\n77 546", "tras D5-D6\n72 501"]
    vals2 = [78634, 77546, 72501]
    b = ax[1].bar(etapas2, vals2, color=[GRIS, TINTA, TINTA], width=.62)
    for r, v in zip(b, vals2):
        ax[1].text(r.get_x() + r.get_width() / 2, v + 1300, f"{v:,}".replace(",", " "),
                   ha="center", fontsize=14, weight="600", color=NEGRO)
    ax[1].annotate("−1 088", xy=(0.5, 74800), fontsize=15, color=AVISO,
                   weight="700", ha="center")
    ax[1].annotate("−5 045\nduplicados", xy=(1.5, 71000), fontsize=14, color=AVISO,
                   weight="700", ha="center")
    ax[1].set_ylim(0, 92000)
    ax[1].set_title("Bloque nuevo del Dataset 2 · se descarta el 7,8 %")
    ax[1].set_yticks([])
    guardar(fig, "limpieza")


# --------------------------------------------------------------------- merge
def merge():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def bloque(x, y, w, h, color, texto, sub, tc="white", fs=14):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color,
                                   edgecolor="none"))
        ax.text(x + w / 2, y + h * .60, texto, ha="center", va="center",
                fontsize=fs, weight="700", color=tc)
        ax.text(x + w / 2, y + h * .26, sub, ha="center", va="center",
                fontsize=11.5, color=tc)

    ax.text(0, 9.4, "DATASET 1", fontsize=13, weight="700", color=GRIS)
    bloque(0, 7.2, 34, 1.8, TINTA, "entrenamiento", "10 597")
    bloque(35, 7.2, 12, 1.8, AMBAR, "validación", "2 528")
    bloque(48, 7.2, 20, 1.8, AVISO, "PRUEBA", "4 225")

    ax.annotate("", xy=(58, 6.6), xytext=(58, 7.1),
                arrowprops=dict(arrowstyle="-|>", color=AVISO, lw=2.4))
    ax.text(59.2, 6.75, "queda intacta, fila por fila", fontsize=13,
            color=AVISO, weight="700", va="center")

    ax.text(0, 5.2, "BLOQUE NUEVO · 72 501 tareas de 10 organizaciones",
            fontsize=13, weight="700", color=GRIS)
    bloque(0, 3.0, 55, 1.8, TINTA, "entrenamiento", "53 329")
    bloque(56, 3.0, 8, 1.8, AMBAR, "val.", "5 917")
    bloque(65, 3.0, 22, 1.8, "#3d4a58", "EXTERNO", "13 255 · 150 proyectos enteros")
    ax.text(88.5, 3.9, "nunca se\nentrena con él", fontsize=12, color=GRIS,
            va="center")

    ax.text(0, 1.9, "DATASET 2", fontsize=13, weight="700", color=GRIS)
    bloque(0, 0.0, 62, 1.5, TINTA, "entrenamiento  63 926", "seis veces más que el Dataset 1")
    bloque(63, 0.0, 10, 1.5, AMBAR, "validación", "")
    bloque(74, 0.0, 13, 1.5, AVISO, "PRUEBA", "la misma")
    bloque(88, 0.0, 12, 1.5, "#3d4a58", "EXTERNO", "")
    guardar(fig, "merge")


# ------------------------------------------------------------ distribuciones
def distribuciones():
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    for nombre, etiqueta, color in [
            ("TAWOS", "Dataset 1 · TAWOS", TINTA),
            ("SiP", "Dataset 1 · SiP", AMBAR),
            ("JiraPublico", "bloque nuevo del Dataset 2", AVISO)]:
        e = sw[sw.source == nombre].effort_hours
        ax[0].hist(np.log10(e), bins=55, alpha=.55, density=True, color=color,
                   label=f"{etiqueta} · mediana {e.median():.2f} h")
    ax[0].set_xlabel("log10(horas)")
    ax[0].set_yticks([])
    ax[0].legend(fontsize=12)
    ax[0].set_title("El bloque nuevo registra tareas más pequeñas")

    med = [sw[sw.source == s].effort_hours.median()
           for s in ("TAWOS", "SiP", "JiraPublico")]
    b = ax[1].bar(["TAWOS", "SiP", "bloque\nnuevo"], med,
                  color=[TINTA, AMBAR, AVISO], width=.6)
    for r, v in zip(b, med):
        ax[1].text(r.get_x() + r.get_width() / 2, v + .06, f"{v:.2f} h",
                   ha="center", fontsize=15, weight="700", color=NEGRO)
    ax[1].set_ylim(0, 2.6)
    ax[1].set_ylabel("mediana de horas")
    ax[1].set_title("2,4 veces menor · hay que declararlo")
    guardar(fig, "distribuciones")


# ------------------------------------------------------------------ pipeline
def pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    pasos = [
        ("Texto\nde la tarea", TINTA),
        ("SBERT\nmultilingüe\ncongelado", TINTA),
        ("vector 768\n+ categoría\n+ numéricas", "#3d4a58"),
        ("HistGradient\nBoosting\nsobre log(horas)", "#3d4a58"),
        ("× mediana\ndel proyecto", AMBAR),
        ("horas\n+ intervalo\n+ aviso de riesgo", AVISO),
    ]
    w, hueco = 13.5, 3.8
    x = 0
    for texto, color in pasos:
        ax.add_patch(plt.Rectangle((x, 2.6), w, 4.8, facecolor=color,
                                   edgecolor="none"))
        ax.text(x + w / 2, 5.0, texto, ha="center", va="center", fontsize=12.5,
                weight="600", color="white", linespacing=1.5)
        if x + w + hueco < 100:
            ax.annotate("", xy=(x + w + hueco - .6, 5.0), xytext=(x + w + .6, 5.0),
                        arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=2.2))
        x += w + hueco
    ax.text(0, 1.2, "El codificador NO se reajusta: el ajuste fino con 63 926 filas "
                    "empeora en organizaciones nuevas (ρ = −0,014).",
            fontsize=12.5, color=GRIS)
    guardar(fig, "pipeline")


# ------------------------------------------------------------------ ab y techo
def ab_techo():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    r = pd.read_csv("results/exp_v4.csv")
    r = r[~r.arm.str.contains("linea base")]
    ax[0].scatter(r.MdMRE_val, r.MdMRE_test, s=90, color=TINTA, alpha=.55,
                  edgecolors="none")
    ctrl = r[r.arm.str.startswith("A solo v3")]
    ax[0].scatter(ctrl.MdMRE_val, ctrl.MdMRE_test, s=150, color=AVISO,
                  label="entrenado solo con el Dataset 1", zorder=3)
    ax[0].set_xlabel("MdMRE en validación")
    ax[0].set_ylabel("MdMRE en la prueba fija")
    ax[0].legend(fontsize=12)
    ax[0].set_title("22 configuraciones, todas en 0,68-0,71")
    ax[0].axhspan(0.681, 0.705, color=SUAVE, zorder=0)

    etiquetas = ["solo\nDataset 1", "+ corpus\nnuevo", "+ razón y\ncalibración"]
    vals = [1.1407, 0.7032, 0.5906]
    b = ax[1].bar(etiquetas, vals, color=[AVISO, GRIS, TINTA], width=.6)
    for r_, v in zip(b, vals):
        ax[1].text(r_.get_x() + r_.get_width() / 2, v + .03, f"{v:.4f}",
                   ha="center", fontsize=15, weight="700", color=NEGRO)
    ax[1].axhline(0.75, color=NEGRO, ls="--", lw=1.6)
    ax[1].text(2.42, 0.775, "responder siempre\nla mediana global", fontsize=11.5,
               color=NEGRO, ha="right")
    ax[1].set_ylim(0, 1.32)
    ax[1].set_ylabel("MdMRE  (menos es mejor)")
    ax[1].set_title("150 organizaciones nuevas · mismas 13 255 tareas")
    guardar(fig, "ab_techo")


# ------------------------------------------------------------------- despliegue
def despliegue():
    r = pd.read_csv("results/curva_despliegue.csv")
    r = r[r.MdMRE.notna()]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    x = np.arange(len(r))
    ax[0].plot(x, r.MdMRE, marker="o", ms=11, lw=3, color=TINTA)
    for i, v in enumerate(r.MdMRE):
        ax[0].text(i, v + .022, f"{v:.3f}", ha="center", fontsize=13,
                   weight="700", color=NEGRO)
    ax[0].set_xticks(x, r.tramo)
    ax[0].set_xlabel("tareas ya terminadas en ese proyecto")
    ax[0].set_ylabel("MdMRE")
    ax[0].set_ylim(0.5, 0.95)
    ax[0].set_title("El error baja un 34 % con el uso")

    ax[1].plot(x, r.banda80_recal, marker="o", ms=11, lw=3, color=AMBAR)
    for i, v in enumerate(r.banda80_recal):
        ax[1].text(i, v + 3, f"×{v:.0f}", ha="center", fontsize=13,
                   weight="700", color=NEGRO)
    ax[1].set_xticks(x, r.tramo)
    ax[1].set_xlabel("tareas ya terminadas en ese proyecto")
    ax[1].set_ylabel("anchura de la banda del 80 %")
    ax[1].set_ylim(0, 108)
    ax[1].set_title("Y el intervalo se estrecha de ×91 a ×11")
    guardar(fig, "despliegue")


# ------------------------------------------------------------------ criterios
def criterios():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    nombres = ["mejor modelo\n(régimen 50+)", "modelo en\narranque en frío",
               "experto humano\n(SiP)", "experto humano\n(Jira público)"]
    vals = [0.2390, 0.1294, 0.5093, 0.4019]
    colores = [TINTA, AVISO, AMBAR, GRIS]
    b = ax.bar(nombres, vals, color=colores, width=.58)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + .015, f"{v:.1%}",
                ha="center", fontsize=16, weight="700", color=NEGRO)
    ax.axhline(0.75, color=AVISO, lw=2.6)
    ax.text(3.45, 0.772, "criterio de aceptación · PRED(25) ≥ 75 %",
            ha="right", fontsize=14, weight="700", color=AVISO)
    ax.set_ylim(0, 0.88)
    ax.set_ylabel("PRED(25) · tareas estimadas con ±25 % de error")
    ax.set_title("Nadie llega al umbral — tampoco el experto humano")
    guardar(fig, "criterios")


if __name__ == "__main__":
    print("generando figuras para las presentaciones:")
    limpieza()
    merge()
    distribuciones()
    pipeline()
    ab_techo()
    despliegue()
    criterios()
    print("listo")
