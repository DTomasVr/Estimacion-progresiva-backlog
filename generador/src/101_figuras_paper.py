"""Fig. 1 (generation) and Fig. 2 (estimation) of the short paper, as vector PDF.

Version 2, advisor's guide V2 (17/11/2026):
  - width of ONE full column (3,5 in) and lettering readable at that size, unscaled;
  - Fig. 1 with 10 boxes (filter+reassign and translate+repair merged) and LOWER than
    the previous version as it was shown (3,40 in at 70 % = 2,38 in);
  - readable in black and white: each category has its own BORDER, not only its colour;
  - Fig. 2 with the notation lambda_c = 12, lambda_u = 12, lambda'_c = 30.

Legend (border and colour):
  input / output     double border,       blue
  trained model      thick solid border,  green (mT5, Marian, e5 and the heads)
  rules              dashed border,       grey
  post-processing    dotted border,       red

The box labels are in Spanish because the paper is.

  python src/101_figuras_paper.py [output_folder ...]
  FIG_LANG=en python src/101_figuras_paper.py [output_folder ...]   English labels

out: fig1_pipeline_es.pdf and fig2_estimacion.pdf (+ .png for checking and .svg, editable) in each folder;
with FIG_LANG=en, fig1_pipeline_en.pdf and fig2_estimacion_en.pdf (same layout)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

AD = "C:/Users/User/Documents/Programacao/PBE-Dataset/paper"
OUTS = sys.argv[1:] or [f"{AD}/figs", f"{AD}/overleaf/figs"]
LANG = os.environ.get("FIG_LANG", "es")
# English labels for the English version of the paper (same boxes, same layout)
EN = {"Descripción\ndel producto": "Product\ndescription",
      "Traducir\nes→en (Marian)": "Translate\nes→en (Marian)",
      "Plan de roles\ny tokens\nde control": "Role plan\nand control\ntokens",
      "Etapa de\nhistorias (mT5)": "Story\nstage (mT5)",
      "Banco filtrado\ny selección\npor cobertura": "Filtered pool\nand coverage\nselection",
      "Fusión: anclas\nprimero, hasta 6": "Merge: anchors\nfirst, up to 6",
      "Etapa de\ntareas (2 mT5)": "Task\nstage (2 mT5)",
      "Filtrar,\ndeduplicar y\nreasignar tareas": "Filter,\ndeduplicate and\nreassign tasks",
      "Traducir en→es\ny reparar\nléxico": "Translate en→es\nand repair\nwording",
      "Backlog: 12\nhistorias con\nsus tareas": "Backlog: 12\nstories with\ntheir tasks",
      "Tareas del\nbacklog": "Backlog\ntasks",
      "Codificador\ne5-large": "Encoder\ne5-large",
      "Cabezal B:\nmedia": "Head B:\nmean",
      "Anclaje por\nproyecto\nEc. (2)": "Per-project\nanchoring\nEq. (2)",
      "Cabezal A:\nmediana": "Head A:\nmedian",
      "Cerradas y\nestimaciones\ndel usuario": "Closed tasks\nand user\nestimates",
      "Total con\nintervalo\ndel 80 %": "Total with\n80 %\ninterval",
      "entrada /\nsalida": "input /\noutput", "modelo\nentrenado": "trained\nmodel",
      "reglas": "rules", "pos-\nprocesado": "post-\nprocessing"}
tr = (lambda t: EN[t]) if LANG == "en" else (lambda t: t)
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none",
                     "mathtext.fontset": "dejavusans"})

# fill, border, line style, width
EST = {"io": ("#e3ebf6", "#2d4a7c", "-", 0.7),
       "neuronal": ("#d3e5cf", "#2f5a39", "-", 1.6),
       "regla": ("#eeeeec", "#555555", (0, (4, 2)), 0.9),
       "post": ("#f6e0e0", "#8c4646", (0, (1, 1.4)), 1.2)}
NOMBRE = {"io": "entrada /\nsalida", "neuronal": "modelo\nentrenado",
          "regla": "reglas", "post": "pos-\nprocesado"}
ANCHO = 3.5                                  # inches: one IEEE column
W, H = 0.80, 0.58                            # box
XS = (0.44, 1.31, 2.19, 3.06)                # centres of the four columns
TAM, TAM_NOTA = 6.0, 5.2                     # points, at real size


def _rect(ax, x0, y0, w, h, f, b, ls, lw, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.045",
                                linewidth=lw, edgecolor=b, facecolor=f, linestyle=ls, zorder=z))


def marco(ax, x0, y0, w, h, tipo):
    f, b, ls, lw = EST[tipo]
    _rect(ax, x0, y0, w, h, f, b, ls, lw)
    if tipo == "io":                          # double border
        d = 0.035
        _rect(ax, x0 + d, y0 + d, w - 2 * d, h - 2 * d, "none", b, "-", lw, 3)


def caja(ax, c, y, texto, tipo, nota=None):
    x = XS[c]
    marco(ax, x - W / 2, y - H / 2, W, H, tipo)
    ax.text(x, y + (0.075 if nota else 0), tr(texto), ha="center", va="center", fontsize=TAM,
            color="#111111", zorder=4, linespacing=1.12)
    if nota:
        ax.text(x, y - 0.185, nota, ha="center", va="center", fontsize=TAM_NOTA,
                color="#444444", style="italic", zorder=4, linespacing=1.05)
    return dict(x=x, y=y)


def flecha(ax, a, b, lado="h"):
    if lado == "h":
        s = 1 if b["x"] > a["x"] else -1
        p1, p2 = (a["x"] + s * W / 2, a["y"]), (b["x"] - s * W / 2, b["y"])
    elif lado == "v":                         # vertical, top to bottom
        p1, p2 = (a["x"], a["y"] - H / 2), (b["x"], b["y"] + H / 2)
    else:                                     # diagonal, up and to the right
        p1, p2 = (a["x"] + W / 2 - 0.05, a["y"] + H / 2), (b["x"] - W / 2 + 0.05, b["y"] - H / 2)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=6.5,
                                 color="#333333", linewidth=0.8, zorder=1,
                                 shrinkA=0, shrinkB=0))


def leyenda(ax, x, y_top, tipos, paso=0.165):
    for i, t in enumerate(tipos):
        y = y_top - i * paso
        marco(ax, x, y - 0.065, 0.22, 0.13, t)
        ax.text(x + 0.29, y, tr(NOMBRE[t]), fontsize=TAM, va="center", color="#222222",
                linespacing=1.0)


def lienzo(alto):
    fig = plt.figure(figsize=(ANCHO, alto))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, ANCHO)
    ax.set_ylim(0, alto)
    ax.axis("off")
    return fig, ax


def guardar(fig, nombre):
    for out in OUTS:
        os.makedirs(out, exist_ok=True)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(f"{out}/{nombre}.{ext}", dpi=300, facecolor="white")
        print("written", f"{out}/{nombre}.pdf")
    plt.close(fig)


def figura1():
    alto = 2.22
    fig, ax = lienzo(alto)
    y1, y2, y3 = 1.86, 1.11, 0.36
    # row 1, left to right
    d = caja(ax, 0, y1, "Descripción\ndel producto", "io")
    t = caja(ax, 1, y1, "Traducir\nes→en (Marian)", "neuronal")
    p = caja(ax, 2, y1, "Plan de roles\ny tokens\nde control", "regla")
    h = caja(ax, 3, y1, "Etapa de\nhistorias (mT5)", "neuronal")
    flecha(ax, d, t); flecha(ax, t, p); flecha(ax, p, h)
    # row 2, right to left
    s = caja(ax, 3, y2, "Banco filtrado\ny selección\npor cobertura", "regla")
    f = caja(ax, 2, y2, "Fusión: anclas\nprimero, hasta 6", "regla")
    e = caja(ax, 1, y2, "Etapa de\ntareas (2 mT5)", "neuronal")
    flecha(ax, h, s, "v"); flecha(ax, s, f); flecha(ax, f, e)
    # row 3, left to right
    q = caja(ax, 1, y3, "Filtrar,\ndeduplicar y\nreasignar tareas", "post")
    r = caja(ax, 2, y3, "Traducir en→es\ny reparar\nléxico", "post")
    b = caja(ax, 3, y3, "Backlog: 12\nhistorias con\nsus tareas", "io")
    flecha(ax, e, q, "v"); flecha(ax, q, r); flecha(ax, r, b)
    leyenda(ax, 0.06, 1.33, ("io", "neuronal", "regla", "post"), paso=0.30)
    guardar(fig, "fig1_pipeline_en" if LANG == "en" else "fig1_pipeline_es")


def figura2():
    alto = 1.56
    fig, ax = lienzo(alto)
    y1, y2 = 1.20, 0.38
    t = caja(ax, 0, y1, "Tareas del\nbacklog", "io")
    c = caja(ax, 1, y1, "Codificador\ne5-large", "neuronal")
    bm = caja(ax, 2, y1, "Cabezal B:\nmedia", "neuronal")
    n = caja(ax, 3, y1, "Anclaje por\nproyecto\nEc. (2)", "regla")
    flecha(ax, t, c); flecha(ax, c, bm); flecha(ax, bm, n)
    a = caja(ax, 1, y2, "Cabezal A:\nmediana", "neuronal")
    u = caja(ax, 2, y2, "Cerradas y\nestimaciones\ndel usuario", "io")
    o = caja(ax, 3, y2, "Total con\nintervalo\ndel 80 %", "io")
    flecha(ax, c, a, "v"); flecha(ax, u, n, "d"); flecha(ax, n, o, "v")
    leyenda(ax, 0.06, 0.62, ("io", "neuronal", "regla"), paso=0.24)
    guardar(fig, "fig2_estimacion_en" if LANG == "en" else "fig2_estimacion")


if __name__ == "__main__":
    figura1()
    figura2()
