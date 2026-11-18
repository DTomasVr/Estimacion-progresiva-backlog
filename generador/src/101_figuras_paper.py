"""Fig. 1 (generacion) y Fig. 2 (estimacion) del short paper, en PDF vectorial.

Version 2, guia V2 de la asesora (17/11/2026):
  - ancho de UNA columna completa (3,5 in) y letra legible a ese tamano, sin escalar;
  - Fig. 1 con 10 cajas (filtrar+reasignar y traducir+reparar unidas) y MAS BAJA que la
    version anterior tal como se mostraba (3,40 in al 70 % = 2,38 in);
  - legible en blanco y negro: cada categoria tiene su BORDE, no solo su color;
  - Fig. 2 con la notacion lambda_c = 12, lambda_u = 12, lambda'_c = 30.

Leyenda (borde y color):
  entrada / salida   borde doble,       azul
  modelo entrenado   borde solido grueso, verde (mT5, Marian, e5 y los cabezales)
  reglas             borde discontinuo,  gris
  posprocesado       borde punteado,     rojo

  python src/101_figuras_paper.py [carpeta_salida ...]

out: fig1_pipeline_es.pdf y fig2_estimacion.pdf (+ .png de control) en cada carpeta
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

AD = "C:/Users/User/Documents/Programacao/PBE-Dataset/paper"
OUTS = sys.argv[1:] or [f"{AD}/figs", f"{AD}/overleaf/figs"]
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "mathtext.fontset": "dejavusans"})

# relleno, borde, estilo de linea, grosor
EST = {"io": ("#e3ebf6", "#2d4a7c", "-", 0.7),
       "neuronal": ("#d3e5cf", "#2f5a39", "-", 1.6),
       "regla": ("#eeeeec", "#555555", (0, (4, 2)), 0.9),
       "post": ("#f6e0e0", "#8c4646", (0, (1, 1.4)), 1.2)}
NOMBRE = {"io": "entrada /\nsalida", "neuronal": "modelo\nentrenado",
          "regla": "reglas", "post": "pos-\nprocesado"}
ANCHO = 3.5                                  # pulgadas: una columna IEEE
W, H = 0.80, 0.58                            # caja
XS = (0.44, 1.31, 2.19, 3.06)                # centros de las cuatro columnas
TAM, TAM_NOTA = 6.0, 5.2                     # puntos, a tamano real


def _rect(ax, x0, y0, w, h, f, b, ls, lw, z=2):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.045",
                                linewidth=lw, edgecolor=b, facecolor=f, linestyle=ls, zorder=z))


def marco(ax, x0, y0, w, h, tipo):
    f, b, ls, lw = EST[tipo]
    _rect(ax, x0, y0, w, h, f, b, ls, lw)
    if tipo == "io":                          # borde doble
        d = 0.035
        _rect(ax, x0 + d, y0 + d, w - 2 * d, h - 2 * d, "none", b, "-", lw, 3)


def caja(ax, c, y, texto, tipo, nota=None):
    x = XS[c]
    marco(ax, x - W / 2, y - H / 2, W, H, tipo)
    ax.text(x, y + (0.075 if nota else 0), texto, ha="center", va="center", fontsize=TAM,
            color="#111111", zorder=4, linespacing=1.12)
    if nota:
        ax.text(x, y - 0.185, nota, ha="center", va="center", fontsize=TAM_NOTA,
                color="#444444", style="italic", zorder=4, linespacing=1.05)
    return dict(x=x, y=y)


def flecha(ax, a, b, lado="h"):
    if lado == "h":
        s = 1 if b["x"] > a["x"] else -1
        p1, p2 = (a["x"] + s * W / 2, a["y"]), (b["x"] - s * W / 2, b["y"])
    elif lado == "v":                         # vertical, de arriba abajo
        p1, p2 = (a["x"], a["y"] - H / 2), (b["x"], b["y"] + H / 2)
    else:                                     # diagonal hacia arriba a la derecha
        p1, p2 = (a["x"] + W / 2 - 0.05, a["y"] + H / 2), (b["x"] - W / 2 + 0.05, b["y"] - H / 2)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=6.5,
                                 color="#333333", linewidth=0.8, zorder=1,
                                 shrinkA=0, shrinkB=0))


def leyenda(ax, x, y_top, tipos, paso=0.165):
    for i, t in enumerate(tipos):
        y = y_top - i * paso
        marco(ax, x, y - 0.065, 0.22, 0.13, t)
        ax.text(x + 0.29, y, NOMBRE[t], fontsize=TAM, va="center", color="#222222",
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
        for ext in ("pdf", "png"):
            fig.savefig(f"{out}/{nombre}.{ext}", dpi=300, facecolor="white")
        print("escrito", f"{out}/{nombre}.pdf")
    plt.close(fig)


def figura1():
    alto = 2.22
    fig, ax = lienzo(alto)
    y1, y2, y3 = 1.86, 1.11, 0.36
    # fila 1, izquierda a derecha
    d = caja(ax, 0, y1, "Descripción\ndel producto", "io")
    t = caja(ax, 1, y1, "Traducir\nes→en (Marian)", "neuronal")
    p = caja(ax, 2, y1, "Plan de roles\ny tokens\nde control", "regla")
    h = caja(ax, 3, y1, "Etapa de\nhistorias (mT5)", "neuronal")
    flecha(ax, d, t); flecha(ax, t, p); flecha(ax, p, h)
    # fila 2, derecha a izquierda
    s = caja(ax, 3, y2, "Banco filtrado\ny selección\npor cobertura", "regla")
    f = caja(ax, 2, y2, "Fusión: anclas\nprimero, hasta 6", "regla")
    e = caja(ax, 1, y2, "Etapa de\ntareas (2 mT5)", "neuronal")
    flecha(ax, h, s, "v"); flecha(ax, s, f); flecha(ax, f, e)
    # fila 3, izquierda a derecha
    q = caja(ax, 1, y3, "Filtrar,\ndeduplicar y\nreasignar tareas", "post")
    r = caja(ax, 2, y3, "Traducir en→es\ny reparar\nléxico", "post")
    b = caja(ax, 3, y3, "Backlog: 12\nhistorias con\nsus tareas", "io")
    flecha(ax, e, q, "v"); flecha(ax, q, r); flecha(ax, r, b)
    leyenda(ax, 0.06, 1.33, ("io", "neuronal", "regla", "post"), paso=0.30)
    guardar(fig, "fig1_pipeline_es")


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
    guardar(fig, "fig2_estimacion")


if __name__ == "__main__":
    figura1()
    figura2()
