"""Slide primitives for the two decks.

One layout vocabulary shared by both files: title, section divider, bullets,
table, figure, and a "verdict" band for the one sentence a slide exists to
deliver. Everything is positioned explicitly in EMU rather than through
placeholders, because PowerPoint's own layouts differ between installs and a
deck that reflows on the examiner's laptop is worse than a plain one.

Palette and type follow the written report, so the slides and the document read
as one piece of work.
"""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ANCHO, ALTO = Inches(13.333), Inches(7.5)
MARGEN = Inches(0.62)
UTIL = ANCHO - 2 * MARGEN

TINTA = RGBColor(0x12, 0x64, 0x5D)
TINTA_OSC = RGBColor(0x0C, 0x4A, 0x45)
SUAVE = RGBColor(0xDC, 0xEB, 0xE9)
NEGRO = RGBColor(0x11, 0x18, 0x20)
GRIS = RGBColor(0x6B, 0x7A, 0x8A)
GRIS_CLARO = RGBColor(0xF1, 0xF4, 0xF6)
REGLA = RGBColor(0xD5, 0xDD, 0xE4)
AVISO = RGBColor(0xA4, 0x44, 0x2A)
AVISO_SUAVE = RGBColor(0xF4, 0xE2, 0xDC)
AMBAR = RGBColor(0x7D, 0x5A, 0x17)
AMBAR_SUAVE = RGBColor(0xF2, 0xE9, 0xD5)
BLANCO = RGBColor(0xFF, 0xFF, 0xFF)

TITULO_F = "Georgia"
CUERPO_F = "Calibri"
MONO_F = "Consolas"


def nueva():
    p = Presentation()
    p.slide_width, p.slide_height = ANCHO, ALTO
    return p


def _vacia(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _caja(s, x, y, cx, cy, texto="", tam=18, color=NEGRO, negrita=False,
          fuente=CUERPO_F, alineado=PP_ALIGN.LEFT, interlineado=1.0,
          anclaje=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(x, y, cx, cy)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anclaje
    p = tf.paragraphs[0]
    p.alignment = alineado
    p.line_spacing = interlineado
    r = p.add_run()
    r.text = texto
    r.font.size = Pt(tam)
    r.font.bold = negrita
    r.font.color.rgb = color
    r.font.name = fuente
    return tb


def _rect(s, x, y, cx, cy, relleno=None, borde=None, grosor=Pt(1)):
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, cx, cy)
    if relleno is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = relleno
    if borde is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = borde
        sh.line.width = grosor
    sh.shadow.inherit = False
    return sh


# ------------------------------------------------------------------ portadas
def portada(prs, titulo, subtitulo, pie=""):
    s = _vacia(prs)
    _rect(s, 0, 0, ANCHO, ALTO, relleno=TINTA)
    _rect(s, MARGEN, Inches(2.55), Inches(1.5), Pt(4), relleno=BLANCO)
    _caja(s, MARGEN, Inches(1.5), UTIL, Inches(1.0),
          "PBE · COMPONENTE DE ESTIMACIÓN DE ESFUERZO", tam=13,
          color=SUAVE, negrita=True)
    _caja(s, MARGEN, Inches(2.95), UTIL, Inches(2.0), titulo, tam=44,
          color=BLANCO, negrita=True, fuente=TITULO_F, interlineado=1.08)
    _caja(s, MARGEN, Inches(4.75), Inches(9.5), Inches(1.2), subtitulo,
          tam=19, color=SUAVE, interlineado=1.25)
    if pie:
        _caja(s, MARGEN, ALTO - Inches(1.0), UTIL, Inches(0.4), pie, tam=13,
              color=SUAVE)
    return s


def seccion(prs, numero, titulo, resumen=""):
    s = _vacia(prs)
    _rect(s, 0, 0, ANCHO, ALTO, relleno=GRIS_CLARO)
    _rect(s, 0, 0, Pt(10), ALTO, relleno=TINTA)
    _caja(s, MARGEN, Inches(2.6), Inches(1.2), Inches(0.9), numero, tam=54,
          color=SUAVE, negrita=True, fuente=TITULO_F)
    _caja(s, MARGEN + Inches(1.35), Inches(2.7), Inches(10), Inches(1.2),
          titulo, tam=36, color=TINTA_OSC, negrita=True, fuente=TITULO_F)
    if resumen:
        _caja(s, MARGEN + Inches(1.35), Inches(4.0), Inches(9.6), Inches(1.4),
              resumen, tam=18, color=GRIS, interlineado=1.3)
    return s


# ------------------------------------------------------------- lamina normal
def lamina(prs, titulo, entradilla=""):
    s = _vacia(prs)
    _rect(s, 0, 0, ANCHO, ALTO, relleno=BLANCO)
    _caja(s, MARGEN, Inches(0.48), UTIL, Inches(0.7), titulo, tam=30,
          color=TINTA_OSC, negrita=True, fuente=TITULO_F)
    y = Inches(1.22)
    _rect(s, MARGEN, y, Inches(1.1), Pt(3), relleno=TINTA)
    y = Inches(1.48)
    if entradilla:
        _caja(s, MARGEN, y, UTIL, Inches(0.9), entradilla, tam=17,
              color=GRIS, interlineado=1.25)
        y = y + Inches(0.36) * (1 + entradilla.count("\n")) + Inches(0.34)
    return s, y


def vinetas(s, y, items, tam=17, x=None, ancho=None, salto=Inches(0.5)):
    """items: str, or (texto, resaltado) where the highlight is a leading chip."""
    x = MARGEN if x is None else x
    ancho = UTIL if ancho is None else ancho
    for it in items:
        etiqueta, texto = (it if isinstance(it, tuple) else (None, it))
        if etiqueta:
            w = Inches(0.42 + 0.115 * len(etiqueta))
            _rect(s, x, y + Emu(int(Pt(2))), w, Inches(0.29), relleno=SUAVE)
            _caja(s, x, y + Emu(int(Pt(3.5))), w, Inches(0.26), etiqueta,
                  tam=12, color=TINTA_OSC, negrita=True,
                  alineado=PP_ALIGN.CENTER, fuente=MONO_F)
            dx = w + Inches(0.18)
        else:
            _rect(s, x + Inches(0.03), y + Inches(0.11), Pt(5), Pt(5),
                  relleno=TINTA)
            dx = Inches(0.28)
        n = max(1, len(texto) // 105 + texto.count("\n") + 1)
        _caja(s, x + dx, y, ancho - dx, Inches(0.36) * n, texto, tam=tam,
              color=NEGRO, interlineado=1.22)
        y = y + max(salto, Inches(0.30) * n)
    return y


def tabla(s, y, cabecera, filas, anchos=None, x=None, ancho_total=None,
          tam=14, resaltar=None, alto_fila=Inches(0.36)):
    """resaltar: {indice_fila: color de fondo}"""
    x = MARGEN if x is None else x
    ancho_total = UTIL if ancho_total is None else ancho_total
    n_col = len(cabecera)
    if anchos is None:
        anchos = [1] * n_col
    total = sum(anchos)
    anchos = [Emu(int(ancho_total * a / total)) for a in anchos]
    alto_cab = Inches(0.42)
    forma = s.shapes.add_table(len(filas) + 1, n_col, x, y, ancho_total,
                               alto_cab + alto_fila * len(filas))
    t = forma.table
    t.first_row = True
    for i, w in enumerate(anchos):
        t.columns[i].width = w
    t.rows[0].height = alto_cab
    for j, txt in enumerate(cabecera):
        c = t.cell(0, j)
        c.text = str(txt)
        c.fill.solid()
        c.fill.fore_color.rgb = TINTA
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
        c.margin_left = c.margin_right = Inches(0.09)
        p = c.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.RIGHT if j and str(txt)[:1] != "_" else PP_ALIGN.LEFT
        for r in p.runs:
            r.font.size = Pt(tam - 1)
            r.font.bold = True
            r.font.color.rgb = BLANCO
            r.font.name = CUERPO_F
    for i, fila in enumerate(filas):
        t.rows[i + 1].height = alto_fila
        fondo = (resaltar or {}).get(i)
        for j, txt in enumerate(fila):
            c = t.cell(i + 1, j)
            c.text = str(txt)
            c.fill.solid()
            c.fill.fore_color.rgb = (fondo if fondo else
                                     (BLANCO if i % 2 else GRIS_CLARO))
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.09)
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if j else PP_ALIGN.LEFT
            for r in p.runs:
                r.font.size = Pt(tam)
                r.font.color.rgb = NEGRO
                r.font.name = MONO_F if j else CUERPO_F
                r.font.bold = bool(fondo) and j == 0
    return y + alto_cab + alto_fila * len(filas)


def veredicto(s, y, etiqueta, texto, tono="tinta", ancho=None, x=None):
    x = MARGEN if x is None else x
    ancho = UTIL if ancho is None else ancho
    fondo, barra, tint = {
        "tinta": (SUAVE, TINTA, TINTA_OSC),
        "aviso": (AVISO_SUAVE, AVISO, AVISO),
        "ambar": (AMBAR_SUAVE, AMBAR, AMBAR),
    }[tono]
    n = max(1, len(texto) // 108 + 1)
    alto = Inches(0.52) + Inches(0.29) * n
    _rect(s, x, y, ancho, alto, relleno=fondo)
    _rect(s, x, y, Pt(5), alto, relleno=barra)
    _caja(s, x + Inches(0.22), y + Inches(0.13), ancho - Inches(0.44),
          Inches(0.25), etiqueta.upper(), tam=11.5, color=tint, negrita=True)
    _caja(s, x + Inches(0.22), y + Inches(0.44), ancho - Inches(0.44),
          Inches(0.29) * n, texto, tam=16, color=NEGRO, interlineado=1.22)
    return y + alto


def cifra(s, x, y, cx, valor, etiqueta, tono="tinta", tam=40):
    color = {"tinta": TINTA_OSC, "aviso": AVISO, "ambar": AMBAR,
             "gris": GRIS}[tono]
    _rect(s, x, y, cx, Inches(1.32), relleno=GRIS_CLARO)
    _rect(s, x, y, cx, Pt(3), relleno=color)
    _caja(s, x, y + Inches(0.20), cx, Inches(0.62), valor, tam=tam,
          color=color, negrita=True, alineado=PP_ALIGN.CENTER, fuente=TITULO_F)
    _caja(s, x + Inches(0.12), y + Inches(0.92), cx - Inches(0.24),
          Inches(0.34), etiqueta, tam=12.5, color=GRIS,
          alineado=PP_ALIGN.CENTER, interlineado=1.1)


def figura(s, ruta, x, y, ancho):
    return s.shapes.add_picture(str(ruta), x, y, width=ancho)


def figura_ajustada(s, ruta, y, alto_max, ancho_max=None, x=None):
    """Fit a figure inside the space that is actually left, and centre it.

    Slides overflow silently in PowerPoint -- the picture just runs off the
    bottom -- so the width is derived from the height available rather than
    assumed.
    """
    from PIL import Image
    ancho_max = UTIL if ancho_max is None else ancho_max
    x = MARGEN if x is None else x
    with Image.open(ruta) as im:
        prop = im.width / im.height
    ancho = min(ancho_max, Emu(int(alto_max * prop)))
    alto = Emu(int(ancho / prop))
    return s.shapes.add_picture(str(ruta), x + (ancho_max - ancho) // 2, y,
                                width=ancho), y + alto


def nota(s, texto, y=None):
    y = ALTO - Inches(0.62) if y is None else y
    _caja(s, MARGEN, y, UTIL, Inches(0.34), texto, tam=12, color=GRIS)


def numerar(prs, desde=2):
    for i, s in enumerate(prs.slides, start=1):
        if i < desde:
            continue
        _caja(s, ANCHO - Inches(1.15), ALTO - Inches(0.55), Inches(0.6),
              Inches(0.3), str(i), tam=12, color=GRIS,
              alineado=PP_ALIGN.RIGHT)
