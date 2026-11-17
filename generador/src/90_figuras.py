"""Figuras de la arquitectura para la tesis, en SVG (vectorial) y PNG (300 dpi).

Las mismas que estan como Mermaid en la Parte A de DISENO_PIPELINE.md, que se ven en
GitHub pero no se pegan en un documento. Estado del 21/10/2026.

El lienzo se mide en PULGADAS y la altura de cada caja se CALCULA a partir de su numero
de lineas, en vez de colocar las coordenadas a mano: asi ninguna caja se solapa con la
siguiente aunque cambie el texto.

  python src/90_figuras.py

out: entrega/figuras/fig{1,2,3,4}_*.{svg,png}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAL = f"{BASE}/entrega/figuras"
os.makedirs(SAL, exist_ok=True)

matplotlib.rcParams["svg.fonttype"] = "none"      # texto editable en el SVG
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# Paleta con luminancia distinta, para que se distingan tambien impresas en gris.
C = {
    "entrada": ("#e8eef7", "#2d4a7c"),
    "modelo": ("#d7e6d4", "#3c6b46"),
    "proceso": ("#f3f3f1", "#5a5a5a"),
    "decision": ("#faf0dc", "#8a6d1f"),
    "reparo": ("#f7e9e9", "#8c4646"),
}
TAM = 8.0                 # cuerpo del texto de las cajas
LINEA = 0.155             # alto de una linea de texto, en pulgadas
PAD_V = 0.20              # margen vertical dentro de la caja
GAP = 0.26                # separacion entre cajas


def alto_de(texto, tam=TAM):
    """alto en pulgadas que necesita una caja para su texto, sin apretarlo"""
    n = texto.count("\n") + 1
    return PAD_V + n * LINEA * (tam / TAM)


def caja(ax, x, y, w, texto, estilo="proceso", tam=TAM, negrita=False, h=None):
    """(x, y) es la esquina SUPERIOR izquierda; devuelve el rectangulo colocado"""
    h = h or alto_de(texto, tam)
    relleno, borde = C[estilo]
    ax.add_patch(FancyBboxPatch(
        (x, y - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.1, edgecolor=borde, facecolor=relleno, zorder=2))
    ax.text(x + w / 2, y - h / 2, texto, ha="center", va="center", fontsize=tam,
            color="#1a1a1a", zorder=3, linespacing=1.4,
            fontweight="bold" if negrita else "normal")
    return dict(x=x, y=y, w=w, h=h, cx=x + w / 2, cy=y - h / 2,
                sup=y, inf=y - h, izq=x, der=x + w)


def rombo(ax, cx, cy, w, h, texto, tam=7.4):
    relleno, borde = C["decision"]
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2),
                          (cx - w / 2, cy)], closed=True, linewidth=1.1,
                         edgecolor=borde, facecolor=relleno, zorder=2))
    ax.text(cx, cy, texto, ha="center", va="center", fontsize=tam, color="#1a1a1a",
            zorder=3, linespacing=1.35)
    return dict(cx=cx, cy=cy, w=w, h=h, sup=cy + h / 2, inf=cy - h / 2,
                izq=cx - w / 2, der=cx + w / 2)


def flecha(ax, p1, p2, guiones=False, grosor=1.0):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=10, linewidth=grosor,
        color="#4a4a4a", zorder=1, shrinkA=0, shrinkB=0,
        linestyle=(0, (4, 3)) if guiones else "-"))


def baja(ax, a, b):
    """flecha vertical entre dos cajas apiladas"""
    flecha(ax, (a["cx"], a["inf"]), (b["cx"], b["sup"]))


def etiqueta(ax, x, y, t, tam=6.9, ha="left", rot=0, color="#333333"):
    ax.text(x, y, t, fontsize=tam, ha=ha, va="center", color=color, rotation=rot,
            zorder=4)


def lienzo(ancho, alto):
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, alto)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    return fig, ax


def guardar(fig, nombre):
    for ext, kw in (("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(f"{SAL}/{nombre}.{ext}", bbox_inches="tight", pad_inches=0.12,
                    facecolor="white", **kw)
    plt.close(fig)
    print(f"  {nombre}.svg  +  {nombre}.png")


def leyenda(ax, x, y, items):
    """leyenda horizontal, fuera de la columna"""
    for k, t in items:
        relleno, borde = C[k]
        ax.add_patch(FancyBboxPatch((x, y - 0.09), 0.26, 0.17,
                                    boxstyle="round,pad=0.01,rounding_size=0.04",
                                    linewidth=1.0, edgecolor=borde,
                                    facecolor=relleno, zorder=2))
        ax.text(x + 0.33, y, t, fontsize=7.0, va="center", color="#333333")
        x += 0.33 + len(t) * 0.052 + 0.20


# ---------------------------------------------------------------- figura 1
# Estado del 21/10/2026. Los vecinos recuperados YA NO van en ningun prompt
# (EJEMPLOS_ARRIBA = EJEMPLOS_ABAJO = False): solo salen en el campo vecinos_proyecto
# de la respuesta, asi que no son un paso del flujo y aqui no se dibujan.
PASOS = [
    ("Descripción del producto  (español o inglés)", "entrada", True),
    ("Leer los ACTORES del español\nléxico de 65 actores → rol inglés", "proceso", False),
    ("Traducir a inglés    ·    Marian opus-mt-es-en", "modelo", True),
    ("Planificar ROLES  (figura 3)  y  CLASES de función\n"
     "2 clases por rol, P(clase | familia del rol), piso 0,10", "proceso", False),
    ("MITAD DE ARRIBA:  descripción + fichas → una historia\n"
     "muestreo top-p 0,95 · 6 muestras o más por par (rol, clase)", "modelo", True),
    ("Banco de candidatas\nfiltros de forma, de rol válido y de léxico de Jira", "proceso", False),
    ("SELECCIÓN por cobertura de las funciones de la descripción\n"
     "voraz submodular + 6 castigos de forma", "proceso", False),
    ("FUSIONAR:  las anclas entran siempre, hasta k/2\n"
     "luego la selección sin duplicarlas · actores fuera del plan al final", "proceso", False),
    ("MITAD DE ABAJO:  historia → conjunto de tareas\n"
     "DOS modelos en conjunto · haz de 4 · sin repetir 4-gramas · pen. 1,3", "modelo", True),
    ("Filtrar y deduplicar tareas\nfuera lo que no es desarrollo y las frases de historia\n"
     "0,90 entre historias · 0,95 dentro · 0,95 sin el verbo", "proceso", False),
    ("Tope de arranques repetidos: máximo 2 por backlog", "proceso", False),
    ("Reasignar cada tarea a su historia · piso de pertenencia 0,30", "proceso", False),
    ("Traducir al español    ·    Marian opus-mt-en-es", "modelo", True),
    ("Reparaciones en español\nglosario · glosario con fuente inglesa · glosario general\n"
     "restos en inglés · unificar roles · verbo en infinitivo · eco por flexión",
     "reparo", False),
    ("Backlog: 12 historias con sus tareas", "entrada", True),
]
I_PLAN, I_FUSION = 3, 7          # de donde sale y donde entra el ramal de las anclas


def figura1():
    W, X = 4.45, 0.30
    alto_total = sum(alto_de(t) for t, _, _ in PASOS) + GAP * (len(PASOS) - 1) + 1.25
    fig, ax = lienzo(7.6, alto_total)
    ax.text(3.8, alto_total - 0.22, "Figura 1. Generación de un backlog, de extremo a extremo",
            fontsize=10.5, fontweight="bold", ha="center", color="#1a1a1a")
    y = alto_total - 0.62
    cajas = []
    for t, est, neg in PASOS:
        cajas.append(caja(ax, X, y, W, t, est, negrita=neg))
        y = cajas[-1]["inf"] - GAP
    for a, b in zip(cajas, cajas[1:]):
        baja(ax, a, b)
    # los dos pasos que solo corren si la entrada viene en espanol
    x0 = X + W + 0.16
    ax.plot([x0, x0 + 0.10, x0 + 0.10, x0],
            [cajas[1]["sup"], cajas[1]["sup"], cajas[2]["inf"], cajas[2]["inf"]],
            color="#8a6d1f", linewidth=1.1)
    etiqueta(ax, x0 + 0.17, (cajas[1]["sup"] + cajas[2]["inf"]) / 2,
             "solo si la entrada\nviene en español", color="#8a6d1f")
    # ramal de las ANCLAS: sale del plan de roles, corre en paralelo a la mitad de
    # arriba y entra en la fusion. No depende del banco ni de la seleccion.
    plan, fus = cajas[I_PLAN], cajas[I_FUSION]
    xa, wa = X + W + 0.30, 7.6 - (X + W + 0.30) - 0.12
    texto_a = ("ANCLAS: una historia por\nfunción de la descripción\n\n"
               "prefijo forzado en el\ndecodificador de arriba:\n"
               "«As a <rol>, I want to\n<función>, so that»\n\n"
               "el modelo escribe el beneficio\n6 muestras · elige el\n"
               "reordenador de beneficios")
    ha_ = alto_de(texto_a, 7.3)
    ya = (plan["inf"] + fus["sup"]) / 2 + ha_ / 2
    an = caja(ax, xa, ya, wa, texto_a, "modelo", tam=7.3)
    ax.plot([plan["der"], an["cx"]], [plan["cy"], plan["cy"]], color="#4a4a4a", linewidth=1.0)
    flecha(ax, (an["cx"], plan["cy"]), (an["cx"], an["sup"]))
    ax.plot([an["cx"], an["cx"]], [an["inf"], fus["cy"]], color="#4a4a4a", linewidth=1.0)
    flecha(ax, (an["cx"], fus["cy"]), (fus["der"], fus["cy"]))
    leyenda(ax, 0.12, 0.22,
            [("entrada", "entrada / salida"), ("modelo", "red neuronal"),
             ("proceso", "cálculo determinista"), ("reparo", "corrección posterior")])
    guardar(fig, "fig1_pipeline")


# ---------------------------------------------------------------- figura 2
def figura2():
    fig, ax = lienzo(7.6, 4.95)
    ax.text(3.8, 4.73, "Figura 2. Las dos mitades del generador", fontsize=10.5,
            fontweight="bold", ha="center", color="#1a1a1a")

    def fila(y, titulo, prompt, modelo, salida, nota, nota_arriba=True):
        etiqueta(ax, 0.35, y + 0.20, titulo, tam=8.6, color="#2d4a7c")
        a = caja(ax, 0.35, y, 2.15, prompt, "entrada", tam=7.3)
        b = caja(ax, 2.90, y, 1.85, modelo, "modelo", tam=8.0, negrita=True)
        c = caja(ax, 5.15, y, 2.10, salida, "entrada", tam=7.3)
        h = max(a["h"], b["h"], c["h"])
        flecha(ax, (a["der"], y - h / 2), (b["izq"], y - h / 2))
        flecha(ax, (b["der"], y - h / 2), (c["izq"], y - h / 2))
        if nota_arriba:
            etiqueta(ax, 7.25, y + 0.20, nota, tam=7.0, ha="right")
        else:
            etiqueta(ax, 3.80, y - h - 0.24, nota, tam=7.0, ha="center")
        return a, b, c, h

    a1, b1, c1, h1 = fila(
        4.17, "descripción  →  UNA historia",
        "feature -> user story\n| role: X | funcion: Y |\n<descripción> | examples:",
        "MITAD DE ARRIBA\nmT5-small podado\n121 M · One2One",
        "Una historia de usuario\nAs a X, I want ...,\nso that ...",
        "muestreo: top-p 0,95 · T 1,0 · 6 muestras o más por par (rol, clase)")
    a2, b2, c2, h2 = fila(
        2.62, "historia  →  SUS tareas",
        "decompose story into\ntasks: <historia>\n| examples:",
        "MITAD DE ABAJO\n2 mT5-small podados\nen conjunto · One2Seq",
        "Conjunto de tareas\nt1 ; t2 ; t3 ...\n(se juntan los dos)",
        "haz de 4 · sin repetir 4-gramas · penalización de repetición 1,3",
        nota_arriba=False)

    xbaja = a2["der"] - 0.30
    ax.plot([c1["cx"], c1["cx"], xbaja], [c1["inf"], 2.97, 2.97],
            color="#4a4a4a", linewidth=1.2)
    flecha(ax, (xbaja, 2.97), (xbaja, a2["sup"]), grosor=1.2)
    etiqueta(ax, (c1["cx"] + xbaja) / 2, 3.10, "las 12 historias elegidas",
             tam=7.4, ha="center")

    ax.text(3.8, 1.20,
            "Tres modelos: uno arriba y dos abajo, entrenados por separado y con datos distintos.\n"
            "«examples:» va vacío en las dos mitades: los vecinos recuperados ya no entran en el prompt.\n"
            "En las anclas, el decodificador de arriba arranca forzado con «As a X, I want to <función>, so that».",
            fontsize=7.4, ha="center", color="#444444", style="italic",
            linespacing=1.55)
    leyenda(ax, 0.35, 0.38,
            [("entrada", "texto de entrada o de salida"), ("modelo", "red neuronal")])
    guardar(fig, "fig2_dos_mitades")


# ---------------------------------------------------------------- figura 3
# En el codigo esto es un arbol de condiciones, pero lo que hace es probar FUENTES DE
# ACTORES en orden de fiabilidad y parar en la primera que sirve. Dibujarlo como una
# cascada es mas fiel y se lee de una vez.
REGLAS = [
    ("1", "Actores que la descripción nombra EN ESPAÑOL",
     "Léxico de 65 actores del español al rol inglés que el generador espera.\n"
     "Se leen ANTES de traducir: el traductor rompe «apoderados» → «agents».",
     "si da 2 o más, se le cree y no se consulta la traducción"),
    ("2", "Actores que el extractor lee del inglés traducido",
     "Léxico de 935 roles y 262 núcleos de persona. Se descartan los que son\n"
     "MÁQUINA y no persona: «runs build pipelines on agents».",
     "con plantel basta 1 actor; sin plantel hacen falta 2"),
    ("3", "Plantel del sector más parecido de la taxonomía",
     "Solo si la descripción se parece a algún sector: similitud ≥ 0,65.\n"
     "Bajo 0,75 solo entran los roles que la descripción RESPALDA.",
     "rellena hasta 4 actores, detrás de los que la descripción nombra"),
    ("4", "Cliente implícito, colectivos y, si aún faltan, genéricos",
     "«reservar … pagos en la app» → customer. «notifies the team» → team\n"
     "member. Y de último recurso: administrator, user, support agent.",
     "el plan NUNCA queda vacío"),
]


def figura3():
    fig, ax = lienzo(7.6, 8.3)
    ax.text(3.8, 8.08, "Figura 3. De dónde salen los actores del backlog",
            fontsize=10.5, fontweight="bold", ha="center", color="#1a1a1a")
    ax.text(3.8, 7.80,
            "Cuatro fuentes en orden de fiabilidad; se acumulan hasta reunir los actores del plan.",
            fontsize=7.8, ha="center", color="#444444", style="italic")
    e = caja(ax, 2.10, 7.55, 3.40, "Descripción del producto", "entrada",
             negrita=True, tam=8.4)
    y = e["inf"] - GAP
    prev = e
    for num, titulo, cuerpo, cond in REGLAS:
        h = PAD_V + 4.7 * LINEA + 0.10      # titulo, 2 de cuerpo y la condicion
        b = caja(ax, 0.90, y, 5.80, "", "proceso", h=h)
        ax.text(1.12, y - 0.24, num, fontsize=13, fontweight="bold",
                color="#8a6d1f", ha="center", va="center", zorder=4)
        ax.text(1.38, y - 0.24, titulo, fontsize=8.6, fontweight="bold",
                color="#1a1a1a", ha="left", va="center", zorder=4)
        ax.text(1.38, y - 0.62, cuerpo, fontsize=7.3, color="#333333",
                ha="left", va="center", zorder=4, linespacing=1.45)
        ax.text(6.50, y - h + 0.20, cond, fontsize=7.1, color="#8a6d1f",
                ha="right", va="center", zorder=4, style="italic")
        baja(ax, prev, b)
        prev = b
        y = b["inf"] - GAP
    f = caja(ax, 1.55, y, 4.50,
             "plan_funciones:  2 clases por rol según P(clase | familia del rol), piso 0,10\n"
             "Clases: receive · view · set · track · monitor · manage",
             "proceso", tam=7.6)
    baja(ax, prev, f)
    s = caja(ax, 1.90, f["inf"] - GAP, 3.80,
             "Pares (rol, clase): una tanda de muestreo por par", "entrada",
             negrita=True, tam=8.4)
    baja(ax, f, s)
    ax.text(3.8, s["inf"] - 0.32,
            "Los sectores acertados están en 0,75 o más; los equivocados, en 0,69-0,71: limpieza a\n"
            "domicilio caía en atención domiciliaria por 0,009 y salía entera de enfermeras.",
            fontsize=7.4, ha="center", color="#444444", style="italic", linespacing=1.5)
    guardar(fig, "fig3_plan_roles")


# ---------------------------------------------------------------- figura 4
# Procedencia: de donde sale cada modelo y con que datos se entreno. Es lo que hay que
# poder repetir para que el resultado sea reproducible. Hiperparametros leidos de los
# registros de entrenamiento (logs/09_v9bt.log, logs/44_train_abajo_v5.log y
# logs/09_train_hist_hp.log), no de memoria.
def figura4():
    fig, ax = lienzo(7.6, 6.35)
    ax.text(3.8, 6.13, "Figura 4. Procedencia de los datos y de los modelos",
            fontsize=10.5, fontweight="bold", ha="center", color="#1a1a1a")

    etiqueta(ax, 0.30, 5.77, "DATOS", tam=8.4, color="#2d4a7c")
    d1 = caja(ax, 0.30, 5.60, 2.05, "Corpus real de Jira / TAWOS\n23.247 textos de producto",
              "entrada", tam=7.3)
    d2 = caja(ax, 2.70, 5.60, 2.05,
              "Taxonomía sembrada\n40 sectores · 622 descripciones", "entrada", tam=7.3)
    d3 = caja(ax, 5.10, 5.60, 2.20,
              "Profesor LLM local\nqwen2.5:7b · Apache-2.0", "modelo", tam=7.3)
    flecha(ax, (d2["der"], d2["cy"]), (d3["izq"], d3["cy"]))

    c1 = caja(ax, 0.30, 4.67, 2.05, "s2s_o2o  +  s2s_o2ofmt\ndescripción → una historia",
              "proceso", tam=7.3)
    c2 = caja(ax, 2.70, 4.67, 2.05, "s2s_hist\nhistoria → tareas", "proceso", tam=7.3)
    c3 = caja(ax, 5.10, 4.67, 2.20,
              "Backlogs del profesor\nvalidados y curados", "proceso", tam=7.3)
    flecha(ax, (d1["cx"], d1["inf"]), (c1["cx"], c1["sup"]))
    flecha(ax, (d1["der"] - 0.30, d1["inf"]), (c2["izq"] + 0.30, c2["sup"]))
    flecha(ax, (d3["cx"], d3["inf"]), (c3["cx"], c3["sup"]))

    m1 = caja(ax, 0.30, 3.73, 4.45,
              "s2s_v9bt  =  o2ofmt + profesor + RETROTRADUCCIÓN de la descripción\n"
              "25.938 ejemplos   ·   src/88_retrotraducir_datos.py", "proceso", tam=7.3)
    m2 = caja(ax, 5.10, 3.73, 2.20, "s2s_v5\nhist + profesor curado", "proceso", tam=7.3)
    flecha(ax, (c1["cx"], c1["inf"]), (1.30, m1["sup"]))
    flecha(ax, (c3["cx"], c3["inf"]), (3.90, m1["sup"]))
    flecha(ax, (c3["cx"] + 0.55, c3["inf"]), (m2["cx"] + 0.55, m2["sup"]))
    flecha(ax, (c2["der"] - 0.20, c2["inf"]), (m2["izq"] + 0.30, m2["sup"]))

    etiqueta(ax, 0.30, 3.00, "MODELOS", tam=8.4, color="#2d4a7c")
    b0 = caja(ax, 2.40, 2.85, 2.80,
              "google/mt5-small  →  poda de vocabulario\n"
              "250.100 → 75.360 piezas  ·  121 M parámetros", "entrada", tam=7.3)
    a1 = caja(ax, 0.20, 1.98, 2.35,
              "ARRIBA  _o2o_o2ofmt_v9bt\nbs 32 · lr 5e-4 · mejor ép. 3/4\n"
              "sigue a _o2o_o2ofmt", "modelo", tam=6.9)
    a2 = caja(ax, 2.68, 1.98, 2.35,
              "ABAJO 1  _hist_v5\nbs 16 · lr 5e-4 · mejor ép. 1/4\n"
              "sigue a _hist_pbe_v1", "modelo", tam=6.9)
    a3 = caja(ax, 5.16, 1.98, 2.24,
              "ABAJO 2  _hist_bs8_lr9.2e-4\nbs 8 · lr 9,2e-4 · mejor ép. 3\n"
              "búsqueda de hiperparámetros", "modelo", tam=6.9)
    for a_ in (a1, a2, a3):
        flecha(ax, (b0["cx"] + (a_["cx"] - b0["cx"]) * 0.35, b0["inf"]),
               (a_["cx"], a_["sup"]))
    flecha(ax, (m1["cx"] - 1.20, m1["inf"]), (a1["cx"] - 0.40, a1["sup"]), guiones=True)
    # v5 -> abajo 1: baja por fuera de la caja del mT5 base y entra por arriba
    ax.plot([5.32, 5.32], [m2["inf"], 2.18], color="#4a4a4a", linewidth=1.0,
            linestyle=(0, (4, 3)))
    flecha(ax, (5.32, 2.18), (a2["der"] - 0.20, a2["sup"]), guiones=True)
    # el segundo de abajo se entrena SOLO con datos reales, sin el profesor: por eso
    # yerra distinto que el primero, que es lo que hace util el conjunto. La linea baja
    # por el pasillo entre s2s_v9bt y s2s_v5 para no cruzar ninguna caja.
    xc, yc = (m1["der"] + m2["izq"]) / 2, m1["inf"] - 0.12
    ax.plot([c2["der"], xc, xc, 7.48, 7.48],
            [c2["cy"], c2["cy"], yc, yc, a3["sup"] + 0.15],
            color="#4a4a4a", linewidth=0.9, linestyle=(0, (4, 3)))
    flecha(ax, (7.48, a3["sup"] + 0.15), (a3["der"] - 0.10, a3["sup"]), guiones=True)
    etiqueta(ax, 7.40, yc - 0.13, "solo datos reales, sin profesor", tam=6.6, ha="right")

    caja(ax, 0.30, 1.20, 3.30,
         "SBERT propio   sbert_pbe_v1\ndesde paraphrase-multilingual-mpnet-base-v2",
         "modelo", tam=7.1)
    etiqueta(ax, 3.80, 0.97,
             "recupera, selecciona el backlog y deduplica,\nen todo el pipeline",
             tam=7.2)
    ax.text(3.8, 0.30,
            "Se guarda la mejor época por pérdida de validación, no la última. "
            "Flechas discontinuas: datos de entrenamiento.",
            fontsize=7.2, ha="center", color="#444444", style="italic")
    guardar(fig, "fig4_procedencia")



# ------------------------------------------------- figura 1 para el artículo
# Version compacta de la figura 1 para UNA COLUMNA de IEEE (8,9 cm). No es la figura 1
# reducida: a 8,9 cm su cuerpo de 8 pt quedaria en 3,7 pt. Aqui el lienzo mide 8,9 cm
# de verdad (3,26 pulgadas mas el margen que anade guardar()), el texto va a 6,4 pt
# -legible impreso- y los quince pasos se agrupan en diez, sin nombres de funciones
# del codigo. Las cajas llevan la altura calculada aparte, mas justa que alto_de(),
# porque aqui el alto total es el recurso escaso.
# Las dos ramas van lado a lado porque es lo que son: el banco de candidatas y las
# anclas salen los dos del plan de roles y se juntan en la fusion.
ANCHO_IEEE = 3.26          # + 0,24 de margen = 3,50 pulgadas = 8,89 cm
TAM_ES = 6.4
GAP_ES = 0.10


def _alto_es(texto):
    return 0.10 + (texto.count("\n") + 1) * 0.125


def figura1_es():
    W, X = 3.00, 0.13
    wc = (W - 0.10) / 2                       # las dos columnas de la parte paralela
    fig, ax = lienzo(ANCHO_IEEE, 5.62)
    ax.text(ANCHO_IEEE / 2, 5.47, "Generación de un backlog, de extremo a extremo",
            fontsize=7.6, fontweight="bold", ha="center", color="#1a1a1a")
    y = 5.28

    def fila(texto, estilo, negrita=False):
        nonlocal y
        c = caja(ax, X, y, W, texto, estilo, tam=TAM_ES, negrita=negrita,
                 h=_alto_es(texto))
        y = c["inf"] - GAP_ES
        return c

    e = fila("Descripción del producto  (español o inglés)", "entrada", True)
    tr = fila("Leer los actores en español  ·  traducir con Marian (es→en)\n"
              "(solo si la descripción viene en español)", "modelo")
    pl = fila("Plan de roles  y  fichas de control  (rol y función)", "proceso")
    baja(ax, e, tr)
    baja(ax, tr, pl)

    # --- las dos ramas, lado a lado ---
    t_iz = ("Mitad de arriba\nuna historia por ficha\nmuestreo top-p",
            "Banco de candidatas\nfiltros de forma y de rol",
            "Selección por cobertura\nde rasgos")
    izq, yi = [], y
    for s in t_iz:
        izq.append(caja(ax, X, yi, wc, s, "proceso" if izq else "modelo",
                        tam=TAM_ES, h=_alto_es(s)))
        yi = izq[-1]["inf"] - GAP_ES
    for a_, b_ in zip(izq, izq[1:]):
        baja(ax, a_, b_)
    t_an = ("Historias ancla\nuna por función que la\ndescripción nombra\n"
            "el decodificador arranca\nforzado con la función;\n"
            "el modelo escribe el\nbeneficio")
    h_iz = y - izq[-1]["inf"]
    an = caja(ax, X + wc + 0.10, y - (h_iz - _alto_es(t_an)) / 2, wc, t_an, "modelo",
              tam=TAM_ES, h=_alto_es(t_an))
    ymed = pl["inf"] - GAP_ES / 2
    ax.plot([pl["cx"], pl["cx"]], [pl["inf"], ymed], color="#4a4a4a", linewidth=0.9)
    for hijo in (izq[0], an):
        ax.plot([pl["cx"], hijo["cx"]], [ymed, ymed], color="#4a4a4a", linewidth=0.9)
        flecha(ax, (hijo["cx"], ymed), (hijo["cx"], hijo["sup"]))

    y = min(izq[-1]["inf"], an["inf"]) - GAP_ES
    fu = fila("Fusión:  las anclas primero, hasta k/2,\n"
              "y el resto de la selección sin repetirlas", "proceso")
    for padre in (izq[-1], an):
        ax.plot([padre["cx"], padre["cx"], fu["cx"]],
                [padre["inf"], fu["sup"] + GAP_ES / 2, fu["sup"] + GAP_ES / 2],
                color="#4a4a4a", linewidth=0.9)
    flecha(ax, (fu["cx"], fu["sup"] + GAP_ES / 2), (fu["cx"], fu["sup"]))

    resto = [fila("Mitad de abajo:  historia → sus tareas\n"
                  "dos modelos  ·  búsqueda en haz", "modelo"),
             fila("Filtrar y deduplicar las tareas", "proceso"),
             fila("Reasignar cada tarea a su historia", "proceso"),
             fila("Traducir con Marian (en→es)", "modelo"),
             fila("Reparaciones de léxico en español", "reparo"),
             fila("Backlog:  12 historias con sus tareas", "entrada", True)]
    for a_, b_ in zip([fu] + resto, resto):
        baja(ax, a_, b_)
    guardar(fig, "fig1_pipeline_es")


if __name__ == "__main__":
    print("escribiendo en entrega/figuras/")
    figura1()
    figura2()
    figura3()
    figura4()
    figura1_es()
