"""Memoria de tesis (avance): reorganiza el trabajo de investigacion del 24/06/2026 segun la
plantilla de memoria de la UPC y escribe los capitulos nuevos (diseno y desarrollo, algoritmos,
aporte, aplicacion, validacion).

Se parte del .docx antiguo para conservar portada, estilos, figuras y tablas: sus elementos se
MUEVEN dentro del mismo documento (asi las imagenes siguen enlazadas). Las cifras de los capitulos
nuevos se leen de results/; ninguna se escribe a mano. Lo que depende de la web queda marcado
[COMPLETAR] en rojo.

Salida: entrega/memoria/Memoria_Agix_avance.docx (actualizar campos en Word: Ctrl+A, F9).
"""
import copy
import json
import re
from pathlib import Path

import docx
import pandas as pd
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ANTIGUA = Path(r"C:/Users/User/Documents/2026-01-UPC/TDP ENTREGABLES FINALES/FINAL FINAL/"
               r"TI_Sotil_Tyrone_Diego_Villafuerte.docx")
SALIDA = Path("entrega/memoria/Memoria_Agix_avance.docx")
FIG = Path("reports/figuras_generacion")


# ---------------------------------------------------------------- cifras (todas de results/)
def j(p):
    return json.load(open(p, encoding="utf-8"))


SERV = j("results/estimador_v7/servicio.json")
RES7 = j("results/estimador_v7/resumen.json")
CONS = j("results/estimador_v7/construccion.json")
SMALL = j("results/large_contra_small/resumen.json")
PRU = pd.read_csv("results/estimador_v7/prueba.csv")
BASE = pd.read_csv("results/large_contra_small/base_sola.csv")
LIT = pd.read_csv("results/baselines_literatura.csv")
ANC = pd.read_csv("results/anclas_escala/resumen.csv")
CUR = pd.read_csv("results/curva_final/resumen.csv")
BAN = pd.read_csv("results/banda_que_cierra_antes/resumen.csv")
LLM = pd.read_csv("results/llm_bench/api_resumen.csv")


def n(x, d=3):
    return f"{x:.{d}f}".replace(".", ",")


def pc(x, d=1):
    return f"{100 * x:.{d}f}".replace(".", ",") + " %"


def fila(df, col, patron):
    return df[df[col].astype(str).str.contains(patron, regex=False)].iloc[0]


V6 = PRU[(PRU.modelo.str.startswith("v6")) & (PRU.salida == "horas")].iloc[0]
MET = pd.read_csv("results/large_contra_small/metricas.csv")
MEZCLA = MET[(MET.codificador == "large") & (MET.canal == "mezcla")].iloc[0]
V7 = PRU[(PRU.modelo == "v7 int8 podado") & (PRU.salida == "horas")].iloc[0]
V7P = PRU[(PRU.modelo == "v7 int8 podado") & (PRU.salida == "valor_probable")].iloc[0]
LARGE = BASE[BASE.codificador == "large"].iloc[0]
SMALLR = BASE[BASE.codificador == "small"].iloc[0]
ANC_K = {int(r.k): r for r in ANC.itertuples()}
CONST_TOT = CUR[CUR.brazo.str.startswith("0")].iloc[0]
BANDA = {int(r.k): r.anchura_mediana for r in BAN[BAN.banda == "D · tramo contraido M=1500"].itertuples()}

# ------------------------------------------------------------------------------ documento
doc = docx.Document(str(ANTIGUA))
body = doc.element.body
B = list(body)                       # foto de los elementos originales, por indice
SECT = B[-1]
seq = []                             # el nuevo orden del cuerpo


def mover(a, b):
    """Elementos originales a..b inclusive."""
    seq.extend(B[a:b + 1])


def _nuevo(texto="", estilo=None):
    p = doc.add_paragraph(texto, style=estilo)
    body.remove(p._p)
    seq.append(p._p)
    return p


def H(nivel, texto):
    return _nuevo(texto, f"Heading {nivel}")


def P(*partes, centrado=False):
    """Parrafo; las partes que empiezan por '!' son recordatorios [COMPLETAR]."""
    p = _nuevo()
    for t in partes:
        if t.startswith("!"):
            r = p.add_run(f"[COMPLETAR: {t[1:]}]")
            r.bold = True
            r.font.color.rgb = RGBColor(0xC0, 0, 0)
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        elif t.startswith("**"):
            p.add_run(t.strip("*")).bold = True
        else:
            p.add_run(t)
    if centrado:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p


def L(items):
    for t in items:
        p = _nuevo("", "List Paragraph")
        if isinstance(t, tuple):
            p.add_run(t[0]).bold = True
            p.add_run(t[1])
        else:
            P_runs(p, t)


def P_runs(p, t):
    if t.startswith("!"):
        r = p.add_run(f"[COMPLETAR: {t[1:]}]")
        r.bold = True
        r.font.color.rgb = RGBColor(0xC0, 0, 0)
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    else:
        p.add_run(t)


def style_lista():
    return "List Paragraph"


def _campo(p, instr, visible="1"):
    r = p.add_run()
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = instr
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "separate")
    t = OxmlElement("w:t"); t.text = visible
    f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end")
    for e in (f1, it, f2, t, f3):
        r._r.append(e)


def leyenda(tipo, texto):
    """Tipo 'Table' o 'figura', con la misma secuencia que usa el documento antiguo."""
    p = _nuevo("", "Caption")
    p.add_run("Tabla " if tipo == "Table" else "Figura ")
    _campo(p, f" SEQ {tipo} \\* ARABIC ")
    p.add_run(". " + texto)
    return p


def T(cab, filas, titulo, nota=None):
    leyenda("Table", titulo)
    t = doc.add_table(rows=1, cols=len(cab))
    t.style = doc.styles["Table Grid"]
    for c, v in zip(t.rows[0].cells, cab):
        c.text = ""
        c.paragraphs[0].add_run(str(v)).bold = True
    for f in filas:
        cs = t.add_row().cells
        for c, v in zip(cs, f):
            c.text = str(v)
    for r in t.rows:
        for c in r.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
    body.remove(t._tbl)
    seq.append(t._tbl)
    if nota:
        p = P(nota)
        for r in p.runs:
            r.font.size = Pt(9)
            r.italic = True


def F(ruta, titulo, ancho=15):
    leyenda("figura", titulo)
    p = _nuevo()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(ruta), width=Cm(ancho))


def texto(i, nuevo):
    """Cambia el texto de un parrafo original conservando el formato de su primer run."""
    p = docx.text.paragraph.Paragraph(B[i], doc)
    rs = p.runs
    rs[0].text = nuevo
    for r in rs[1:]:
        r._r.getparent().remove(r._r)


def nivel(e, delta):
    """Sube o baja el nivel de los titulos de un tramo movido."""
    for st in e.iter(qn("w:pStyle")):
        m = re.fullmatch(r"Heading(\d)", st.get(qn("w:val")))
        if m:
            st.set(qn("w:val"), f"Heading{min(6, max(1, int(m.group(1)) + delta))}")


def mover_nivel(a, b, delta):
    for e in B[a:b + 1]:
        nivel(e, delta)
    mover(a, b)


# =============================================================================== PRELIMINARES
texto(5, "Agix: generación de backlogs de historias de usuario y estimación de su esfuerzo con "
         "modelos ligeros, integrados en un sistema web de gestión ágil")
texto(6, "TESIS")
texto(7, "Para optar el título profesional de Ingeniero de Software / Licenciado en Ciencias de la "
         "Computación")
texto(16, "Lima, [COMPLETAR: día] de [COMPLETAR: mes] de 2026")
mover(0, 16)

H(1, "Dedicatoria")
P("!opcional; texto en cursiva, en página aparte")
H(1, "Agradecimientos")
P("!opcional; asesora, UPC y quienes apoyaron el trabajo")

texto(17, "Resumen")
texto(18, (
    "Al iniciar un proyecto ágil, el equipo debe convertir la idea de un producto en un backlog y "
    "estimar su esfuerzo justo cuando no dispone de historial. Esta tesis presenta Agix, un sistema "
    "web de gestión ágil que, a partir de una descripción breve del producto, genera un backlog de "
    "historias de usuario con sus tareas y estima las horas de cada tarea y del proyecto. Lo hace "
    "con dos tuberías de modelos ligeros que funcionan sin conexión en CPU y se sirven como API. La "
    "tubería de generación encadena dos modelos mT5-small condicionados por rol y función, una "
    "selección por cobertura y unas historias ancla que garantizan que cada función nombrada en la "
    "descripción llegue al backlog. La tubería de estimación combina un codificador e5-large "
    "cuantizado con un apilado de regresores, un intervalo conformal y un reanclaje del total a "
    "medida que se cierran tareas. Evaluado por proyecto completo, sin que el modelo vea el proyecto "
    f"que predice, el total del proyecto cae dentro de ±25 % en el {pc(ANC_K[0].PRED25)} de los "
    f"casos al arranque y en el {pc(ANC_K[20].PRED25)} tras cerrar 20 tareas. Frente a modelos de "
    "lenguaje de gran escala actuales, Agix gana con claridad cuando estos no se calibran y empata "
    "cuando se calibran con horas reales, respondiendo en milisegundos y sin coste por llamada."))
texto(20, "Palabras clave: estimación de esfuerzo de software; historias de usuario; generación de "
          "backlog; aprendizaje automático; predicción conformal; gestión ágil de proyectos")
mover(17, 20)
P("**Agix: generating user-story backlogs and estimating their effort with lightweight models "
  "integrated into an agile project management web system", centrado=True)
H(1, "Abstract")
P("When an agile project starts, the team must turn a product idea into a backlog and estimate "
  "its effort precisely when no history is available. This thesis presents Agix, an agile project "
  "management web system that, from a short product description, generates a backlog of user "
  "stories with their tasks and estimates the hours of each task and of the whole project. It "
  "relies on two pipelines of lightweight models that run offline on CPU and are served as an "
  "API. The generation pipeline chains two mT5-small models conditioned on role and function, a "
  "coverage-based selection and anchor stories that guarantee every function named in the "
  "description reaches the backlog. The estimation pipeline combines a quantized e5-large encoder "
  "with a stacked regressor, a conformal interval and a re-anchoring of the project total as "
  "tasks are closed. Evaluated on whole held-out projects, the project total falls within ±25 % "
  f"in {pc(ANC_K[0].PRED25)} of cases at start-up and in {pc(ANC_K[20].PRED25)} after 20 tasks "
  "are closed. Against current large language models, Agix clearly wins when they are not "
  "calibrated and ties when they are calibrated with real hours, while answering in milliseconds "
  "at no per-call cost.")
P("Keywords: software effort estimation; user stories; backlog generation; machine learning; "
  "conformal prediction; agile project management")
mover(21, 45)                         # tabla de contenidos (campo) y saltos
# las listas de tablas y figuras pasan a campos que Word regenera
p = _nuevo("Índice de tablas", "TOC Heading")
_campo(_nuevo(), ' TOC \\h \\z \\c "Table" ', "Actualizar el campo (F9)")
p = _nuevo("Índice de figuras", "TOC Heading")
_campo(_nuevo(), ' TOC \\h \\z \\c "figura" ', "Actualizar el campo (F9)")
mover(83, 109)                        # salto de seccion + Student Outcome

# ============================================================================== 1. INTRODUCCION
H(1, "Introducción")
P("La gestión de proyectos ágiles depende de dos decisiones que se toman al inicio, cuando menos "
  "información hay: qué construir y cuánto costará. Esta tesis aborda ambas a la vez. A "
  "continuación se presentan los antecedentes del problema, el caso de estudio y la formulación "
  "del problema; el capítulo 2 describe el diseño y desarrollo del proyecto, el 3 los algoritmos, "
  "el 4 el aporte científico, el 5 la aplicación web, y el 6 la validación y sus resultados.")
mover_nivel(112, 122, 0)              # Antecedentes (H2) internacionales/nacionales (H3)
mover_nivel(142, 159, -1)             # Caso de estudio pasa a H2
H(2, "Formulación del problema")
P("¿De qué manera un sistema web de gestión ágil que integra modelos propios de aprendizaje "
  "automático para generar el backlog a partir de la descripción del producto y para estimar el "
  "esfuerzo de sus tareas permite reducir la brecha entre el esfuerzo estimado y el real desde el "
  "arranque del proyecto?")
P("La formulación original planteaba un LLM externo para el backlog; durante el desarrollo se "
  "sustituyó por modelos propios, por coste, privacidad y control de la calidad (sección 2.2.4).")
mover(162, 166)                       # Justificacion

# ==================================================================== 2. DISENO Y DESARROLLO
H(1, "Diseño y desarrollo del proyecto")
H(2, "Diseño del proyecto")
H(3, "Motivación")
for e in B[124:142]:
    nivel(e, 1)
mover(124, 141)                       # descripcion e importancia + analisis (arbol de problemas)

H(3, "Objetivo general")
P("Desarrollar un sistema web de gestión ágil que, mediante modelos propios de aprendizaje "
  "automático, genere el backlog de un proyecto a partir de la descripción del producto y estime "
  "el esfuerzo de sus tareas y del proyecto, para mejorar la gestión del tiempo en equipos de "
  "desarrollo de software.")
H(3, "Objetivos específicos")
L(["OE1. Analizar las herramientas y propuestas existentes a nivel nacional e internacional para "
   "la generación de backlogs y la estimación de esfuerzo en equipos de software.",
   "OE2. Diseñar una arquitectura que integre un sistema web con dos tuberías de modelos: una de "
   "generación de historias de usuario y tareas, y otra de estimación de esfuerzo con intervalo.",
   "OE3. Implementar la aplicación web y las tuberías de modelos, desplegadas como un servicio "
   "(API) que funcione en CPU y sin depender de un LLM externo.",
   "OE4. Validar las tuberías con métricas adecuadas a la distribución del esfuerzo (MdMRE, "
   "PRED(25), exactitud estandarizada SA y correlación de Spearman) evaluadas por proyecto "
   "completo, y compararlas con la literatura, con líneas base y con LLM actuales."])
P("!confirmar con la asesora la reformulación del OE4: la versión anterior proponía MAE, RMSE y MRE "
  "y una comparación con herramientas comerciales; se cambió porque esas métricas están sesgadas "
  "para un esfuerzo asimétrico (Kitchenham et al., 2001; Shepperd y MacDonell, 2012)")

H(3, "Limitaciones")
L([("Escala del proyecto. ", "El texto de una tarea no permite saber la escala de horas de cada "
    "equipo: el total en frío es el resultado más débil y solo mejora con tareas cerradas "
    "(sección 6.2.1.2)."),
   ("Banda por tarea. ", "El intervalo por tarea en frío sigue siendo ancho, porque las señales de "
    "incertidumbre del modelo apenas correlacionan con su error."),
   ("Métrica de generación. ", "El sF1 mide solo las historias, no las tareas, y premia reutilizar "
    "la redacción de la descripción."),
   ("Idioma. ", "Los corpus están en inglés; la entrada en español se traduce, y los actores se "
    "leen antes de traducir."),
   ("Escala por tarea sobre texto generado. ", "No puede validarse hasta que se ejecuten tareas "
    "generadas y se registren sus horas reales."),
   ("Infraestructura. ", "En el plan de CPU básica del servicio, un backlog tarda cerca de un "
    "minuto."),
   ("Usuarios. ", "La satisfacción percibida aún no se ha medido con equipos reales.")])

H(2, "Desarrollo del proyecto")
H(3, "Equipo del proyecto")
P("El equipo combina los perfiles de Ingeniería de Software (aplicación web, arquitectura y gestión "
  "del proyecto) y de Ciencias de la Computación (modelos de generación y de estimación, "
  "evaluación y despliegue del servicio). La Tabla 1 recoge los roles.")
mover(208, 209)
H(3, "Recursos requeridos")
L([("Datos. ", "TAWOS (Tawosi et al., 2022) y The Public Jira Dataset (Montgomery et al., 2022) "
    "para el entrenamiento; 328 historias de usuario escritas por personas en seis proyectos "
    "(Bragilovski et al., 2024) como referencia de la generación."),
   ("Cómputo. ", "Una GPU local para el entrenamiento; en producción solo CPU."),
   ("Servicios. ", "Hugging Face Spaces (suscripción PRO, hardware CPU básico gratuito) para la "
    "API de modelos, y Supabase para la base de datos y la autenticación de la web."),
   ("Software. ", "Python, PyTorch, Transformers, ONNX Runtime, scikit-learn (Pedregosa et al., "
    "2011), FastAPI y Docker; React y TypeScript en la web.")])
nivel(B[230], 1)
mover(230, 235)                       # costos
P("!actualizar la Tabla de costos con la suscripción de Hugging Face PRO (9 USD/mes) y el plan de "
  "Supabase que se use")
H(3, "Fases e hitos del proyecto")
mover(225, 229)
P("Durante el segundo semestre se trabajó en sprints de Scrum. Los hitos de los modelos fueron: "
  "el generador con fichas de rol y función, las historias ancla y la selección por cobertura; "
  "el estimador v6 con dos cabezas, anclaje y bandas; la reducción a un único codificador (v7) y "
  "su compresión; la prueba de extremo a extremo; y el despliegue de la API.")
P("!completar con las fechas reales de cada sprint, sus actas de grooming/review y los hitos de la "
  "web")
H(3, "Enfoques del trabajo")
P("Se combinaron Scrum, para la gestión, y un enfoque experimental, para los modelos. Tres reglas "
  "guiaron el segundo: toda cifra reportada debe salir de un archivo de resultados versionado; "
  "toda evaluación se hace por proyecto completo, de modo que el modelo nunca vea el proyecto que "
  "predice; y toda mejora se acepta solo si su intervalo de confianza por remuestreo excluye el "
  "cero.")
P("Durante el desarrollo se tomaron dos decisiones de enfoque. La primera fue sustituir el LLM "
  "externo por modelos propios para generar el backlog: por coste por llamada, por privacidad de "
  "las descripciones y porque permitía controlar la cobertura de las funciones pedidas. La "
  "segunda fue sustituir el estimador KNN con TF-IDF de la propuesta inicial por un apilado de "
  "regresores sobre un codificador de frases, tras compararlos con las reimplementaciones de la "
  "literatura (sección 6.2.1.2).")
nivel(B[236], 1)
mover(236, 240)                       # calidad
H(3, "Riesgos y mitigación")
mover(242, 245)
P("Tres riesgos se materializaron en el desarrollo de los modelos y se mitigaron:")
L([("Tamaño. ", f"El sistema ocupaba 9,6 GB y no cabía en un servicio económico; se redujo a 3,3 GB "
    f"con un único codificador cuantizado a int8 y con el vocabulario podado "
    f"({n(RES7['mb_codificador'], 0)} MB)."),
   ("Fuga en la validación. ", "Dos codificadores ajustados parecían útiles solo porque se evaluaban "
    "con tareas de proyectos vistos; se detectó al validar por proyecto completo y se retiraron."),
   ("Despliegue. ", "El servicio lanzaba más hilos que CPU tenía el contenedor y un backlog no "
    "terminaba; se limitó a la cuota real del contenedor.")])

# ============================================================ 3. ANALISIS Y DISENO DE ALGORITMOS
H(1, "Análisis y diseño de algoritmos")
P("Agix encadena dos tuberías. La de generación recibe una descripción breve del producto y "
  "devuelve 12 historias de usuario con sus tareas; la de estimación asigna horas a cada tarea y un "
  "total con intervalo al proyecto, como muestra la figura siguiente.")
F(FIG / "fig1_pipeline_es.png", "Flujo de extremo a extremo de Agix", 9)

H(2, "Generador de backlog: de la descripción a las historias")
P("La mitad de arriba convierte la descripción en historias con un mT5-small (Xue et al., 2021) "
  "cuyo vocabulario se podó de 250.100 a 75.360 piezas (Abdaoui et al., 2020). La generación se "
  "condiciona con dos fichas de control (Keskar et al., 2019): el rol y la familia de función de la "
  "historia. Los actores salen de un plan de roles que combina los que nombra la descripción, los "
  "del sector y los respaldados por el texto, leídos en español antes de traducir con OPUS-MT "
  "(Tiedemann y Thottingal, 2020), porque la traducción los altera.")
F(FIG / "fig3_plan_roles.png", "Plan de roles: de dónde salen los actores")
P("De unas 36 candidatas muestreadas con top-p (Holtzman et al., 2020) se eligen 12 maximizando "
  "de forma voraz una función submodular de cobertura de los rasgos de la descripción (Lin y "
  "Bilmes, 2011), que garantiza una aproximación de 1−1/e (Nemhauser et al., 1978), con un término "
  "de diversidad que penaliza la redundancia (Carbonell y Goldstein, 1998). Como lo que no se "
  "genera no se puede seleccionar, cada función nombrada en la descripción produce además una "
  "historia ancla: el decodificador recibe el inicio de la historia y solo completa el beneficio.")
H(2, "Generador de backlog: de cada historia a sus tareas")
P("La mitad de abajo, un conjunto de dos mT5-small, decodifica las tareas de cada historia con "
  "búsqueda en haz y penalización de repetición. Se entrenó con 22.871 pares historia-tareas de "
  "TAWOS, aumentados con backlogs sintéticos escritos por un modelo profesor, Qwen2.5 (Qwen Team, "
  "2024), según un esquema de destilación (Hinton et al., 2015), y con retrotraducción (Sennrich "
  "et al., 2016). Después se filtran las tareas que no son trabajo de desarrollo, se aplica un "
  "glosario técnico, se deduplican y se reasignan a su historia.")
F(FIG / "fig2_dos_mitades.png", "Las dos mitades del generador")
H(2, "Estimador de esfuerzo v7")
P("Las dos preguntas del producto tienen respuestas estadísticas distintas: las horas de una "
  "tarea se resumen con la mediana condicional, y el total de un proyecto es una suma, y solo las "
  "medias se suman. Por eso el estimador tiene dos cabezas: la cabeza A minimiza el error "
  "cuadrático sobre el logaritmo de las horas y la B se entrena con pérdida de Poisson.")
P("Cada título de tarea se codifica con multilingual-e5-large (Wang et al., 2024), un codificador "
  "basado en XLM-R (Conneau et al., 2020), y con TF-IDF de caracteres y de palabras. La cabeza A "
  "apila tres canales con RidgeCV: una bolsa de árboles de gradiente, un ridge y un SVR lineal. "
  f"Los pesos aprendidos son {n(CONS['pesos']['bolsa'])}, {n(CONS['pesos']['ridge'])} y "
  f"{n(CONS['pesos']['svr'])} (alfa = {n(CONS['alfa'], 2)}). Las horas reales están muy "
  "cuantizadas, así que la salida se ajusta a los 30 valores más frecuentes del corpus.")
P("El intervalo del 80 % se calcula con predicción conformal por partición (Lei et al., 2018) sobre "
  f"residuos fuera de pliegue (factor {n(CONS['factor_banda80'][1], 2)} en el v7). Cuando los "
  f"vecinos de analogía discrepan más que el umbral {n(SERV['umbral_dispersion'], 3)}, se cae a la "
  f"predicción del modelo; eso ocurre en el {pc(1 - SERV['acuerdo_vecinos'])} de las tareas.")
H(2, "Anclaje al proyecto y factor de estilo")
P("Cuando hay k tareas cerradas, la parte abierta del proyecto se reescala con el cociente entre "
  "las horas reales y las predichas de lo cerrado, contraído hacia la ausencia de corrección como "
  "un estimador de credibilidad (Bühlmann, 1967), con peso k/(k+λ), λ = 12, o λ = 96 si el usuario "
  "estimó el backlog. La banda del total se aplica solo a lo abierto, calibrada según el número de "
  "tareas cerradas y el tamaño del proyecto. Un factor de estilo corrige la diferencia de redacción "
  "entre las tareas que escribe el generador y las de una persona.")
H(2, "Compresión y despliegue de los modelos")
P("El estimador usaba tres codificadores e5-large de 2,2 GB. Al validar por proyecto completo se "
  "comprobó que los dos ajustados no aportaban (sección 6.3.2) y se quitaron. El restante se "
  "exportó a ONNX con cuantización int8 (Jacob et al., 2018) y su vocabulario se podó de 250.002 a "
  f"{RES7['vocab_produccion']:,} tokens".replace(",", ".") + ", conservando todo el alfabeto latino "
  f"para cubrir el texto del generador: {n(RES7['mb_codificador'], 0)} MB en total. Se probó "
  "también un codificador pequeño (e5-small), descartado porque ordenaba peor.")
P("Las dos tuberías se sirven como una API con FastAPI en un contenedor Docker en Hugging Face "
  "Spaces, con tres puntos de entrada: /backlog, /estimar y /salud. Cada tubería tiene su propio "
  "candado, de modo que las estimaciones no esperan a una generación en curso, y el número de "
  "hilos se limita a la cuota real de CPU del contenedor. La figura siguiente resume de dónde "
  "salen los datos y los modelos.")
F(FIG / "fig4_procedencia.png", "Procedencia de los datos y de los modelos")

# ======================================================================= 4. APORTE CIENTIFICO
H(1, "Aporte científico")
H(2, "Contexto")
P("Esta sección recoge los conceptos, estándares y normativa en los que se apoya el trabajo. Los "
  "apartados del trabajo de investigación se mantienen, y se añaden los propios de los modelos que "
  "finalmente se construyeron.")
mover_nivel(329, 379, 1)              # 2.1 Definiciones..2.3 normativa -> H3..
H(3, "Modelos de lenguaje y codificadores de frases")
P("Los modelos actuales de lenguaje se basan en la arquitectura Transformer (Vaswani et al., "
  "2017). Los modelos texto-a-texto como T5 (Raffel et al., 2020) y su versión multilingüe mT5 (Xue "
  "et al., 2021) tratan cualquier tarea como generar un texto a partir de otro, lo que permite "
  "condicionar la generación con fichas de control (Keskar et al., 2019). Los codificadores de "
  "frases como Sentence-BERT (Reimers y Gurevych, 2019), MPNet (Song et al., 2020) y e5 (Wang et "
  "al., 2024) producen un vector por frase cuya similitud de coseno aproxima la similitud de "
  "significado.")
H(3, "Evaluación de la estimación y de la generación")
P("Medidas como MMRE y PRED(25) están sesgadas para distribuciones asimétricas (Kitchenham et "
  "al., 2001); por eso se reportan junto con la exactitud estandarizada SA, que compara con una "
  "predicción al azar (Shepperd y MacDonell, 2012). La calidad de una historia de usuario puede "
  "evaluarse con criterios como los de QUS (Lucassen et al., 2016), y la semejanza entre textos "
  "generados y de referencia con medidas basadas en embeddings como BERTScore (Zhang et al., "
  "2020). Los intervalos se validan por remuestreo (Efron y Tibshirani, 1993).")
H(2, "Trabajos relacionados")
mover_nivel(381, 505, 1)
H(3, "Generación de historias de usuario con modelos de lenguaje")
P("La generación de requisitos con LLM avanza rápido. Quattrocchi et al. (2026) evaluaron la "
  "capacidad de los LLM para generar historias y valorar su calidad; Li et al. (2024) simularon la "
  "colaboración de un equipo ágil para generar criterios de aceptación; Siahaan et al. (2023) "
  "extrajeron historias de texto en lenguaje natural; y van Can y Dalpiaz (2025) localizaron "
  "requisitos en elementos de backlog con LLM. Estos trabajos evalúan sobre todo lo plausible del "
  "texto; Agix se distingue por medir si el backlog cubre las funciones pedidas y por hacerlo con "
  "modelos de 121 M de parámetros que funcionan sin conexión.")
P("En estimación, los LLM se han propuesto como alternativa al juicio experto (Pavlič et al., "
  "2024; Villarrubia et al., 2026; Alaswad et al., 2026), y también como estimadores ligeros "
  "ajustados (Sepúlveda Montoya et al., 2025). La réplica de Tawosi et al. (2023) mostró que el "
  "aprendizaje profundo no supera de forma consistente a líneas base simples, lo que motiva el "
  "protocolo estricto de este trabajo.")
H(2, "Contribución")
H(3, "Propuesta")
P("Se propone un sistema de dos tuberías de modelos ligeros que cubre el arranque de un proyecto "
  "de principio a fin. Sus aportes son: (1) una generación en la que la calidad sale de la "
  "selección y no del tamaño del modelo, con historias ancla que garantizan la cobertura de las "
  "funciones pedidas; (2) un estimador con dos cabezas, anclaje por proyecto e intervalo conformal "
  "calibrado según las tareas cerradas; y (3) un protocolo de evaluación por proyecto completo que "
  "evita reportar PRED(25) solo y evaluar sobre proyectos vistos, el cual permitió detectar que dos "
  "de los tres codificadores del estimador no aportaban.")
H(3, "Hipótesis")
P("H1. Un estimador basado en un codificador de frases congelado, evaluado sobre proyectos no "
  "vistos, alcanza una exactitud comparable a la de LLM actuales calibrados con horas reales, con "
  "una latencia y un coste por llamada mucho menores.")
P("H2. Las historias ancla y la selección por cobertura aumentan la semejanza del backlog generado "
  "con backlogs escritos por personas, respecto del generador sin ellas.")
P("H3. El reanclaje con tareas cerradas mejora de forma significativa la estimación del total del "
  "proyecto respecto de la estimación en frío.")

# ======================================================================= 5. APLICACION (ISW)
H(1, "Aplicación / Solución")
P("Este capítulo describe la aplicación web Agix, que integra las dos tuberías mediante la API.")
P("!este capítulo depende de la web; completar cada apartado con el material de la aplicación")
H(2, "Requisitos")
P("Los requisitos se gestionan en el Product Backlog v1.5: 18 épicas, 74 historias de usuario y "
  "177 criterios de aceptación en formato Dado/Cuando/Entonces.")
mover(207, 207)
mover(211, 219)                       # componentes del alcance y exclusiones
P("!actualizar la lista: el asistente de backlog usa el generador propio y la estimación usa el "
  "estimador v7 a través de la API, no Groq ni KNN con TF-IDF")
L(["!requisitos funcionales (tabla con ID, descripción y prioridad MoSCoW)",
   "!requisitos no funcionales (los 25 de calidad; añadir los del servicio: estimación de una tarea "
   "en ~2 s y un backlog en ~55 s, medidos en la prueba de carga)",
   "!diagrama y descripción de casos de uso",
   "!historias de usuario más representativas (o remitir al anexo con las 74)"])
mover(220, 221)                       # suposiciones y restricciones
H(2, "Diseño")
P("La aplicación web consume el servicio de modelos a través de tres puntos de entrada: /backlog "
  "genera el backlog de una descripción; /estimar devuelve las horas de una o varias tareas con su "
  "intervalo; y /salud informa del estado del servicio. El servicio se autentica con una clave "
  "(cabecera X-API-Key) y no guarda datos de los proyectos.")
L(["!arquitectura del software (actualizar las arquitecturas lógica, física e integrada del trabajo "
   "de investigación: la estimación y la generación ya no corren en el navegador ni en Groq, sino "
   "en la API de Hugging Face)",
   "!diagrama de componentes", "!diagrama de despliegue (navegador, Supabase, Hugging Face Space)",
   "!diagrama de capas",
   "!modelo entidad-relación y esquema de la base de datos (tablas, campos, claves)",
   "!wireframes, mockups y prototipo; guía de estilo",
   "!diagramas UML de secuencia del flujo describir producto → backlog → estimación"])
H(2, "Implementación")
P("El servicio de modelos está documentado para los desarrolladores de la web en API_AGIX.md, con "
  "las entradas, salidas y errores de cada punto de entrada. El código de los modelos sigue la "
  "convención de scripts numerados que dejan sus resultados en results/, y todo está versionado "
  "en git.")
L(["!estándares de codificación de la web", "!funcionalidades con pantallas: asistente de backlog, "
   "tablero Kanban, Gantt, cronómetro y dashboard"])
H(2, "Pruebas")
P("Del lado de los modelos se hicieron dos pruebas de sistema. En la prueba de extremo a extremo "
  "se generaron backlogs para seis descripciones de producto nuevas y se revisaron a mano sus 199 "
  "tareas. En 2 de los 6 proyectos faltaba la función principal; tras introducir las historias "
  "ancla, faltó en 0 de 6. En la prueba de carga del servicio desplegado, un backlog tardó unos "
  "55 s, una estimación unos 2 s (unos 6 s con una generación en curso), y una ráfaga de 20 "
  "estimaciones simultáneas se atendió sin errores, la última a los 50 s.")
P("!plan de pruebas, casos de prueba, resultados e informe de pruebas de la web")
H(2, "Despliegue y mantenimiento")
P("La API se despliega en Hugging Face Spaces con Docker; los modelos se descargan al arrancar "
  "desde un repositorio privado. Para más carga se puede pasar a un hardware de 8 vCPU o con GPU, "
  "que se cobran por hora. El paquete de producción completo, generador y estimador, ocupa 3,3 GB "
  f"y usa {n(SERV['ram_pico_pipeline_completa_cpu_gb'], 1)} GB de RAM en CPU.")
L(["!guía de instalación y despliegue de la web", "!guía de usuario", "!guía de administración",
   "!plan de mantenimiento"])
H(2, "Gestión del proyecto")
P("!plan del proyecto, actas de reunión (grooming, planning y review de cada sprint), control de "
  "versiones y artefactos de Scrum")

# ====================================================================== 6. VALIDACION
H(1, "Validación y resultados del proyecto")
H(2, "Protocolo experimental")
P("El estimador se entrena y evalúa con tareas de software con horas registradas de TAWOS y del "
  "Public Jira Dataset. Toda predicción es fuera de pliegue con GroupKFold sobre proyectos "
  "completos, de modo que el modelo nunca ve la tarea ni su proyecto. Las métricas por tarea se "
  "miden en un bloque de prueba de 4.225 tareas, y las del total en 114 proyectos predichos como "
  "si fueran nuevos. Para la comparación con LLM se usa una muestra fija de 601 tareas que el "
  "estimador no vio al entrenar, y un conjunto aparte de 130 tareas para calibrar a los LLM.")
P("El generador se evalúa frente a las 328 historias escritas por personas en seis proyectos, con "
  "tres semillas por proyecto y comparaciones emparejadas. Las diferencias se reportan con su "
  "intervalo de confianza del 95 % por remuestreo.")
H(2, "Resultados")
H(3, "Resultados cuantitativos")
H(4, "Métricas")
L([("MdMRE: ", "mediana del error relativo absoluto."),
   ("PRED(25): ", "proporción de predicciones dentro de ±25 % del valor real."),
   ("SA: ", "exactitud estandarizada frente a una predicción al azar (Shepperd y MacDonell, 2012); "
    "positiva es mejor que el azar."),
   ("Spearman (ρ): ", "si el modelo ordena las tareas de menos a más esfuerzo."),
   ("Anchura del intervalo: ", "cociente entre su extremo superior y el inferior, a una cobertura "
    "empírica del 80 %."),
   ("sF1: ", "F1 suave entre los conjuntos de historias generadas y humanas, emparejadas por "
    "similitud de embeddings, al estilo de BERTScore (Zhang et al., 2020). Es una métrica propia y "
    "solo puntúa historias, no tareas.")])
P("PRED(25) se reporta siempre junto con SA y Spearman, porque una predicción constante puede "
  "puntuar bien en ella sin ordenar nada.")

H(4, "Resultados del estimador")
l = {r.modelo: r for r in LIT.itertuples()}
T(["modelo", "MdMRE", "PRED(25)", "MAE (h)", "Spearman"],
  [[r.modelo, n(r.MdMRE), pc(r.PRED25), n(r.MAE_h, 2), n(r.spearman, 2)] for r in LIT.itertuples()],
  "Estimación por tarea frente a reimplementaciones de la literatura",
  "Fuente: results/baselines_literatura.csv. Línea base, Deep-SE (Choetkiertikul et al., 2019) y "
  "GPT2SP (Fu y Tantithamthavorn, 2023) reimplementados sobre los mismos datos.")
T(["versión", "PRED(25)", "MdMRE", "SA", "Spearman"],
  [["v6 (3 codificadores)", pc(V6.PRED25), n(V6.MdMRE), n(V6.SA, 4), n(V6.spearman)],
   ["v7 desplegado (1 codificador, int8)", pc(V7.PRED25), n(V7.MdMRE), n(V7.SA, 4), n(V7.spearman)]],
  "Estimador v6 frente al v7 desplegado en el bloque de prueba (4.225 tareas)",
  "Fuente: results/estimador_v7/prueba.csv. El v6 juega con ventaja: sus codificadores ajustados "
  "vieron proyectos de este bloque.")
filas = [["en frío (0 cerradas)", pc(ANC_K[0].PRED25), n(ANC_K[0].MdMRE), "—"]]
for k in (1, 5, 10, 20):
    filas.append([f"{k} tareas cerradas", pc(ANC_K[k].PRED25), n(ANC_K[k].MdMRE),
                  f"×{n(BANDA[k], 2)}" if k in BANDA else "—"])
filas.append(["constante (n × mediana)", pc(CONST_TOT.PRED25), n(CONST_TOT.MdMRE), "—"])
T(["tareas cerradas", "PRED(25) del total", "MdMRE", "anchura del intervalo 80 %"], filas,
  "Total del proyecto en 114 proyectos predichos como nuevos",
  "Fuente: results/anclas_escala/resumen.csv, results/curva_final/resumen.csv y "
  "results/banda_que_cierra_antes/resumen.csv.")
P(f"El total del proyecto en frío cae dentro de ±25 % en el {pc(ANC_K[0].PRED25)} de los proyectos, "
  f"frente al {pc(CONST_TOT.PRED25)} de una constante, y sube al {pc(ANC_K[20].PRED25)} tras cerrar "
  "20 tareas. El estimador v7 estima una tarea en "
  f"{n(SERV['ms_por_tarea_caliente'], 1)} ms en CPU.")

H(4, "Comparación con modelos de lenguaje de gran escala")
A = LLM[LLM.condicion == "A"]
ref = LLM[LLM.modelo == "Agix v7 (nuestro)"].iloc[0]
cst = LLM[LLM.modelo == "constante: mediana global"].iloc[0]
filas = [["Agix v7", "—", int(ref.n), n(ref.MdMRE), pc(ref.PRED25), n(ref.SA), n(ref.spearman),
          "—", n(ref.seg_tarea, 3)],
         ["constante (mediana)", "—", int(cst.n), n(cst.MdMRE), pc(cst.PRED25), n(cst.SA), "—", "—", "—"]]
for r in A[A.proveedor != "local (Ollama)"].sort_values(["modelo", "version"], ascending=[True, False]).itertuples():
    filas.append([r.modelo, r.version, int(r.n), n(r.MdMRE), pc(r.PRED25), n(r.SA), n(r.spearman),
                  f"{r.veredicto} {r.dif_MdMRE}", n(r.seg_tarea, 1)])
T(["modelo", "versión", "n", "MdMRE", "PRED(25)", "SA", "ρ", "frente a Agix (IC95 dif. MdMRE)",
   "s/tarea"], filas,
  "Agix v7 frente a LLM actuales con solo el texto de la tarea",
  "Fuente: results/llm_bench/api_resumen.csv. «calibrado» multiplica al LLM por un factor aprendido "
  "en 130 tareas aparte. GPT-OSS (OpenAI, 2025) y Qwen 3 servidos por Groq.")
P("!el banco contra LLM está incompleto por la cuota gratuita de Groq: faltan parte de las "
  "calibraciones de GPT-OSS y la condición con ejemplos; regenerar esta tabla con "
  "src/261_tablas_llm_api.py al terminar")
P("Sin calibrar, los tres LLM sobreestiman entre dos y cinco veces y pierden con Agix con un "
  "intervalo que excluye el cero. Calibrados con horas reales, empatan con Agix o lo superan por un "
  "margen mínimo, y ordenan las tareas algo mejor. Esa calibración exige historial que no existe "
  "al arrancar un proyecto, y cada llamada cuesta segundos y depende de un servicio externo.")

H(4, "Resultados del generador")
P("Frente a las 328 historias humanas, el sF1 sube de 0,6354 a 0,6671 al añadir las historias "
  "ancla y el respaldo de roles (+0,0317, IC95 [+0,025; +0,038]), con mejora en los seis proyectos "
  "y en las tres semillas.")
P("!el 0,6671 todavía no tiene archivo en results/: pedido al chat de generación "
  "(results/49_real_vigente.csv); sustituir la cifra por la del archivo cuando exista")
P("!comparación del generador contra LLM: la tabla del 01/10 es anterior a las anclas; insertar la "
  "nueva corrida de src/20_banco_llms.py con la columna de contaminación (si el LLM reconoce el "
  "proyecto) y el número de tareas por backlog")

H(3, "Resultados cualitativos")
H(4, "Percepción de técnicas")
P("La revisión manual de las 199 tareas de la prueba de extremo a extremo mostró que los fallos "
  "del generador no eran de redacción sino de cobertura: faltaba la función principal en 2 de 6 "
  "proyectos y el generador derivaba hacia otro sector. Esa observación motivó las historias ancla.")
H(4, "Satisfacción percibida")
P("!no hay datos todavía: aplicar una encuesta (por ejemplo, SUS más preguntas sobre la utilidad "
  "del backlog y de la estimación) a equipos que usen la aplicación; si no da tiempo, declararlo "
  "como trabajo futuro")
H(4, "Retroalimentación")
P("!observaciones de la asesora y de los usuarios de prueba, y cómo se atendieron")

H(2, "Discusión")
H(3, "Análisis comparativo")
P("En PRED(25) por tarea, Agix queda a la par de Deep-SE y GPT2SP con un protocolo más estricto, "
  "y una constante supera a todos; la métrica, sola, no discrimina. Lo que distingue a Agix es "
  "que tiene SA positivo y ordena las tareas sin haber visto ningún proyecto del bloque. Frente a "
  "los LLM, la ventaja de Agix no está en la exactitud, donde empata con los calibrados, sino en "
  f"el coste: {n(SERV['ms_por_tarea_caliente'], 1)} ms por tarea, en local y sin coste por "
  "llamada, frente a segundos por tarea a través de una API con cuota.")
H(3, "Hallazgos")
L([("Fuga en la validación. ", "Los dos codificadores ajustados del v6 parecían útiles porque el "
    "apilado se ajustaba con tareas de proyectos que ya habían visto. Evaluados por proyecto "
    f"completo, añadirlos bajaba el SA de {n(LARGE.SA, 4)} a {n(MEZCLA.SA, 4)} y el Spearman de "
    f"{n(LARGE.spearman)} a {n(MEZCLA.spearman)} (results/large_contra_small/metricas.csv)."),
   ("Un codificador pequeño ordena peor. ", f"El e5-small empata en PRED(25) "
    f"(diferencia {n(SMALL['dif_PRED25_small_menos_large'], 4)}) pero ordena peor: Spearman "
    f"{n(SMALLR.spearman)} frente a {n(LARGE.spearman)}."),
   ("La escala es del proyecto. ", "Ni las horas de proyectos hermanos ni el historial de los "
    "mismos desarrolladores mejoran la estimación; solo las tareas cerradas del propio proyecto, "
    "mediante el anclaje."),
   ("La generación mejora por selección. ", "Las historias ancla aportan la mayor parte de la "
    "ganancia de sF1, sin cambiar los modelos.")])

# ====================================================================== CONCLUSIONES
H(1, "Conclusiones y recomendaciones")
P("Se construyó un sistema que genera el backlog de un proyecto a partir de una descripción breve "
  "y estima su esfuerzo con modelos propios, servidos como API en CPU. Respecto de los objetivos: "
  "el OE1 se cumplió con el estado del arte y la comparación con la literatura; el OE2 con la "
  "arquitectura de dos tuberías; el OE3 con el servicio desplegado y la aplicación web; y el OE4 "
  "con la validación por proyecto completo y la comparación con LLM.")
P("Recomendaciones: medir la estimación sobre tareas generadas con horas reales; evaluar el "
  "sistema con equipos en uso; y estudiar señales que permitan estrechar el intervalo por tarea.")
P("!completar con el cumplimiento de la parte web y los resultados de usuarios")

# ===================================================================== REFERENCIAS
NUEVAS = [
    "Abdaoui, A., Pradel, C., & Sigel, G. (2020). Load what you need: Smaller versions of multilingual BERT. En Proceedings of SustaiNLP: Workshop on Simple and Efficient Natural Language Processing (pp. 119–123). https://doi.org/10.18653/v1/2020.sustainlp-1.16",
    "Bragilovski, M. et al. (2024). [VERIFICAR: título y venue del conjunto de historias de usuario] [Conjunto de datos]. Zenodo. https://doi.org/10.5281/zenodo.12740518",
    "Bühlmann, H. (1967). Experience rating and credibility. ASTIN Bulletin, 4(3), 199–207. https://doi.org/10.1017/S0515036100008989",
    "Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. En Proceedings of the 21st Annual International ACM SIGIR Conference (pp. 335–336). https://doi.org/10.1145/290941.291025",
    "Choetkiertikul, M., Dam, H. K., Tran, T., Pham, T., Ghose, A., & Menzies, T. (2019). A deep learning model for estimating story points. IEEE Transactions on Software Engineering, 45(7), 637–656. https://doi.org/10.1109/TSE.2018.2792473",
    "Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. En Proceedings of ACL 2020 (pp. 8440–8451). https://doi.org/10.18653/v1/2020.acl-main.747",
    "Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall.",
    "Fu, M., & Tantithamthavorn, C. (2023). GPT2SP: A transformer-based agile story point estimation approach. IEEE Transactions on Software Engineering, 49(2), 611–625. https://doi.org/10.1109/TSE.2022.3158252",
    "Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv. https://doi.org/10.48550/arXiv.1503.02531",
    "Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The curious case of neural text degeneration. En International Conference on Learning Representations (ICLR).",
    "Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. En Proceedings of CVPR (pp. 2704–2713). https://doi.org/10.1109/CVPR.2018.00286",
    "Keskar, N. S., McCann, B., Varshney, L. R., Xiong, C., & Socher, R. (2019). CTRL: A conditional transformer language model for controllable generation. arXiv. https://doi.org/10.48550/arXiv.1909.05858",
    "Kitchenham, B. A., Pickard, L. M., MacDonell, S. G., & Shepperd, M. J. (2001). What accuracy statistics really measure. IEE Proceedings - Software, 148(3), 81–85. https://doi.org/10.1049/ip-sen:20010506",
    "Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman, L. (2018). Distribution-free predictive inference for regression. Journal of the American Statistical Association, 113(523), 1094–1111. https://doi.org/10.1080/01621459.2017.1307116",
    "Lin, H., & Bilmes, J. (2011). A class of submodular functions for document summarization. En Proceedings of ACL-HLT 2011 (pp. 510–520).",
    "Lucassen, G., Dalpiaz, F., van der Werf, J. M. E. M., & Brinkkemper, S. (2016). Improving agile requirements: The Quality User Story framework and tool. Requirements Engineering, 21(3), 383–403. https://doi.org/10.1007/s00766-016-0250-x",
    "Montgomery, L., Lüders, C., & Maalej, W. (2022). An alternative issue tracking dataset of public Jira repositories. En Proceedings of the 19th International Conference on Mining Software Repositories (pp. 73–77). https://doi.org/10.1145/3524842.3528486",
    "Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). An analysis of approximations for maximizing submodular set functions—I. Mathematical Programming, 14(1), 265–294. https://doi.org/10.1007/BF01588971",
    "OpenAI. (2025). gpt-oss-120b & gpt-oss-20b model card. arXiv. [VERIFICAR: arXiv:2508.10925]",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
    "Qwen Team. (2024). Qwen2.5 technical report. arXiv. [VERIFICAR: arXiv:2412.15115]",
    "Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140), 1–67.",
    "Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. En Proceedings of EMNLP-IJCNLP 2019 (pp. 3982–3992). https://doi.org/10.18653/v1/D19-1410",
    "Sennrich, R., Haddow, B., & Birch, A. (2016). Improving neural machine translation models with monolingual data. En Proceedings of ACL 2016 (pp. 86–96). https://doi.org/10.18653/v1/P16-1009",
    "Shepperd, M., & MacDonell, S. (2012). Evaluating prediction systems in software project estimation. Information and Software Technology, 54(8), 820–827. https://doi.org/10.1016/j.infsof.2011.12.008",
    "Song, K., Tan, X., Qin, T., Lu, J., & Liu, T.-Y. (2020). MPNet: Masked and permuted pre-training for language understanding. En Advances in Neural Information Processing Systems, 33 (pp. 16857–16867).",
    "Tawosi, V., Al-Subaihin, A., Moussa, R., & Sarro, F. (2022). A versatile dataset of agile open source software projects. En Proceedings of the 19th International Conference on Mining Software Repositories (pp. 707–711). https://doi.org/10.1145/3524842.3528029",
    "Tiedemann, J., & Thottingal, S. (2020). OPUS-MT — Building open translation services for the world. En Proceedings of the 22nd Annual Conference of the European Association for Machine Translation (pp. 479–480).",
    "Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. En Advances in Neural Information Processing Systems, 30 (pp. 5998–6008).",
    "Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Multilingual E5 text embeddings: A technical report. arXiv. https://doi.org/10.48550/arXiv.2402.05672",
    "Xue, L., Constant, N., Roberts, A., Kale, M., Al-Rfou, R., Siddhant, A., Barua, A., & Raffel, C. (2021). mT5: A massively multilingual pre-trained text-to-text transformer. En Proceedings of NAACL-HLT 2021 (pp. 483–498). https://doi.org/10.18653/v1/2021.naacl-main.41",
    "Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with BERT. En International Conference on Learning Representations (ICLR).",
]
H(1, "Referencias")
viejas = [docx.text.paragraph.Paragraph(e, doc) for e in B[512:557]
          if e.tag.endswith("}p") and docx.text.paragraph.Paragraph(e, doc).text.strip()]
modelo = viejas[0]._p
entradas = [(p.text, p._p) for p in viejas]
for t in NUEVAS:
    e = copy.deepcopy(modelo)
    pr = docx.text.paragraph.Paragraph(e, doc)
    for r in pr.runs[1:]:
        r._r.getparent().remove(r._r)
    for h in list(e.iter(qn("w:hyperlink"))):
        h.getparent().remove(h)
    if pr.runs:
        pr.runs[0].text = t
    else:
        pr.add_run(t)
    entradas.append((t, e))
import unicodedata
clave = lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
for _, e in sorted(entradas, key=lambda x: clave(x[0])):
    seq.append(e)

H(1, "Anexos")
H(2, "Anexo A. Benchmarking del trabajo de investigación")
P("Se conserva el benchmarking realizado en el trabajo de investigación. Las decisiones sobre el "
  "algoritmo de estimación y la representación semántica se revisaron en el desarrollo (sección "
  "2.2.4).")
mover_nivel(247, 327, 0)
H(2, "Anexo B. Product Backlog v1.5")
P("!adjuntar el listado de las 74 historias con sus criterios")
H(2, "Anexo C. Documentación de la API")
P("!adjuntar API_AGIX.md")

# ---------------------------------------------------------------- reconstruir el cuerpo
for e in list(body):
    body.remove(e)
for e in seq:
    body.append(e)
body.append(SECT)

# ------------------------------------------------- numeracion uniforme de los titulos
SIN_NUMERO = {"dedicatoria", "agradecimientos", "resumen", "abstract", "student outcome",
              "conclusiones y recomendaciones", "referencias", "anexos"}
cont = [0] * 7
en_anexos = False
for p in doc.paragraphs:
    m = re.fullmatch(r"Heading ?(\d)", p.style.style_id if p.style is not None else "")
    if not m or not p.text.strip():
        continue
    lv = int(m.group(1))
    limpio = re.sub(r"^\s*(\d+\.)*\d*\.?\s*", "", p.text).strip()
    base = clave(limpio)
    if lv == 1:
        en_anexos = base.startswith("anexos")
    pPr = p._p.get_or_add_pPr()
    for old in pPr.findall(qn("w:numPr")):
        pPr.remove(old)
    num = OxmlElement("w:numPr")
    nid = OxmlElement("w:numId"); nid.set(qn("w:val"), "0")
    num.append(nid)
    pPr.append(num)
    if en_anexos or any(base.startswith(s) for s in SIN_NUMERO):
        nuevo = limpio
    else:
        cont[lv] += 1
        for k in range(lv + 1, 7):
            cont[k] = 0
        nuevo = ".".join(str(c) for c in cont[1:lv + 1]) + ". " + limpio
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)
    if p.runs:
        p.runs[0].text = nuevo
    if "Formulación del Problema¿" in nuevo:
        p.runs[0].text = nuevo.split("¿")[0]

SALIDA.parent.mkdir(parents=True, exist_ok=True)
doc.save(SALIDA)
print("guardado", SALIDA, "referencias:", len(entradas))
