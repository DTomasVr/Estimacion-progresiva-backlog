"""Deck 1: how the two datasets were built, cleaned and merged, and which to use.

Written for someone who has not read the report. The two corpora are called
Dataset 1 and Dataset 2 throughout, as requested. Every number on a slide comes
from the delivered files, so the deck and the anexo cannot disagree.
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")
from pptx.util import Inches, Pt  # noqa: E402

import pptx_comun as P  # noqa: E402

FIG = Path("reports/figuras_pptx")
SALIDA = "entrega/presentaciones/1_construccion_datasets.pptx"


def main():
    prs = P.nueva()

    # ------------------------------------------------------------------ 1
    P.portada(prs, "Cómo se construyeron\nlos dos conjuntos de datos",
              "De qué fuentes salen, qué se descartó y por qué, cómo se "
              "fusionaron,\ny cuál conviene para entrenar el modelo final.",
              "Anexo de datos · 10 de septiembre de 2026")

    # ------------------------------------------------------------------ 2
    s, y = P.lamina(prs, "Qué tiene que predecir el modelo",
                    "Agix genera las tareas de un proyecto con un LLM. En ese "
                    "momento, sin que nadie haya trabajado todavía, el modelo "
                    "debe decir cuántas horas llevará cada tarea.")
    y = P.vinetas(s, y, [
        ("OBJETIVO", "Horas de reloj realmente registradas. Nunca puntos de "
                     "historia, nunca duración de calendario."),
        ("ENTRADA", "El texto de la tarea: título y descripción."),
        ("PROBLEMA", "Ningún conjunto público reúne las dos cosas en cantidad. "
                     "Hubo que construirlo cruzando fuentes."),
    ])
    y += Inches(0.15)
    y = P.veredicto(s, y, "la decisión que condiciona todo lo demás",
                    "Se midió que convertir puntos de historia a horas no "
                    "funciona: R² = 0,058. El convertidor acaba siendo una "
                    "constante. Por eso ninguna hora del conjunto procede de "
                    "una conversión.")
    P.tabla(s, y + Inches(0.25),
            ["_Fuente", "Organizaciones", "Aporta", "Objetivo"],
            [["TAWOS (Jira abierto)", "26 proyectos", "9 559 tareas", "Timespent / 60"],
             ["SiP (empresa comercial)", "20 proyectos", "7 791 tareas", "HoursActual"],
             ["The Public Jira Dataset", "10 organizaciones", "72 501 tareas",
              "timespent / 3600"]],
            anchos=[3.1, 2.0, 1.8, 2.0], tam=14)

    # ------------------------------------------------------------------ 3
    P.seccion(prs, "1", "Dataset 1",
              "TAWOS + SiP · 18 474 filas · 17 350 tareas de software\n"
              "Dos organizaciones: 26 proyectos abiertos y 20 de una empresa comercial.")

    # ------------------------------------------------------------------ 4
    s, y = P.lamina(prs, "Dataset 1 · de dónde sale",
                    "Dos adquisiciones muy distintas, unificadas en un solo "
                    "esquema.")
    y = P.vinetas(s, y, [
        ("TAWOS", "Volcado MySQL de 4,31 GB. Se lee en flujo sin instalar "
                  "MySQL: mysqldump escapa los saltos de línea, así que cada "
                  "INSERT ocupa una línea física."),
        ("SiP", "Dos CSV de una empresa británica. Una fila por desarrollador "
                "y tarea, con las horas del equipo repetidas en todas ellas."),
        ("UNIFICAR", "Taxonomía común de categorías (bug, feature, task…), "
                     "conservando siempre la etiqueta original."),
    ])
    y += Inches(0.2)
    P.veredicto(s, y, "trabajo no informático",
                "SiP incluye marketing, selección de personal y reuniones de "
                "consejo. Esas filas se conservan marcadas con "
                "is_software_task = 0 y quedan fuera de las particiones. Es una "
                "decisión reversible, no una eliminación.")

    # ------------------------------------------------------------------ 5
    s, y = P.lamina(prs, "Dataset 1 · primera limpieza",
                    "Cada criterio responde a que la fila no representa lo que "
                    "el modelo tiene que predecir.")
    y = P.tabla(s, y,
                ["_Criterio", "_Qué hace", "_Por qué", "Filas"],
                [["D8 · normalizar texto", "reescribe", "URL, correos, trazas de pila: dos incidencias iguales salvo la traza son el mismo texto", "0"],
                 ["D6 · texto < 3 palabras", "descarta", "no hay nada que vectorizar", "465"],
                 ["D5 · esfuerzo < 0,1 h", "descarta", "6 minutos no es una tarea", "408"],
                 ["D4 · SiP > 200 h", "descarta", "líneas de presupuesto paraguas", "43"],
                 ["D1a · mismo texto, esfuerzo muy distinto", "descarta grupo", "si varía más de 10×, el texto no determina el esfuerzo", "229"],
                 ["D2 · duplicados restantes", "colapsa", "evita repetir el mismo ejemplo y repartirlo entre bloques", "665→215"],
                 ["D3a · casi duplicados (coseno ≥ 0,95)", "agrupa", "impide que textos casi iguales caigan a ambos lados", "0"]],
                anchos=[3.0, 1.3, 5.4, 0.9], tam=12.5, alto_fila=Inches(0.42))
    P.veredicto(s, y + Inches(0.2), "el orden importa",
                "El texto se normaliza ANTES de buscar duplicados. Si se hiciera "
                "al revés, dos incidencias que solo difieren en una traza de "
                "pila se contarían como distintas.")

    # ------------------------------------------------------------------ 6
    s, y = P.lamina(prs, "Dataset 1 · segunda limpieza",
                    "El cronómetro de Agix mide una persona trabajando en una "
                    "tarea. Una fila que signifique otra cosa enseña una "
                    "magnitud que nunca se va a pedir.")
    y = P.tabla(s, y,
                ["_Criterio", "_Qué hace", "_Por qué", "Filas"],
                [["E3 · tareas de varias personas (SiP)", "reexpresa", "HoursActual es la suma del equipo; se toma el contribuyente principal", "606"],
                 ["E1 · épicas", "descarta", "son contenedores por definición", "15"],
                 ["E2 · contenedores > 16 h", "descarta", "proyectos, planes, reuniones, cajones de sastre", "109"],
                 ["E4 · esfuerzo en horas enteras", "marca", "el objetivo está cuantizado: se documenta, no se altera", "7 012"]],
                anchos=[3.2, 1.3, 5.2, 0.9], tam=13, alto_fila=Inches(0.44))
    y += Inches(0.25)
    y = P.veredicto(s, y, "E3 va primero, y eso cambia el resultado de E2",
                    "115 filas superan las 16 h con las horas del equipo, pero "
                    "solo 109 con las horas ya reexpresadas por desarrollador. "
                    "Seis filas dejan de ser contenedores al medirlas bien.")
    P.nota(s, "Verificado en el cuaderno: las 465, 408, 43, 15 y 109 filas "
              "afectadas están todas ausentes del fichero de salida.")

    # ------------------------------------------------------------------ 7
    s, y = P.lamina(prs, "Dataset 1 · resultado")
    P.figura(s, FIG / "limpieza.png", P.MARGEN, y, P.UTIL)
    y2 = y + Inches(3.05)
    w = (P.UTIL - Inches(0.6)) / 4
    for i, (v, e, t) in enumerate([
            ("18 474", "filas finales", "tinta"),
            ("17 350", "tareas de software", "tinta"),
            ("8,8 %", "descartado en total", "ambar"),
            ("2", "organizaciones", "gris")]):
        P.cifra(s, P.MARGEN + i * (w + Inches(0.2)), y2, w, v, e, t, tam=34)
    P.veredicto(s, y2 + Inches(1.5), "el límite del Dataset 1",
                "Dos organizaciones. Bastan para entrenar, pero no para saber "
                "si el modelo funciona en una empresa que no ha visto nunca "
                "— que es justo lo que le pasa a Agix con cada cliente nuevo.")

    # ------------------------------------------------------------------ 8
    P.seccion(prs, "2", "Dataset 2",
              "Dataset 1 + The Public Jira Dataset · 90 975 filas · 89 851 "
              "tareas de software\nDoce organizaciones, 533 proyectos.")

    # ------------------------------------------------------------------ 9
    s, y = P.lamina(prs, "Dataset 2 · la fuente nueva",
                    "2 686 282 incidencias de dieciséis Jiras públicos. Se "
                    "distribuye como un volcado de MongoDB de 5,87 GB que "
                    "ocupa unos 60 GB restaurado. No se restauró.")
    y = P.vinetas(s, y, [
        ("CÓMO", "El archivo es un único flujo gzip. Cada incidencia se "
                 "recorre a nivel de byte, saltando por su prefijo de longitud "
                 "los ~300 campos vacíos, el historial y los comentarios."),
        ("COSTE", "2,69 millones de incidencias en 383 segundos, sin instalar "
                  "nada y sin escribir 60 GB en disco."),
        ("POR QUÉ", "Es el único corpus que trae la estimación humana junto al "
                    "tiempo real, en más de una organización."),
    ])
    y += Inches(0.15)
    y = P.tabla(s, y,
                ["_Dónde se van los 5,87 GB", "% del volcado", "_Se usa"],
                [["historial de cambios (32 M)", "33,2 %", "no"],
                 ["comentarios (9 M)", "25,8 %", "no"],
                 ["campos personalizados (~300 por incidencia)", "10,7 %", "no"],
                 ["descripción de la tarea", "4,2 %", "SÍ"],
                 ["proyecto, estado, tipo, asignado, fechas…", "~12 %", "SÍ"]],
                anchos=[6.2, 2.0, 1.4], tam=13.5)
    P.nota(s, "Solo el 2,93 % de las incidencias registra alguna hora: el "
              "registro de trabajo es opcional en Jira. Ese 2,93 % son 78 634 tareas.")

    # ------------------------------------------------------------------ 10
    s, y = P.lamina(prs, "Dataset 2 · qué se descarta del bloque nuevo")
    y = P.tabla(s, y,
                ["_Criterio", "_Por qué", "Filas"],
                [["D1 · esfuerzo fuera de [0,05 h , 200 h]", "el máximo bruto eran 208 571 horas; el mínimo, 6 segundos", "311"],
                 ["D2 · tipo contenedor (Epic, New Project)", "agrupan trabajo de otras incidencias", "125"],
                 ["D3 · el tiempo pertenece a las subtareas", "agg_timespent > timespent y tiene hijos", "652"],
                 ["D4 · sin texto utilizable", "nada que vectorizar", "0"],
                 ["D5 · ya presente a través de TAWOS", "los dos corpus minan los mismos Jiras", "5 045"],
                 ["D6 · texto ya en validación o prueba", "guarda de fuga", "0"]],
                anchos=[4.4, 5.0, 1.0], tam=13.5, alto_fila=Inches(0.42),
                resaltar={4: P.AVISO_SUAVE})
    y += Inches(0.22)
    y = P.veredicto(s, y, "D5 no es una precaución teórica", "5 045 incidencias "
                    "aparecían literalmente dos veces. Sin desduplicar, parte "
                    "del «corpus nuevo» sería el viejo contado otra vez, y "
                    "cualquier mejora medida estaría inflada.", tono="aviso")
    P.nota(s, "Sobreviven 72 501 de 78 634 (92,2 %). D4 y D6 descartan cero: "
              "es lo que se espera de una guarda que se declara y se comprueba.")

    # ------------------------------------------------------------------ 11
    s, y = P.lamina(prs, "Dataset 2 · cómo se fusionan",
                    "Una sola regla gobierna la mezcla, y de ella dependen "
                    "todos los resultados posteriores.")
    y = P.veredicto(s, y, "regla central",
                    "El bloque de PRUEBA del Dataset 1 queda intacto, fila por "
                    "fila. Lo nuevo solo entra en entrenamiento, en validación, "
                    "o en un bloque externo que nunca se usa para entrenar.",
                    tono="aviso")
    P.figura_ajustada(s, FIG / "merge.png", y + Inches(0.28),
                      P.ALTO - Inches(0.95) - (y + Inches(0.28)))
    P.nota(s, "Si al añadir datos se cambiara también el examen, la comparación "
              "mediría la dificultad de dos exámenes distintos en vez de la "
              "calidad de dos modelos.")

    # ------------------------------------------------------------------ 12
    s, y = P.lamina(prs, "Dataset 2 · las seis comprobaciones",
                    "El cuaderno de entrega falla si alguna de estas no pasa.")
    y = P.tabla(s, y,
                ["_Qué se comprueba", "Esperado", "Obtenido", "_"],
                [["El bloque de prueba del Dataset 1 es idéntico", "sí", "sí", "PASA"],
                 ["Ninguna fila nueva entró en la prueba", "0", "0", "PASA"],
                 ["Textos compartidos entre entrenamiento y prueba", "0", "0", "PASA"],
                 ["Textos compartidos entre validación y prueba", "0", "0", "PASA"],
                 ["Proyectos del bloque externo vistos al entrenar", "0", "0", "PASA"],
                 ["Identificadores duplicados", "0", "0", "PASA"]],
                anchos=[6.0, 1.4, 1.4, 1.3], tam=13.5, alto_fila=Inches(0.42))
    y += Inches(0.25)
    P.veredicto(s, y, "por qué se retienen proyectos enteros y no filas sueltas",
                "Si un proyecto apareciera a la vez en entrenamiento y en el "
                "bloque externo, el modelo ya conocería la escala de esa "
                "organización y el «arranque en frío» sería falso.")

    # ------------------------------------------------------------------ 13
    s, y = P.lamina(prs, "Lo que hay que declarar del Dataset 2",
                    "El bloque nuevo no procede de la misma distribución. No es "
                    "un defecto: es una diferencia real de cultura de registro.")
    P.figura(s, FIG / "distribuciones.png", P.MARGEN, y, P.UTIL)
    P.veredicto(s, y + Inches(3.15), "consecuencia práctica",
                "En proyectos abiertos se apunta el rato concreto dedicado —el "
                "valor más frecuente son 20 minutos—, mientras que SiP mide con "
                "cronómetro la tarea entera. Entrenar mezclando sin más desplaza "
                "las predicciones hacia abajo, y hay que corregirlo "
                "explícitamente.", tono="ambar")

    # ------------------------------------------------------------------ 14
    P.seccion(prs, "3", "¿Cuál conviene para el entrenamiento final?",
              "La respuesta depende de qué se le vaya a pedir al modelo.")

    # ------------------------------------------------------------------ 15
    s, y = P.lamina(prs, "Los dos escenarios dan respuestas distintas")
    P.figura(s, FIG / "ab_techo.png", P.MARGEN, y, P.UTIL)
    y2 = y + Inches(3.15)
    y2 = P.vinetas(s, y2, [
        ("MISMO CLIENTE", "Sobre proyectos que el modelo ya conoce, los datos "
                          "extra no cambian nada: 22 configuraciones caben "
                          "entre 0,681 y 0,705."),
        ("CLIENTE NUEVO", "Sobre 150 organizaciones nunca vistas, el Dataset 1 "
                          "por sí solo da 1,1407 — peor que responder siempre "
                          "la mediana global. El Dataset 2 lo arregla."),
    ], tam=15.5, salto=Inches(0.46))

    # ------------------------------------------------------------------ 16
    s, y = P.lamina(prs, "Recomendación")
    y = P.veredicto(s, y, "para el entrenamiento final: Dataset 2",
                    "El escenario real de Agix es un cliente nuevo con un "
                    "proyecto nuevo. Ahí un modelo entrenado solo con el "
                    "Dataset 1 no transfiere: se equivoca más que una "
                    "constante. Con el Dataset 2, el error mediano baja un 48 % "
                    "sobre las mismas 13 255 tareas.")
    y += Inches(0.25)
    y = P.tabla(s, y,
                ["_Sobre 150 organizaciones nuevas (13 255 tareas)", "MdMRE",
                 "PRED25", "Spearman"],
                [["Entrenado solo con el Dataset 1", "1,1407", "0,125", "0,220"],
                 ["Entrenado con el Dataset 2", "0,7032", "0,159", "0,167"],
                 ["Dataset 2 + razón contra la mediana del proyecto",
                  "0,5906", "0,233", "0,564"]],
                anchos=[6.4, 1.5, 1.5, 1.5], tam=14, alto_fila=Inches(0.44),
                resaltar={2: P.SUAVE})
    y += Inches(0.28)
    P.vinetas(s, y, [
        "Usar el Dataset 1 solo si hay que comparar con la literatura sobre su "
        "partición original, o para reproducir el protocolo preinscrito.",
        "El Dataset 2 conserva el Dataset 1 completo dentro: no se pierde nada "
        "al elegirlo.",
        "Al entrenar con el Dataset 2 hay que corregir el desplazamiento de "
        "distribución, no ignorarlo.",
    ], tam=15.5, salto=Inches(0.44))

    # ------------------------------------------------------------------ 17
    s, y = P.lamina(prs, "Resumen")
    y = P.tabla(s, y,
                ["_", "Dataset 1", "Dataset 2"],
                [["Filas", "18 474", "90 975"],
                 ["Tareas de software", "17 350", "89 851"],
                 ["Organizaciones", "2", "12"],
                 ["Proyectos", "46", "533"],
                 ["Filas de entrenamiento", "10 597", "63 926"],
                 ["Bloque de prueba", "4 225", "4 225 · el mismo"],
                 ["Bloque externo (cliente nuevo)", "—", "13 255 · 150 proyectos"],
                 ["Con estimación humana", "7 791", "13 283"],
                 ["Mediana de esfuerzo", "2,00 h", "1,00 h"]],
                anchos=[4.6, 3.0, 4.4], tam=14, alto_fila=Inches(0.40),
                resaltar={6: P.SUAVE})
    P.veredicto(s, y + Inches(0.25), "en una frase",
                "El Dataset 2 es el Dataset 1 más 72 501 tareas de diez "
                "organizaciones nuevas, fusionadas sin tocar una sola fila del "
                "examen, y es el que conviene para entrenar el modelo que Agix "
                "va a desplegar.")

    P.numerar(prs)
    Path(SALIDA).parent.mkdir(parents=True, exist_ok=True)
    prs.save(SALIDA)
    print(f"escrito {SALIDA}  ({len(prs.slides.__iter__.__self__._sldIdLst)} "
          f"diapositivas)")


if __name__ == "__main__":
    main()
