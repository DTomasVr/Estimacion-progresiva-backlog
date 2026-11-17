"""Deck 2: pipeline, experimental results, and what can be done with them.

Shorter than deck 1 and blunter. The headline result is negative -- no
configuration meets the acceptance criteria, and neither does a human expert --
so the deck spends its second half on what the artefact does deliver and on the
routes that are still open, which is the part that is actually decidable.
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")
from pptx.util import Inches  # noqa: E402

import pptx_comun as P  # noqa: E402

FIG = Path("reports/figuras_pptx")
SALIDA = "entrega/presentaciones/2_experimentos_resultados.pptx"


def main():
    prs = P.nueva()

    # ------------------------------------------------------------------ 1
    P.portada(prs, "Experimentos, resultados\ny qué se puede hacer con ellos",
              "279 corridas registradas sobre 89 851 tareas de doce "
              "organizaciones.\nQué funciona, qué no, y qué opciones quedan "
              "abiertas.",
              "Resultados experimentales · 10 de septiembre de 2026")

    # ------------------------------------------------------------------ 2
    s, y = P.lamina(prs, "La tubería completa",
                    "Del texto de una tarea a una estimación con intervalo.")
    P.figura(s, FIG / "pipeline.png", P.MARGEN, y + Inches(0.1), P.UTIL)
    y2 = y + Inches(2.5)
    y2 = P.vinetas(s, y2, [
        ("1", "El texto pasa por un SBERT multilingüe congelado. Multilingüe "
              "porque el corpus es inglés y PBE opera en español."),
        ("2", "Al vector se le añaden la categoría de la tarea, la fuente y "
              "cuatro variables numéricas (enlaces, componentes, asignación)."),
        ("3", "Un HistGradientBoosting predice el logaritmo de las horas."),
        ("4", "El resultado se multiplica por la mediana del proyecto, que la "
              "aplicación conoce con las tareas ya cerradas."),
        ("5", "Se acompaña de un intervalo conforme y de un aviso si la tarea "
              "parece grande."),
    ], tam=15, salto=Inches(0.42))

    # ------------------------------------------------------------------ 3
    s, y = P.lamina(prs, "Cómo se evita engañarse",
                    "El protocolo se fijó antes de mirar los resultados.")
    y = P.vinetas(s, y, [
        ("SELECCIÓN", "Toda decisión de modelo se toma mirando SOLO el bloque "
                      "de validación."),
        ("UNA LECTURA", "El bloque de prueba se lee una sola vez, al final. Lo "
                        "que se reporte después se marca como posterior."),
        ("SIN FUGA", "Particiones cronológicas dentro de cada proyecto; ningún "
                     "texto repetido cruza la frontera."),
        ("TRAZABILIDAD", "Cada una de las 279 corridas guarda el SHA-256 del "
                         "fichero de datos con el que se produjo."),
    ])
    y += Inches(0.2)
    P.veredicto(s, y, "un error que se detectó y se corrigió",
                "Un codificador ajustado antes del protocolo tenía 2 981 de las "
                "4 225 tareas de prueba en su propio entrenamiento: un 70,9 % "
                "de fuga. Se reentrenó desde cero solo con el bloque temporal "
                "de entrenamiento y todos los resultados se rehicieron.",
                tono="aviso")

    # ------------------------------------------------------------------ 4
    P.seccion(prs, "1", "Qué se midió",
              "Cuatro resultados, dos de ellos negativos.")

    # ------------------------------------------------------------------ 5
    s, y = P.lamina(prs, "Resultado 1 · más datos no mueven la aguja",
                    "Seis veces más entrenamiento y diez organizaciones nuevas, "
                    "sobre el mismo examen de 4 225 tareas.")
    y = P.tabla(s, y,
                ["_Configuración", "n entrenamiento", "Validación", "Prueba"],
                [["Solo Dataset 1 (control)", "10 597", "0,6086", "0,6869"],
                 ["Agrupado sin ajuste", "63 926", "0,6146", "0,6854"],
                 ["Agrupado, Dataset 1 con peso ×12", "63 926", "0,6140", "0,6812"],
                 ["Agrupado + recalibrado de mediana", "63 926", "0,6146", "0,6865"],
                 ["Bloque nuevo recortado al rango", "59 512", "0,6153", "0,6815"],
                 ["Transferencia como rasgo extra", "63 926", "0,6167", "0,6887"]],
                anchos=[5.4, 2.2, 1.9, 1.9], tam=14, alto_fila=Inches(0.42),
                resaltar={0: P.SUAVE})
    y += Inches(0.25)
    y = P.veredicto(s, y, "la selección honesta elige el control",
                    "Las 22 configuraciones caben entre 0,681 y 0,705, un rango "
                    "más estrecho que la diferencia entre dos semillas "
                    "aleatorias.")
    P.nota(s, "MdMRE = error relativo mediano. Menos es mejor. Una constante da 0,73.")

    # ------------------------------------------------------------------ 6
    s, y = P.lamina(prs, "Resultado 2 · el techo no es de datos, ni de algoritmo, "
                         "ni de representación",
                    "Tres explicaciones posibles, las tres descartadas por "
                    "medición.")
    y = P.tabla(s, y,
                ["_Explicación candidata", "_Cómo se probó", "_Resultado"],
                [["Faltan datos", "seis veces más entrenamiento", "sin cambio (0,6869 → 0,6812)"],
                 ["Falta el algoritmo adecuado", "16 familias, conjuntos, GPU, 21 configuraciones", "todas entre 0,657 y 0,727"],
                 ["Falta ajustar la representación", "ajuste fino de extremo a extremo con 63 926 filas", "0,6860 · y ρ = −0,014 en organizaciones nuevas"],
                 ["Faltaba la etapa de categorización", "ablación del bloque de categoría", "aporta 0,008 · el vector ya la codifica"]],
                anchos=[3.3, 4.3, 4.5], tam=13, alto_fila=Inches(0.52))
    y += Inches(0.25)
    y = P.veredicto(s, y, "conclusión",
                    "El texto de una tarea, por sí solo, no contiene la "
                    "información necesaria para fijar sus horas. Es un "
                    "resultado negativo, y es defendible porque está medido "
                    "por cuatro vías independientes.", tono="aviso")
    P.nota(s, "El ajuste fino con 63 926 filas pierde TODO el orden en "
              "organizaciones nuevas: aprendió el vocabulario de los proyectos "
              "de entrenamiento, no una noción transferible de tamaño.")

    # ------------------------------------------------------------------ 7
    s, y = P.lamina(prs, "Resultado 3 · lo que sí funciona es la escala del proyecto",
                    "La mediana del proyecto, una constante que no lee texto, "
                    "gana al modelo de texto dentro del corpus público.")
    y = P.tabla(s, y,
                ["_Método", "MdMRE", "PRED25", "Spearman"],
                [["Mediana del proyecto (no lee texto)", "0,5455", "0,265", "0,508"],
                 ["Modelo de texto, valor absoluto", "0,6765", "0,176", "0,367"],
                 ["Modelo sobre la RAZÓN × mediana del proyecto", "0,5488", "0,247", "0,571"]],
                anchos=[6.4, 1.6, 1.6, 1.6], tam=14.5, alto_fila=Inches(0.46),
                resaltar={2: P.SUAVE})
    y += Inches(0.3)
    y = P.veredicto(s, y, "el cambio de planteamiento",
                    "En vez de predecir las horas en el vacío, predecir cuánto "
                    "se aparta esta tarea de lo normal en su proyecto. Al texto "
                    "solo se le pide «más grande o más pequeña de lo habitual "
                    "aquí», que es la pregunta que sabe responder: el orden sube "
                    "de 0,367 a 0,571.")
    P.nota(s, "Consecuencia de producto: el cronómetro deja de ser un accesorio "
              "y pasa a ser un insumo del predictor.")

    # ------------------------------------------------------------------ 8
    s, y = P.lamina(prs, "Resultado 4 · el sistema mejora con el uso",
                    "150 organizaciones que ningún modelo vio jamás. Cada tarea "
                    "se puntúa solo con las que terminaron antes en su proyecto.")
    P.figura(s, FIG / "despliegue.png", P.MARGEN, y, P.UTIL)
    y2 = y + Inches(3.2)
    y2 = P.tabla(s, y2,
                 ["_Tareas ya cerradas en ese proyecto", "0", "5-9", "20-49", "50 o más"],
                 [["MdMRE", "0,860", "0,714", "0,640", "0,566"],
                  ["PRED(25)", "0,120", "0,216", "0,248", "0,239"],
                  ["Banda del 80 %", "×91", "×28", "×15", "×11"]],
                 anchos=[4.6, 1.7, 1.7, 1.7, 1.9], tam=14,
                 alto_fila=Inches(0.42))

    # ------------------------------------------------------------------ 9
    P.seccion(prs, "2", "¿Cumple los criterios de aceptación?",
              "La pregunta que decide si el componente es defendible.")

    # ------------------------------------------------------------------ 10
    s, y = P.lamina(prs, "No. Cero de doce regímenes.",
                    "Los cuatro criterios, con el mismo rasero para el modelo, "
                    "la línea base y el experto humano.")
    y = P.tabla(s, y,
                ["_Criterio", "_Umbral", "_Mejor cifra obtenida", "_"],
                [["MAE frente a la duración media", "≤ 30 %", "71,6 %", "NO"],
                 ["RMSE frente al MAE", "≤ 2×", "2,48×", "NO"],
                 ["MdAPE", "≤ 20 %", "25,0 % (experto humano)", "NO"],
                 ["PRED(25)", "≥ 75 %", "50,9 % (experto) · 24,8 % (modelo)", "NO"]],
                anchos=[4.0, 1.6, 4.6, 1.0], tam=14, alto_fila=Inches(0.46),
                resaltar={3: P.AVISO_SUAVE})
    y += Inches(0.28)
    y = P.veredicto(s, y, "el criterio 2 no es alcanzable en este problema",
                    "«RMSE no más del doble del MAE» falla en las doce filas, "
                    "también para el experto humano y para una constante. El "
                    "esfuerzo es log-normal: el RMSE queda dominado por unas "
                    "pocas tareas grandes por mucho que la mediana esté bien. "
                    "El criterio está escrito para errores simétricos.",
                    tono="ambar")

    # ------------------------------------------------------------------ 10b
    s, y = P.lamina(prs, "Pero el propio campo abandonó esos criterios",
                    "Kitchenham (2001) y Foss (2003) demostraron que MMRE "
                    "favorece a los modelos que subestiman. Shepperd & MacDonell "
                    "(2012) lo sustituyen por una pregunta que ningún umbral "
                    "esquiva: ¿esto vale más que adivinar al azar?")
    y = P.tabla(s, y,
                ["_Escenario", "_Quién estima", "SA", "Tamaño del efecto"],
                [["SiP con estimación", "experto humano", "0,485", "−29,2"],
                 ["Jira público con estimación", "humano × modelo combinados", "0,477", "−27,0"],
                 ["Dentro del corpus público", "modelo (razón × mediana)", "0,465", "−17,8"],
                 ["150 organizaciones nuevas · régimen", "modelo + calibración", "0,443", "−25,2"],
                 ["Prueba fija del Dataset 1", "modelo v4", "0,422", "−18,2"],
                 ["150 organizaciones nuevas · arranque en frío", "modelo + calibración", "0,387", "−5,6"],
                 ["Jira público con estimación", "experto humano", "−0,077", "+4,4"]],
                anchos=[4.6, 3.6, 1.3, 2.2], tam=13, alto_fila=Inches(0.42),
                resaltar={6: P.AVISO_SUAVE})
    y += Inches(0.22)
    P.veredicto(s, y, "diez de once escenarios superan al azar con holgura",
                "El modelo elimina entre el 39 % y el 46 % del error de adivinar, "
                "en todos los regímenes y con tamaños de efecto enormes. El único "
                "que NO supera al azar es la estimación humana en Jira público: "
                "SA = −0,077.")
    P.nota(s, "SA = 1 − MAR/MAR_azar. El azar predice para cada tarea el valor "
              "de otra tarea del mismo bloque, promediado sobre 1 000 corridas.")

    # ------------------------------------------------------------------ 11
    s, y = P.lamina(prs, "Tampoco lo cumple nadie más",
                    "El umbral del 85 % de acierto no lo alcanza ningún método "
                    "publicado, ni el juicio de un desarrollador experto.")
    P.figura(s, FIG / "criterios.png", P.MARGEN, y, Inches(8.1))
    P.tabla(s, y + Inches(0.15),
            ["_Comparación", "MdMRE"],
            [["Deep-SE (LSTM)", "0,7133"],
             ["Nuestro modelo", "0,6993"],
             ["GPT2SP (GPT-2 ajustado)", "0,6804"]],
            anchos=[2.6, 1.1], ancho_total=Inches(3.7),
            x=P.MARGEN + Inches(8.4), tam=13, alto_fila=Inches(0.42),
            resaltar={2: P.SUAVE})
    P.veredicto(s, y + Inches(2.35), "sobre nuestros datos y particiones",
                "Reimplementados y evaluados igual. GPT2SP nos gana. No hemos "
                "superado el estado del arte.", tono="aviso",
                x=P.MARGEN + Inches(8.4), ancho=Inches(3.7))

    # ------------------------------------------------------------------ 12
    P.seccion(prs, "3", "Opciones de uso y de mejora",
              "Qué se puede desplegar hoy y qué caminos quedan abiertos.")

    # ------------------------------------------------------------------ 13
    s, y = P.lamina(prs, "Lo que el artefacto sí entrega hoy",
                    "Ninguna de estas funciones necesita alcanzar PRED(25) ≥ 75 %.")
    y = P.tabla(s, y,
                ["_Función", "_Medida", "_Estado"],
                [["Aviso de tarea grande (> 8 h)", "AUC 0,816 en organizaciones nuevas", "listo"],
                 ["Aviso de tarea muy grande (> 16 h)", "AUC 0,827", "listo"],
                 ["Ordenar tareas por tamaño relativo", "Spearman 0,564", "listo"],
                 ["Intervalo con cobertura declarada", "80 % real medido, banda ×11 en régimen", "listo"],
                 ["Estimación puntual en régimen (50+ tareas)", "MdMRE 0,566 · PRED(25) 0,239", "con reservas"],
                 ["Estimación puntual en arranque en frío", "MdMRE 0,860 · banda ×91", "no usable sola"]],
                anchos=[4.4, 5.0, 1.6], tam=13.5, alto_fila=Inches(0.44),
                resaltar={0: P.SUAVE, 1: P.SUAVE, 5: P.AVISO_SUAVE})
    y += Inches(0.25)
    P.veredicto(s, y, "cómo presentarlo en el producto",
                "Mostrar un rango y un aviso de riesgo, no un número. Un «entre "
                "2 y 6 horas, posible tarea grande» es honesto con los datos; un "
                "«3,4 horas» no lo es.")

    # ------------------------------------------------------------------ 14
    s, y = P.lamina(prs, "Alternativas de mejora, ordenadas por lo que costarían",
                    "Cada una ataca una causa distinta del techo.")
    y = P.tabla(s, y,
                ["_Vía", "_Qué haría falta", "_Evidencia a favor", "Esfuerzo"],
                [["Combinar con el LLM que ya genera las tareas", "una corrección multiplicativa aprendida sobre ~50 tareas", "media geométrica LLM×modelo: 0,5746 frente a 0,60 y 0,82 por separado", "bajo"],
                 ["Corregir la estimación de la persona en vez de sustituirla", "que el formulario pida una estimación", "en SiP el experto da 0,2500; corregido por desarrollador, 0,2500 → mejor", "bajo"],
                 ["Calibración en línea por proyecto", "ya está: usar las tareas cerradas", "el error baja de 0,860 a 0,566 con 50 tareas", "hecho"],
                 ["Añadir señal fuera del texto", "tamaño del cambio, ficheros tocados, historial del desarrollador", "el texto solo explica hasta 0,68; la escala del proyecto aporta más", "medio"],
                 ["Recoger datos propios de PBE", "el cronómetro del producto en clientes reales", "misma cultura de registro que el objetivo, sin desplazamiento", "alto · continuo"]],
                anchos=[3.4, 3.4, 4.4, 1.1], tam=12, alto_fila=Inches(0.62))
    P.nota(s, "La primera vía es la más barata y la que mejor encaja: PBE ya "
              "usa un LLM para generar las tareas, así que la estimación del "
              "LLM está disponible sin coste adicional.")

    # ------------------------------------------------------------------ 15
    s, y = P.lamina(prs, "Recomendación")
    y = P.veredicto(s, y, "qué desplegar",
                    "Entrenar con el Dataset 2, predecir la razón contra la "
                    "mediana del proyecto, calibrar en línea con las tareas "
                    "cerradas, y presentar rango más aviso de riesgo en lugar "
                    "de un número.")
    y += Inches(0.25)
    y = P.vinetas(s, y, [
        ("TESIS", "La aportación defendible no es un estimador que acierte al "
                  "85 %, sino la evidencia de que ese objetivo no es alcanzable "
                  "con la información disponible —ni con seis veces más datos, "
                  "ni ajustando el codificador entero, ni con un experto."),
        ("PRODUCTO", "Lo que sí funciona está medido: orden útil, aviso de "
                     "tareas grandes con AUC 0,82 en organizaciones nuevas, "
                     "intervalo con cobertura declarada, y una curva de mejora "
                     "con cada tarea que el equipo cierra."),
        ("SIGUIENTE", "La vía más barata es combinar con el LLM que ya genera "
                      "las tareas: una sola corrección multiplicativa reduce su "
                      "error un 44 %, y la combinación gana a ambos por separado."),
    ], tam=15, salto=Inches(0.5))

    P.numerar(prs)
    Path(SALIDA).parent.mkdir(parents=True, exist_ok=True)
    prs.save(SALIDA)
    print(f"escrito {SALIDA}  ({len(prs.slides._sldIdLst)} diapositivas)")


if __name__ == "__main__":
    main()
