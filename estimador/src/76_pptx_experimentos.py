"""Deck 3: the three new experiments against the baseline we already had.

Organised around the names of the algorithms, as requested, and grouped by test
block: the three experiments do not share one test set, so the deck says which
block each number comes from instead of putting incomparable figures side by
side.
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")
from pptx.util import Inches  # noqa: E402

import pptx_comun as P  # noqa: E402

FIG = Path("reports/figuras_pptx")
SALIDA = "entrega/presentaciones/3_experimentos_nuevos.pptx"


def main():
    prs = P.nueva()

    # ------------------------------------------------------------------ 1
    P.portada(prs, "Tres experimentos nuevos\nfrente a la línea de partida",
              "LLMs locales, la estimación del usuario y un dataset solo de flags,\n"
              "comparados con los algoritmos que ya habíamos medido.",
              "Resultados experimentales · 17 de septiembre de 2026")

    # ------------------------------------------------------------------ 2
    s, y = P.lamina(prs, "De dónde partíamos",
                    "19 configuraciones sobre la prueba fija de 4.225 tareas. "
                    "Cambiar de algoritmo mueve el error unas seis centésimas.")
    y = P.tabla(s, y,
                ["_Algoritmo", "MdMRE", "PRED25", "Spearman", "Gana a la base"],
                [["hgb · HistGradientBoosting", "0,6586", "0,172", "0,415", "14/24"],
                 ["rf_profundo · Random Forest (hojas ≥ 5)", "0,6587", "0,174", "0,411", "13/24"],
                 ["voting · conjunto por votación", "0,6601", "0,176", "0,448", "15/24"],
                 ["xgboost_gpu · XGBoost en GPU", "0,6644", "0,184", "0,415", "17/24"],
                 ["stacking · conjunto apilado", "0,6647", "0,170", "0,445", "17/24"],
                 ["svr_lineal · SVR lineal", "0,6691", "0,178", "0,416", "18/24"],
                 ["ridge · regresión ridge", "0,6721", "0,165", "0,425", "17/24"],
                 ["knn · vecinos más cercanos", "0,6844", "0,188", "0,384", "15/24"],
                 ["arbol · árbol de decisión", "0,7151", "0,151", "0,273", "11/24"],
                 ["línea base: mediana del proyecto", "0,7273", "0,164", "0,089", "0/24"]],
                anchos=[5.4, 1.5, 1.4, 1.5, 2.1], tam=13, alto_fila=Inches(0.36),
                resaltar={0: P.SUAVE, 9: P.AVISO_SUAVE})
    y += Inches(0.2)
    P.veredicto(s, y, "y la literatura, reimplementada sobre los mismos datos",
                "GPT2SP (GPT-2 ajustado) 0,6804 · Deep-SE (LSTM) 0,7133 · nuestro ridge "
                "sobre SBERT congelado 0,6993. GPT2SP nos gana: no hemos superado el "
                "estado del arte.")

    # ------------------------------------------------------------------ 3
    s, y = P.lamina(prs, "Antes de comparar: cuatro exámenes distintos",
                    "Los tres experimentos no comparten bloque de prueba. Las cifras "
                    "solo son comparables dentro de cada bloque.")
    y = P.tabla(s, y,
                ["_Bloque de prueba", "Tareas", "_Quién lo usa"],
                [["prueba fija del Dataset 1", "4.225", "barrido inicial, literatura y el experimento de flags"],
                 ["muestra estratificada", "601", "experimento de LLMs (mezcla prueba fija y externo)"],
                 ["prueba de SiP", "1.885", "estimación del usuario · escenario misma empresa"],
                 ["bloque externo (organizaciones nuevas)", "13.255", "flags; 1.163 de ellas con estimación del usuario"]],
                anchos=[4.6, 1.4, 6.1], tam=13.5, alto_fila=Inches(0.44))
    y += Inches(0.25)
    P.veredicto(s, y, "cómo se lee esta presentación",
                "Cada lámina dice sobre qué bloque está midiendo y contra qué referencia. "
                "Los veredictos vienen de pruebas pareadas sobre las mismas tareas, con "
                "intervalo de confianza del 95 %.", tono="ambar")

    # ------------------------------------------------------------------ 4
    P.seccion(prs, "1", "LLMs locales frente al modelo",
              "llama3.1:8b · qwen2.5:7b · gemma2:9b · mistral:7b\n"
              "10.528 respuestas, todas válidas, en la GPU de casa.")

    # ------------------------------------------------------------------ 5
    s, y = P.lamina(prs, "Cómo se probó",
                    "Tres condiciones igualadas en información, para que el LLM y el "
                    "modelo vean lo mismo.")
    y = P.vinetas(s, y, [
        ("A", "Solo el texto de la tarea y su categoría. Es el arranque en frío."),
        ("B", "Además, 10 tareas resueltas fijas con sus horas reales."),
        ("C", "Además, las últimas 10 tareas cerradas de ESE proyecto. Nuestro modelo "
              "escala por la mediana de exactamente esas 10."),
        ("CONTROL", "Claves de incidencia enmascaradas, temperatura 0, y una sonda que "
                    "pregunta al LLM si reconoce la incidencia."),
    ], tam=15.5, salto=Inches(0.5))
    y += Inches(0.15)
    P.veredicto(s, y, "tres versiones de cada LLM",
                "crudo (lo que responde), calibrado (con un factor multiplicativo "
                "aprendido en 130 tareas aparte) y combinado (media geométrica con "
                "nuestro modelo). Las respuestas no interpretables se sustituyen por la "
                "línea base, no se descartan.")

    # ------------------------------------------------------------------ 6
    s, y = P.lamina(prs, "Resultado · los tres escenarios")
    P.figura_ajustada(s, FIG / "exp_llms.png", y, Inches(3.5))
    y2 = y + Inches(3.6)
    P.tabla(s, y2,
            ["_Condición", "_Nuestro modelo", "_Mejor LLM solo", "_Veredicto", "_Mejor combinación"],
            [["A · solo texto", "hgb 0,682", "gemma2 calibrado 0,743", "nuestro modelo mejor", "0,687 · empate"],
             ["B · 10 ejemplos", "hgb 0,682", "qwen2.5 calibrado 0,699", "empate", "0,689 · empate"],
             ["C · historial", "razón × mediana 0,637", "qwen2.5 crudo 0,597", "empate", "gemma2 × modelo 0,569 · LLM mejor"]],
            anchos=[2.4, 2.4, 3.0, 2.5, 3.8], tam=12.5, alto_fila=Inches(0.42),
            resaltar={2: P.SUAVE})

    # ------------------------------------------------------------------ 7
    s, y = P.lamina(prs, "La única victoria de un LLM, y por qué",
                    "Con historial del proyecto, gemma2:9b combinado con nuestro modelo "
                    "sí mejora, y con respaldo estadístico.")
    y = P.tabla(s, y,
                ["_Sobre las mismas 590 tareas con historial", "MdMRE", "PRED25", "Spearman", "SA"],
                [["mediana del historial (no lee texto)", "0,659", "0,236", "0,511", "0,436"],
                 ["nuestro modelo · razón × mediana", "0,637", "0,203", "0,554", "0,448"],
                 ["qwen2.5:7b crudo", "0,597", "0,210", "0,575", "0,490"],
                 ["gemma2:9b × nuestro modelo", "0,569", "0,234", "0,615", "0,494"]],
                anchos=[5.8, 1.5, 1.5, 1.6, 1.4], tam=14, alto_fila=Inches(0.42),
                resaltar={3: P.SUAVE})
    y += Inches(0.25)
    y = P.veredicto(s, y, "−10,7 % de error, IC95 [−0,106, −0,027]",
                    "Funciona porque se equivocan de forma distinta. Pero cuidado: los "
                    "LLMs se anclan en lo que tienen delante —qwen2.5 repite exactamente "
                    "una hora del historial en el 41 % de sus respuestas, llama3.1 copia "
                    "la etiqueta de un ejemplo en el 47 %.")
    P.nota(s, "Sin calibrar, los LLMs sobreestiman entre 2,4 y 6 veces: piensan en "
              "jornadas (8 h, 4 h, 16 h) cuando la mediana real es 1,5 h.")

    # ------------------------------------------------------------------ 8
    s, y = P.lamina(prs, "Coste, velocidad y contaminación",
                    "Todo local y gratis: sin claves, sin enviar datos de clientes a "
                    "ningún proveedor.")
    y = P.tabla(s, y,
                ["_Modelo", "Tamaño", "s / tarea", "Válidas", "_¿Reconoce incidencias públicas?"],
                [["qwen2.5:7b", "4,7 GB", "0,31-0,59", "100 %", "0 claves exactas de 450"],
                 ["mistral:7b", "4,1 GB", "0,32-0,62", "100 %", "0 de 450"],
                 ["llama3.1:8b", "4,9 GB", "0,63", "100 %", "0 de 450 · dice reconocer el 40 %"],
                 ["gemma2:9b", "5,4 GB", "0,84-1,34", "100 %", "0 de 450"]],
                anchos=[2.4, 1.5, 1.8, 1.4, 5.0], tam=13.5, alto_fila=Inches(0.42))
    y += Inches(0.25)
    P.veredicto(s, y, "sin rastro de memorización",
                "Ninguno acierta una sola clave de incidencia. llama3.1 afirma reconocer "
                "el 40 % de las tareas y no acierta ninguna: inventa el reconocimiento. "
                "La comparación no está contaminada por datos memorizados.")

    # ------------------------------------------------------------------ 9
    P.seccion(prs, "2", "La estimación del usuario",
              "En vez de predecir en frío, el modelo recibe cuántas horas cree el "
              "desarrollador que tardará.\nEs el cambio más grande que hemos medido.")

    # ------------------------------------------------------------------ 10
    s, y = P.lamina(prs, "Ocho brazos, todos sobre las mismas tareas")
    y = P.tabla(s, y,
                ["_Brazo", "_Qué recibe el modelo", "_Algoritmo"],
                [["R0 · en frío", "texto + categoría + numéricas, sin estimación", "HistGradientBoosting"],
                 ["H0 · usuario", "la estimación tal cual", "ninguno"],
                 ["H1 · factor global", "estimación × factor aprendido en entrenamiento", "mediana de razones"],
                 ["H2 · por desarrollador", "estimación × factor del desarrollador, contraído", "Bayes empírico"],
                 ["M1 · sin texto", "estimación + categoría", "HistGradientBoosting"],
                 ["M2 · con texto", "texto + estimación", "HistGradientBoosting"],
                 ["M3 · corrector", "texto + estimación, aprende log(real / estimación)", "HistGradientBoosting"],
                 ["G1 · combinado", "media geométrica de estimación y modelo en frío", "—"]],
                anchos=[2.6, 6.2, 3.3], tam=13, alto_fila=Inches(0.40),
                resaltar={4: P.SUAVE})
    y += Inches(0.2)
    P.veredicto(s, y, "elegido en validación: M1",
                "Solo hay estimaciones humanas reales en SiP y en el bloque público de "
                "Jira: 8.679 tareas de entrenamiento y 1.556 de validación. TAWOS solo "
                "tiene puntos de historia, que ya sabíamos que no predicen horas.")

    # ------------------------------------------------------------------ 11
    s, y = P.lamina(prs, "Resultado · el salto más grande de todo el trabajo")
    P.figura_ajustada(s, FIG / "exp_estimacion.png", y, Inches(3.4))
    y2 = y + Inches(3.5)
    P.tabla(s, y2,
            ["_Escenario", "_En frío (R0)", "_Usuario (H0)", "_M1 con estimación", "_Cambio en MdMRE"],
            [["SiP · misma empresa (1.885)", "0,713", "0,250", "0,246", "−65,6 %"],
             ["Jira · cliente nuevo (1.163)", "0,725", "0,400", "0,440", "−39,3 %"]],
            anchos=[3.4, 2.1, 2.1, 2.4, 2.3], tam=14, alto_fila=Inches(0.44),
            resaltar={0: P.SUAVE, 1: P.SUAVE})

    # ------------------------------------------------------------------ 12
    s, y = P.lamina(prs, "Los matices, que son importantes")
    y = P.vinetas(s, y, [
        ("EMPATE", "En la mediana, el modelo no gana al usuario: el intervalo incluye el "
                   "cero en los dos escenarios."),
        ("PERO", "En error absoluto sí. En cliente nuevo el usuario tiene MAE 12,55 h y "
                 "SA −0,444 —en horas totales, peor que adivinar— por unas pocas "
                 "sobreestimaciones enormes. M1 lo baja a 3,92 h y SA 0,549."),
        ("SIN EMPATES", "En las tareas donde lo real NO coincide con lo estimado, M1 gana "
                        "al usuario con significación: 0,480 frente a 0,500 en SiP y "
                        "0,602 frente a 0,833 en cliente nuevo."),
        ("OJO", "Un tercio de las tareas registra exactamente lo estimado (35 % en SiP, "
                "29 % en Jira). Eso infla todo lo que se parezca a la estimación."),
    ], tam=15, salto=Inches(0.48))
    y += Inches(0.1)
    P.veredicto(s, y, "sigue sin cumplirse el criterio de aceptación",
                "El mejor PRED(25) es 0,508: la mitad del umbral de 0,75 de Conte. Y el "
                "texto no aporta nada sobre la estimación —M1, que no lo usa, iguala o "
                "supera a M2 y M3.", tono="aviso")

    # ------------------------------------------------------------------ 13
    P.seccion(prs, "3", "Dataset solo de flags",
              "¿Basta con convertir el vector en flags de categoría?\n"
              "89.851 tareas × 106 flags en 1,3 MB, frente a 276 MB de vectores.")

    # ------------------------------------------------------------------ 14
    s, y = P.lamina(prs, "Resultado · los flags no sustituyen al vector",
                    "10 algoritmos × 6 combinaciones de familias, entrenados solo con "
                    "flags y horas. Ganador en validación: SVR lineal sobre los grupos.")
    y = P.tabla(s, y,
                ["_Modelo", "_Prueba fija · MdMRE · err. log · ρ", "_Organizaciones nuevas"],
                [["modelo actual (vector + categoría + fuente) · HGB", "0,685 · 0,828 · 0,39", "0,703 · 0,817 · 0,17"],
                 ["vector 768 solo · HGB", "0,695 · 0,924 · 0,29", "0,716 · 0,831 · 0,14"],
                 ["vector 768 solo · SVR lineal", "0,718 · 0,961 · 0,25", "0,705 · 0,845 · 0,10"],
                 ["flags de grupos · SVR lineal", "0,729 · 1,057 · 0,09", "0,715 · 0,810 · 0,07"],
                 ["flags todos · SVR lineal", "0,738 · 1,046 · 0,13", "0,717 · 0,799 · 0,09"],
                 ["constante (mediana global)", "0,750 · 1,099 · —", "0,750 · 0,916 · —"]],
                anchos=[5.6, 3.3, 3.2], tam=13, alto_fila=Inches(0.42),
                resaltar={0: P.SUAVE, 5: P.AVISO_SUAVE})
    y += Inches(0.22)
    y = P.veredicto(s, y, "con el mismo algoritmo, los flags pierden",
                    "SVR contra SVR: en la prueba fija los flags son peores en error "
                    "logarítmico (IC95 [+0,052, +0,142]) y ordenan mucho menos (ρ 0,09 "
                    "frente a 0,25). En organizaciones nuevas ganan en error logarítmico "
                    "pero pierden en MdMRE: sus predicciones se apiñan cerca de la mediana.")
    P.nota(s, "La categoría real (la etiqueta original de la fuente) queda al nivel de una "
              "constante en validación: 0,915 frente a 0,916 de error logarítmico. La "
              "categoría casi no informa de las horas.")

    # ------------------------------------------------------------------ 15
    s, y = P.lamina(prs, "Todos los algoritmos, un solo examen",
                    "Los flags son la única familia nueva medible sobre la prueba fija, "
                    "así que se pueden poner al lado de los algoritmos iniciales.")
    P.figura_ajustada(s, FIG / "exp_algoritmos.png", y, P.ALTO - Inches(0.9) - y)

    # ------------------------------------------------------------------ 16
    s, y = P.lamina(prs, "Qué mueve de verdad la aguja",
                    "Cada barra compara un cambio con su propia referencia, en su propio "
                    "bloque de prueba.")
    P.figura_ajustada(s, FIG / "exp_palancas.png", y, P.ALTO - Inches(0.9) - y)

    # ------------------------------------------------------------------ 17
    s, y = P.lamina(prs, "Qué cambia en las conclusiones")
    y = P.tabla(s, y,
                ["_Lo que pensábamos", "_Lo que dicen los tres experimentos"],
                [["Quizá un LLM estime mejor que el modelo",
                  "No en frío: nuestro hgb gana a los cuatro LLMs. Sí combinándolos cuando hay historial"],
                 ["El cuello de botella es el algoritmo",
                  "19 algoritmos caben en seis centésimas. La información disponible pesa mucho más"],
                 ["El vector ya clasifica, con la categoría bastará",
                  "La categoría casi no informa de las horas; comprimir a flags pierde señal y organización"],
                 ["Predecir en frío es el objetivo",
                  "Pedir la estimación al usuario reduce el error un 39-66 %: ocho veces más que cambiar de algoritmo"]],
                anchos=[5.4, 6.7], tam=13.5, alto_fila=Inches(0.58),
                resaltar={3: P.SUAVE})
    y += Inches(0.25)
    P.veredicto(s, y, "lo que no cambia",
                "Ninguna configuración cumple los criterios de aceptación, y el experto "
                "humano tampoco. El mejor PRED(25) de todo el trabajo sigue siendo 0,508, "
                "con la estimación del usuario dentro.", tono="aviso")

    # ------------------------------------------------------------------ 18
    s, y = P.lamina(prs, "Recomendación para el producto")
    y = P.tabla(s, y,
                ["_Situación en Agix", "_Qué usar", "_Error esperado"],
                [["Tarea recién generada, proyecto nuevo", "modelo en frío (HistGradientBoosting sobre el vector)", "MdMRE ~0,72"],
                 ["Proyecto con tareas cerradas", "razón × mediana del proyecto, y combinar con el LLM que ya genera las tareas", "~0,57"],
                 ["El usuario escribe su estimación", "M1: estimación + categoría (HistGradientBoosting)", "0,25 misma empresa · 0,44 cliente nuevo"]],
                anchos=[4.2, 5.6, 3.1], tam=13.5, alto_fila=Inches(0.56),
                resaltar={2: P.SUAVE})
    y += Inches(0.25)
    P.vinetas(s, y, [
        "Mostrar siempre rango y aviso de tarea grande, no un número solo: el aviso "
        "mantiene AUC 0,82 en organizaciones nuevas.",
        "Los flags (tipo y área) no sirven para predecir, pero sí para explicar y filtrar "
        "en la interfaz.",
        "Para pedir la estimación hace falta un campo de horas en el formulario de tareas, "
        "rellenado antes de empezar el trabajo.",
    ], tam=15, salto=Inches(0.46))

    P.numerar(prs)
    Path(SALIDA).parent.mkdir(parents=True, exist_ok=True)
    prs.save(SALIDA)
    print(f"escrito {SALIDA}  ({len(prs.slides._sldIdLst)} diapositivas)")


if __name__ == "__main__":
    main()
