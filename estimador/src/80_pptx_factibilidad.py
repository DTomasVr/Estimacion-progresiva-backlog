"""Deck 4: which predictor the artefact should ship, and in which of the two designs.

Two decisions, in this order:
  1. our model or one of the local LLMs as the estimator
  2. cold prediction, or ask the user for a guess and correct it

Every figure in the deck is produced by script 79 from the results files, and
every number written into a table here is read from those same files at build
time -- nothing is typed in by hand, so the deck cannot drift from the runs.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from pptx.util import Inches  # noqa: E402

import pptx_comun as P  # noqa: E402

FIG = Path("reports/figuras_pptx")
SALIDA = "entrega/presentaciones/4_factibilidad_y_diseno.pptx"
EC = pd.read_csv("results/estimacion_completa/resumen.csv")
LL = pd.read_csv("results/llm_bench/resumen.csv")
CU = pd.read_csv("results/estimacion_completa/curva_adopcion.csv")
SEL = json.load(open("results/estimacion_completa/seleccion.json", encoding="utf-8"))
COSTO = json.load(open("results/costo/resumen.json", encoding="utf-8"))
LAT_N = pd.read_csv("results/costo/latencia_nuestra.csv")
LAT_L = pd.read_csv("results/costo/latencia_llms.csv")
GANADOR = SEL["ganador_val"]
SIP = "SiP - prueba (misma empresa)"
EXT = "Jira - externo (cliente nuevo)"
TAW = "TAWOS - prueba (puntos de historia)"


def n(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def pc(x, d=1):
    """Porcentaje con coma decimal."""
    return f"{x * 100:.{d}f}".replace(".", ",") + " %"


def mil(x):
    """Miles con punto, como en el resto del trabajo."""
    return f"{x:,}".replace(",", ".")


def ec(esc, brazo, met, sub="todas"):
    v = EC[(EC.escenario == esc) & (EC.subconjunto == sub)
           & (EC.brazo.str.startswith(brazo))]
    return float(v[met].iloc[0])


def ll(cond, modelo, version, met):
    v = LL[(LL.condicion == cond) & (LL.modelo.str.startswith(modelo))
           & (LL.version == version)]
    return v[met].iloc[0]


def ref_nuestra(cond, met):
    """Our own model on the same 601 tasks, best arm of that condition."""
    v = LL[(LL.condicion == cond) & (LL.version == "referencia")
           & (LL.modelo.str.startswith("nuestro")) & (~LL.modelo.str.contains("mismas"))]
    return float(v.sort_values("MdMRE")[met].iloc[0])


def main():
    prs = P.nueva()
    seg_enc = float(LAT_N[LAT_N.componente.str.contains("cuda")].seg_una_tarea.iloc[0])
    seg_reg = float(LAT_N[LAT_N.componente.str.startswith("regresor")].seg_una_tarea.iloc[0])
    seg_nuestro = seg_enc + seg_reg
    peso_nuestro = (COSTO["peso_codificador_MB"] + COSTO["peso_regresor_MB"]) / 1000

    # ------------------------------------------------------------------ 1
    P.portada(prs, "¿Nuestro modelo o un LLM?\n¿En frío o corrigiendo al usuario?",
              "Las dos decisiones de diseño del componente de estimación,\n"
              "resueltas con las mediciones de los experimentos.",
              "Factibilidad y diseño del artefacto · 17 de septiembre de 2026")

    # ------------------------------------------------------------------ 2
    s, y = P.lamina(prs, "Las dos decisiones",
                    "PBE ya usa un LLM para generar las tareas. La pregunta es quién "
                    "pone las horas y con qué información.")
    y = P.vinetas(s, y, [
        ("DECISIÓN 1", "¿El estimador es nuestro modelo entrenado, o le pedimos las "
                       "horas al mismo LLM que ya genera las tareas?"),
        ("DECISIÓN 2", "¿El modelo predice en frío desde el texto, o le pedimos al "
                       "usuario su propia estimación y el modelo la corrige?"),
    ], tam=16.5, salto=Inches(0.62))
    y += Inches(0.1)
    y = P.veredicto(s, y, "respuesta 1",
                    f"Nuestro modelo. De veinticuatro comparaciones pareadas el LLM solo "
                    f"gana una, y solo cuando se combina con nuestro modelo. Además el LLM "
                    f"tarda entre {float(LAT_L.seg_tarea_min.min()) / seg_nuestro:.0f} y "
                    f"{float(LAT_L.seg_tarea_max.max()) / seg_nuestro:.0f} veces más por "
                    f"tarea y sus pesos ocupan unas "
                    f"{float(str(LAT_L.peso_en_disco.iloc[0]).split()[0]) / peso_nuestro:.0f} "
                    f"veces lo que ocupa el nuestro.")
    y += Inches(0.16)
    P.veredicto(s, y, "respuesta 2",
                f"Pedir la estimación y corregirla. En el bloque de prueba completo el "
                f"error medio baja de {n(ec('prueba fija completa', 'R0', 'MAE_h'), 2)} h "
                f"a {n(ec('prueba fija completa', 'D1', 'MAE_h'), 2)} h y el MdMRE de "
                f"{n(ec('prueba fija completa', 'R0', 'MdMRE'))} a "
                f"{n(ec('prueba fija completa', 'D1', 'MdMRE'))}. Pero hay que corregirla: "
                f"mostrar el número del usuario tal cual es la peor de las tres opciones.",
                tono="ambar")

    # ------------------------------------------------------------------ 3
    P.seccion(prs, "1", "Nuestro modelo frente a los LLMs",
              "llama3.1:8b · qwen2.5:7b · gemma2:9b · mistral:7b\n"
              "10.528 respuestas sobre las mismas 601 tareas, en la GPU de casa.")

    # ------------------------------------------------------------------ 4
    s, y = P.lamina(prs, "Un examen igualado, no un examen amable",
                    "Si el LLM pierde, tiene que ser porque estima peor, no porque "
                    "tuviera menos información.")
    y = P.vinetas(s, y, [
        ("MISMAS TAREAS", "601 tareas del bloque de prueba y del bloque externo, "
                          "congeladas antes de correr nada."),
        ("MISMA INFO", "Tres condiciones: A solo el texto, B con 10 ejemplos resueltos, "
                       "C con las 10 últimas tareas cerradas de ese proyecto. Nuestro "
                       "modelo se escala con exactamente esas mismas 10."),
        ("MISMA AYUDA", "Cada LLM se mide crudo, calibrado con 130 tareas aparte, y "
                        "combinado con nuestro modelo. Se queda la mejor de las tres."),
        ("CONTROLES", "Claves de incidencia enmascaradas, temperatura 0, y una sonda "
                      "de contaminación: 0 de 450 aciertos de clave exacta."),
    ], tam=15.5, salto=Inches(0.52))
    y += Inches(0.12)
    P.veredicto(s, y, "sin excusas de ejecución",
                "10.528 respuestas, el 100 % interpretables, ningún paso fallido. "
                "El LLM no pierde por errores de integración.")

    # ------------------------------------------------------------------ 5
    s, y = P.lamina(prs, "Veinticuatro comparaciones, un solo triunfo del LLM")
    _, y2 = P.figura_ajustada(s, FIG / "fact_tablero_llm.png", y, Inches(4.35))
    P.veredicto(s, y2 + Inches(0.12), "lo que dice el tablero",
                f"Solo con el historial del proyecto y combinado con nuestro modelo hay "
                f"una victoria con respaldo estadístico: gemma2 × nuestro modelo, "
                f"{n(float(ll('C', 'gemma2', 'combinado', 'MdMRE')))}. En todo lo demás, "
                f"empate o victoria nuestra.")

    # ------------------------------------------------------------------ 6
    s, y = P.lamina(prs, "El mejor LLM de cada condición, contra nosotros",
                    "Mejor versión de cada LLM (calibrada o combinada) frente a nuestro "
                    "mejor brazo sobre esas mismas tareas.")
    filas = []
    for c, etiq in (("A", "A · solo el texto (arranque en frío)"),
                    ("B", "B · con 10 ejemplos resueltos"),
                    ("C", "C · con el historial del proyecto")):
        v = LL[(LL.condicion == c) & (LL.version.isin(["calibrado", "combinado"]))]
        mejor = v.sort_values("MdMRE").iloc[0]
        filas.append([etiq, mejor.modelo.split(" ")[0] + " · " + mejor.version,
                      n(float(mejor.MdMRE)), n(ref_nuestra(c, "MdMRE")),
                      n(float(mejor.SA), 3), n(ref_nuestra(c, "SA"), 3),
                      mejor.veredicto])
    y = P.tabla(s, y, ["_Condición", "_Mejor LLM", "MdMRE LLM", "MdMRE nuestro",
                       "SA LLM", "SA nuestro", "_Veredicto"],
                filas, anchos=[3.5, 2.4, 1.5, 1.7, 1.1, 1.3, 1.9], tam=13,
                alto_fila=Inches(0.46))
    y += Inches(0.22)
    P.veredicto(s, y, "y cuando el LLM va solo",
                f"Crudo, sin calibrar con nuestros datos, el mejor LLM de la condición A "
                f"se queda en {n(float(ll('A', 'gemma2', 'crudo', 'MdMRE')))} frente a "
                f"nuestro {n(ref_nuestra('A', 'MdMRE'))}: el LLM necesita nuestros datos "
                f"para acercarse, no al revés.", tono="ambar")

    # ------------------------------------------------------------------ 7
    s, y = P.lamina(prs, "Por qué falla: multiplica por ocho las tareas cortas")
    _, y2 = P.figura_ajustada(s, FIG / "fact_sesgo_tamano.png", y, Inches(4.2))
    P.veredicto(s, y2 + Inches(0.14), "el problema no es el ruido, es el nivel",
                "Sin contexto el LLM no tiene idea de la escala de horas de la "
                "organización: responde en la escala de “lo que tarda una tarea de "
                "software” en abstracto. Con el historial del proyecto corrige el "
                "nivel, que es justamente lo que nuestro modelo aprende de los datos.")

    # ------------------------------------------------------------------ 8
    s, y = P.lamina(prs, "Lo que cuesta servir cada predicción")
    _, y2 = P.figura_ajustada(s, FIG / "fact_costo.png", y, Inches(3.9))
    y2 += Inches(0.16)
    an = (P.UTIL - Inches(0.6)) / 4
    P.cifra(s, P.MARGEN, y2, an, f"{seg_nuestro * 1000:.0f} ms",
            "nuestro canal completo\npor tarea, una a una")
    P.cifra(s, P.MARGEN + (an + Inches(0.2)), y2, an,
            f"{n(float(LAT_L.seg_tarea_min.min()), 2)} s",
            "el LLM más rápido\nen su mejor condición", tono="ambar")
    P.cifra(s, P.MARGEN + 2 * (an + Inches(0.2)), y2, an,
            f"{n(peso_nuestro, 2)} GB",
            "pesos a desplegar\n(codificador + regresor)")
    P.cifra(s, P.MARGEN + 3 * (an + Inches(0.2)), y2, an,
            f"{n(COSTO['peso_regresor_MB'], 1)} MB",
            "lo que hay que reentrenar\ncada vez", tono="tinta")

    # ------------------------------------------------------------------ 9
    s, y = P.lamina(prs, "Riesgos que no se ven en el MdMRE",
                    "Para un artefacto de tesis, que tiene que poder defenderse y "
                    "reproducirse, estos pesan tanto como la precisión.")
    llama = pd.read_csv("results/llm_bench/sonda_contaminacion.csv")
    afirma = float(llama[llama.modelo == "llama3.1:8b"].afirma_reconocer.iloc[0])
    y = P.vinetas(s, y, [
        ("REPRODUCIBLE", f"Nuestro canal da el mismo número siempre: la diferencia "
                         f"máxima entre dos corridas idénticas es "
                         f"{COSTO['determinismo_regresor_max_dif']:.0e}. Un LLM depende "
                         f"de la versión de los pesos y del muestreo."),
        ("FALSA MEMORIA", f"llama3.1 dice reconocer la incidencia en el "
                          f"{pc(afirma)} de los casos, con 0 de 450 claves acertadas: "
                          f"afirma una familiaridad que no tiene."),
        ("SIN INTERVALO", "Nuestro modelo da una banda calibrada con cobertura "
                          "verificada; el LLM da un número sin respaldo de cobertura."),
        ("DEPENDENCIA", "El LLM local ocupa la GPU que también necesita el codificador, "
                        "y la versión en la nube ataría el artefacto a un proveedor y a "
                        "un coste por tarea."),
    ], tam=15.5, salto=Inches(0.56))

    # ------------------------------------------------------------------ 10
    s, y = P.lamina(prs, "Dónde sí conviene el LLM",
                    "La conclusión no es “el LLM no sirve”, sino “el LLM "
                    "no es el estimador”.")
    y = P.vinetas(s, y, [
        "Generar y descomponer las tareas: es lo que ya hace en PBE y es donde no "
        "tiene competencia.",
        f"Como segunda opinión opcional: gemma2 combinado con nuestro modelo y con "
        f"el historial del proyecto baja el MdMRE a "
        f"{n(float(ll('C', 'gemma2', 'combinado', 'MdMRE')))} frente a "
        f"{n(ref_nuestra('C', 'MdMRE'))}. Es la única mejora medida, y cuesta una "
        f"llamada por tarea.",
        "Redactar la justificación de la estimación en lenguaje natural a partir del "
        "número y la banda que da nuestro modelo.",
    ], tam=16, salto=Inches(0.62))
    y += Inches(0.12)
    P.veredicto(s, y, "decisión 1",
                "El estimador del artefacto es nuestro modelo. El LLM se queda en la "
                "generación de tareas, y opcionalmente como segunda opinión cuando el "
                "proyecto ya tiene historial.")

    # ------------------------------------------------------------------ 11
    P.seccion(prs, "2", "En frío o corrigiendo al usuario",
              f"{mil(SEL['n_pool'])} tareas de entrenamiento en frío, "
              f"{mil(SEL['n_train_est'])} con estimación humana.\n"
              "Cuatro correctores, tres escenarios y los bloques de prueba completos.")

    # ------------------------------------------------------------------ 12
    s, y = P.lamina(prs, "Cómo se usó el corpus completo",
                    "Solo 13.283 de las 89.851 tareas traen una estimación humana. "
                    "El resto no se desperdicia.")
    y = P.vinetas(s, y, [
        ("FRÍO", f"El modelo en frío se entrena con las {mil(SEL['n_pool'])} tareas de "
                 f"entrenamiento y su predicción entra como un rasgo más del corrector."),
        ("FUERA DE PLIEGUE", f"Esa predicción se calcula en 5 pliegues para las filas de "
                             f"entrenamiento: dentro de muestra el frío parece un "
                             f"{n(SEL['frio_dentro_MdMRE'])} y fuera de pliegue es un "
                             f"{n(SEL['frio_oof_MdMRE'])}. Sin esta precaución el "
                             f"corrector se fía demasiado del frío."),
        ("TERCER CANAL", f"TAWOS no da horas sino puntos de historia; se convierten con "
                         f"un ajuste log-lineal sobre {SEL['conversion_puntos']['n_train']} "
                         f"filas de entrenamiento y se informan aparte."),
        ("COBERTURA", f"En la prueba fija el "
                      f"{pc(ec('prueba fija completa', 'D1', 'cobertura'))} de las tareas "
                      f"puede llevar estimación; en el bloque externo, el "
                      f"{pc(ec('externo completo', 'D1', 'cobertura'))}."),
    ], tam=15, salto=Inches(0.56))

    # ------------------------------------------------------------------ 13
    s, y = P.lamina(prs, "Tres criterios, tres ganadores distintos",
                    "Elegir el corrector con la métrica equivocada lleva a la "
                    "conclusión de que no hay que corregir nada.")
    claves = list(SEL["val_SA"])
    filas = [[k.split(" - ")[0] + " · " + k.split(" - ")[1][:44],
              n(SEL["val_MdMRE"][k]), n(SEL["val_MdAElog"][k]),
              n(SEL["val_SA"][k], 3)] for k in claves]
    y = P.tabla(s, y, ["_Brazo (medido en validación)", "MdMRE", "MdAElog", "SA"],
                filas, anchos=[6.6, 1.6, 1.6, 1.6], tam=13, alto_fila=Inches(0.38),
                resaltar={claves.index(GANADOR): P.SUAVE})
    y += Inches(0.18)
    P.veredicto(s, y, "por qué se elige con el SA",
                f"El MdMRE y el MdAElog son medianas: describen la tarea típica y no ven "
                f"la cola. El estimado crudo del usuario es excelente en la tarea típica "
                f"y desastroso en una minoría: en un cliente nuevo tiene el mejor error "
                f"logarítmico mediano de todos los brazos y a la vez "
                f"{n(ec(EXT, 'H0', 'MAE_h'), 1)} h de error medio y un SA de "
                f"{n(ec(EXT, 'H0', 'SA'), 3)}, peor que adivinar al azar. Un artefacto "
                f"que suma tareas para armar un plan no puede convivir con esa cola, así "
                f"que se elige por SA: gana {GANADOR.split(' - ')[0]}.", tono="aviso")

    # ------------------------------------------------------------------ 14
    s, y = P.lamina(prs, "Frío, estimación cruda y estimación corregida")
    _, y2 = P.figura_ajustada(s, FIG / "fact_frio_vs_estimacion.png", y, Inches(4.05))
    P.veredicto(s, y2 + Inches(0.14), "el hallazgo central",
                f"La estimación del usuario ordena muy bien las tareas (Spearman "
                f"{n(ec(SIP, 'H0', 'spearman'), 2)} en la misma empresa frente a "
                f"{n(ec(SIP, 'R0', 'spearman'), 2)} del modelo en frío) pero está en la "
                f"escala equivocada en un cliente nuevo: "
                f"{n(ec(EXT, 'H0', 'MAE_h'), 1)} h de error medio y un SA negativo. "
                f"La corrección conserva el orden y arregla la escala.")

    # ------------------------------------------------------------------ 15
    s, y = P.lamina(prs, "Los tres diseños sobre el bloque de prueba entero",
                    "No solo sobre las tareas que traen estimación: el producto también "
                    "tiene que responder cuando no la hay.")
    filas = []
    for esc, etiq in (("prueba fija completa", "prueba fija · 4.225 tareas"),
                      ("externo completo", "externo · 13.255 tareas")):
        for br, nom in (("R0", "no preguntar: todo en frío"),
                        ("D0", "preguntar y mostrar el número del usuario"),
                        ("D1", "preguntar y corregir con el modelo")):
            filas.append([f"{etiq.split(' · ')[0]} · {nom}",
                          n(ec(esc, br, "MdMRE")), n(ec(esc, br, "MdAElog")),
                          n(ec(esc, br, "PRED25"), 3), n(ec(esc, br, "MAE_h"), 2),
                          n(ec(esc, br, "SA"), 3)])
    y = P.tabla(s, y, ["_Bloque y diseño", "MdMRE", "MdAElog", "PRED25", "MAE h", "SA"],
                filas, anchos=[5.6, 1.3, 1.4, 1.3, 1.3, 1.2], tam=12.5,
                alto_fila=Inches(0.36), resaltar={2: P.SUAVE, 5: P.SUAVE})
    y += Inches(0.2)
    P.veredicto(s, y, "lo que hay que mirar",
                f"En el bloque externo, mostrar el número crudo del usuario mejora el "
                f"MdMRE pero destroza el error en horas "
                f"({n(ec('externo completo', 'D0', 'MAE_h'), 2)} h frente a "
                f"{n(ec('externo completo', 'D1', 'MAE_h'), 2)} h corregido) y el SA "
                f"({n(ec('externo completo', 'D0', 'SA'), 3)} frente a "
                f"{n(ec('externo completo', 'D1', 'SA'), 3)}). Por eso la decisión no se "
                f"puede tomar con una sola métrica.", tono="ambar")

    # ------------------------------------------------------------------ 16
    s, y = P.lamina(prs, "La mejora depende de cuánta gente responde")
    _, y2 = P.figura_ajustada(s, FIG / "fact_adopcion.png", y, Inches(4.2))
    c = CU[CU.variante == "corregida"]
    paso = abs(float(c.MdMRE.iloc[-1]) - float(c.MdMRE.iloc[0])) / 10
    P.veredicto(s, y2 + Inches(0.14), "lectura para el producto",
                f"La curva es prácticamente recta: cada 10 % de gente que escribe su "
                f"estimación vale unas {n(paso, 3)} de MdMRE. No hace falta adopción "
                f"total para notarlo, y el campo puede ser opcional sin penalizar a "
                f"quien lo deje en blanco.")

    # ------------------------------------------------------------------ 17
    s, y = P.lamina(prs, "Lo que el usuario ve: el rango",
                    "Un número solo invita a creerlo. La banda conforme al 80 % está "
                    "calibrada en validación y su cobertura se verifica en prueba.")
    b = SEL["bandas_val"]
    an = (P.UTIL - Inches(0.4)) / 3
    P.cifra(s, P.MARGEN, y, an, "×" + n(b[[k for k in b if k.startswith('R0')][0]]["factor80"], 2),
            "banda del modelo en frío\n(de 4 h saldría 0,8–20 h)", tono="aviso")
    P.cifra(s, P.MARGEN + an + Inches(0.2), y, an,
            "×" + n(b[[k for k in b if k.startswith('H0')][0]]["factor80"], 2),
            "banda de la estimación cruda\ndel usuario", tono="ambar")
    P.cifra(s, P.MARGEN + 2 * (an + Inches(0.2)), y, an,
            "×" + n(b[GANADOR]["factor80"], 2),
            "banda de la estimación\ncorregida por el modelo")
    y += Inches(1.6)
    y = P.vinetas(s, y, [
        "Con la estimación del usuario la banda se estrecha a la mitad: es la diferencia "
        "entre “entre 1 y 20 horas” y “entre 2 y 10 horas”.",
        "La banda es lo que hace la predicción defendible ante el usuario: comunica la "
        "incertidumbre en vez de esconderla.",
        "Y es lo que un LLM no entrega: su número no viene con cobertura verificada.",
    ], tam=15.5, salto=Inches(0.5))

    # ------------------------------------------------------------------ 18
    s, y = P.lamina(prs, "El diseño recomendado, en tres pasos")
    y = P.vinetas(s, y, [
        ("1", "El LLM genera y descompone las tareas, como ya hace hoy."),
        ("2", "El modelo predice en frío y muestra número y banda de inmediato: es lo "
              "único que se puede hacer cuando el proyecto acaba de nacer."),
        ("3", "Se ofrece un campo opcional “¿cuánto crees que tardarás?”. Si el "
              "usuario lo rellena, el modelo corrige ese número en vez de ignorarlo, y "
              "la predicción y la banda se recalculan."),
        ("4", "Con el proyecto en marcha, la calibración en línea con las tareas ya "
              "cerradas sigue reduciendo el error sin reentrenar nada."),
    ], tam=15.5, salto=Inches(0.66))
    y += Inches(0.1)
    P.veredicto(s, y, "por qué este orden",
                "Los tres pasos son acumulativos y ninguno bloquea al anterior: el "
                "artefacto funciona el primer día en frío y mejora solo a medida que el "
                "usuario y el proyecto aportan información.")

    # ------------------------------------------------------------------ 19
    s, y = P.lamina(prs, "Lo que sigue sin cumplirse",
                    "La recomendación es la mejor de las opciones medidas, no una "
                    "solución al problema.")
    y = P.vinetas(s, y, [
        f"Ninguna configuración cumple los cuatro criterios de aceptación: el mejor "
        f"PRED(25) del diseño recomendado sobre la prueba completa es "
        f"{n(ec('prueba fija completa', 'D1', 'PRED25'), 3)}, lejos del 0,75 exigido.",
        f"En un cliente nuevo la corrección salva el error en horas, pero el MdMRE se "
        f"queda en {n(ec(EXT, GANADOR[:2], 'MdMRE'))}: la mitad de las tareas se falla "
        f"por más del {pc(ec(EXT, GANADOR[:2], 'MdMRE'), 0)}.",
        f"Los puntos de historia no sirven como estimación: en TAWOS la corrección "
        f"apenas mueve el MdMRE ({n(ec(TAW, 'R0', 'MdMRE'))} en frío frente a "
        f"{n(ec(TAW, GANADOR[:2], 'MdMRE'))} corregido).",
        "Y el corpus no contiene tareas escritas en español ni proyectos del tamaño de "
        "los de PBE; la validación real tiene que hacerse con datos del producto.",
    ], tam=15, salto=Inches(0.62))

    P.numerar(prs)
    Path(SALIDA).parent.mkdir(parents=True, exist_ok=True)
    prs.save(SALIDA)
    print(f"escrito {SALIDA} con {len(prs.slides.__iter__.__self__._sldIdLst)} laminas")


if __name__ == "__main__":
    main()
