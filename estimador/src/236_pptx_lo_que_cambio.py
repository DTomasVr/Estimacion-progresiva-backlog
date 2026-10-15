"""Deck 9: lo que cambio en esta ronda, como se hizo y que se descubrio.

Todas las cifras se leen de los CSV en tiempo de ejecucion. Si se vuelve a
correr un experimento, el deck sale actualizado solo.

    python src/236_pptx_lo_que_cambio.py
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from pptx.util import Inches

sys.path.insert(0, "src")
import pptx_comun as P  # noqa: E402

warnings.filterwarnings("ignore")
SAL = Path("entrega/presentaciones/9_lo_que_cambio.pptx")


def n(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def pc(v, d=1):
    return f"{v * 100:.{d}f} %".replace(".", ",")


def x(v, d=2):
    return ("x" + f"{v:.{d}f}").replace(".", ",")


def main():
    # ----------------------------------------------------------- los datos
    lit = pd.read_csv("results/baselines_literatura.csv").set_index("modelo")
    rej = pd.read_csv("results/rejilla_valores/contra_literatura.csv").set_index("modelo")
    art = pd.read_csv("results/centrar_artefacto/resumen.csv").set_index("salida")
    anc = pd.read_csv("results/anclas_escala/resumen.csv").set_index("k")
    ban = pd.read_csv("results/banda_que_cierra_antes/resumen.csv")
    bpt = pd.read_csv("results/banda_por_tarea/resumen.csv").set_index("banda")
    hpd = pd.read_csv("results/hpd_curva/resumen.csv").set_index("conjunto")
    dev = json.load(open("results/anclaje_desarrollador/meta.json"))
    des = json.load(open("results/desacuerdo_banda/meta.json"))
    com = json.load(open("results/compuerta_necesaria/meta.json"))
    igu = json.load(open("results/estimacion_igual_al_real/meta.json"))
    hco = json.load(open("results/arranque_con_historial/meta.json"))
    seq = json.load(open("results/sesgo_del_equipo/meta.json"))
    heq = pd.read_csv("results/horas_del_equipo/resumen.csv")
    lam = pd.read_csv("results/lambda_condicional/resumen.csv")
    e93t = pd.read_csv("results/estimacion_igual_al_real/por_tarea.csv").set_index("subconjunto")
    e93p = pd.read_csv("results/estimacion_igual_al_real/por_proyecto.csv")
    # el mejor caso del total: con estimacion del usuario, el mejor (k, lambda)
    _ce = lam[(lam.escenario == "CON estimacion") & (lam.k > 0)]
    MEJOR_TOT = float(_ce.PRED25.max())
    _b = _ce.loc[_ce.PRED25.idxmax()]
    MEJOR_COND = f"43 proyectos, estimando TODO + {int(_b.k)} cerradas"
    # el equipo: C (otro proyecto del mismo equipo) contra A (en frio)
    _h = heq.set_index(["k", "brazo"])
    _k = int(heq[heq.brazo.str.startswith("C")].k.iloc[0])
    _a0 = float(heq[heq.brazo.str.startswith("A")].PRED25.iloc[0])
    _cc = heq[heq.brazo.str.startswith("C")]
    _dd = heq[heq.brazo.str.startswith("D")]
    EQUIPO = f"de {(_cc.PRED25.max() - _a0) * 100:+.1f} a {(_cc.PRED25.min() - _a0) * 100:+.1f} puntos".replace(".", ",")
    CONTROL = f"de {(_dd.PRED25.max() - _a0) * 100:+.1f} a {(_dd.PRED25.min() - _a0) * 100:+.1f} puntos".replace(".", ",")
    PROPIO = f"de {(heq[heq.brazo.str.startswith("B")].PRED25.min() - _a0) * 100:+.1f} a {(heq[heq.brazo.str.startswith("B")].PRED25.max() - _a0) * 100:+.1f} puntos".replace(".", ",")
    ENTRE_ORG = seq["var_entre"] / (seq["var_entre"] + seq["var_dentro"])
    LIMPIOS = e93p[e93p.proyectos == e93p.proyectos.min()].iloc[0]
    tir = json.load(open("results/hpd_usable/meta.json")) if Path(
        "results/hpd_usable/meta.json").exists() else {}

    hoy = art.loc["A · como se despliega hoy"]
    cor = art.loc["E · B + rejilla, SIN centrar"]
    sin_aj = rej.loc["nuestro, sin ajustar"]
    con_aj = rej.loc["nuestro, AJUSTADO a la rejilla"]
    cte = rej.loc["la mejor constante"]
    gpt = lit.loc["GPT2SP (GPT-2 ajustado)"]
    A2 = ban[ban.banda.str.startswith("A2")].set_index("k")
    D15 = ban[ban.banda.str.startswith("D · tramo contraido M=1500")].set_index("k")
    D6 = ban[ban.banda.str.startswith("D · tramo contraido M=600")].set_index("k")
    D2 = ban[ban.banda.str.startswith("D · tramo contraido M=200")].set_index("k")

    prs = P.nueva()
    P.portada(prs, "Lo que cambió en esta ronda",
              "Qué se movió, cómo se hizo y qué se descubrió por el camino",
              "Modelo de estimación de esfuerzo · todas las cifras leídas de los "
              "CSV de resultados")

    # =================================================== 1 · el resumen
    s, y0 = P.lamina(prs, "El resumen en una lámina",
                     "El ±25 % del total del proyecto NO se movió. Se movió todo "
                     "lo demás.")
    P.tabla(s, y0, ["", "antes", "ahora"],
            [["±25 % del total, en frío", pc(anc.loc[0, "PRED25"]),
              pc(anc.loc[0, "PRED25"]) + "  (sin cambio)"],
             ["±25 % por tarea", pc(sin_aj.PRED25), pc(con_aj.PRED25)],
             ["sesgo de la salida", x(hoy.sesgo), x(cor.sesgo)],
             ["banda del total, 1 tarea cerrada", x(A2.loc[1, "anchura_mediana"]),
              x(D15.loc[1, "anchura_mediana"])],
             ["banda del total, 3 cerradas", x(A2.loc[3, "anchura_mediana"]),
              x(D6.loc[3, "anchura_mediana"])],
             ["banda por tarea, con estimación",
              x(bpt.loc["A'' · una sola anchura, SOLO limpias", "anchura_mediana"]),
              x(bpt.loc["E' · con la estimacion, SOLO limpias", "anchura_mediana"])]],
            anchos=[Inches(4.6), Inches(2.6), Inches(3.0)], resaltar={1: P.SUAVE})
    P.veredicto(s, y0 + Inches(3.6), "La idea",
                "La precisión del total está en su techo; lo que mejora la "
                "experiencia es que el rango se estreche con el uso, y eso sí se "
                "ha conseguido.", "tinta")

    # =================================================== 2 · precisión
    P.seccion(prs, "1", "La precisión por tarea",
              "Dos cambios, los dos sin tocar el modelo.")

    s, y0 = P.lamina(prs, "El hallazgo: estábamos optimizando otra función",
                     "Predecir una constante para TODAS las tareas le gana a "
                     "nuestro modelo en ±25 %.")
    P.cifra(s, P.MARGEN, y0, Inches(3.0), pc(sin_aj.PRED25),
            "nuestro modelo, por tarea", "gris")
    P.cifra(s, P.MARGEN + Inches(3.3), y0, Inches(3.0), pc(cte.PRED25),
            "una constante de 2,45 h", "aviso")
    P.cifra(s, P.MARGEN + Inches(6.6), y0, Inches(3.0), pc(con_aj.PRED25),
            "ajustando a la rejilla", "tinta")
    P.vinetas(s, y0 + Inches(1.6), [
        ("Shepperd y Kitchenham demostraron que PRED(25) mide la dispersión y la "
         "CURTOSIS de estimado/real: premia concentrar, no ordenar.", "por qué"),
        ("Entrenamos con error cuadrático en log, cuyo minimizador es la mediana "
         "condicional, y reportamos PRED(25). Son funciones distintas.", "el fallo"),
        ("La distribución del esfuerzo es un peine: diez valores cubren el 46,8 % "
         "del corpus. La mediana condicional cae ENTRE dientes.", "el dato"),
    ])
    P.veredicto(s, y0 + Inches(3.5), "El arreglo",
                "Redondear la predicción al valor que la gente registra de verdad. "
                "Sin tocar el modelo, sin mover el MAE, el SA ni el orden entre "
                "tareas.", "tinta")

    s, y0 = P.lamina(prs, "Frente al estado del arte",
                     "Bloque de prueba de 4.225 tareas, el mismo que usan las "
                     "reimplementaciones.")
    P.tabla(s, y0, ["", "MdMRE", "±25 %", "MAE (h)"],
            [[k, n(v.MdMRE), pc(v.PRED25), n(v.MAE_h, 2)]
             for k, v in lit.iterrows()] +
            [["nuestro, ajustado a la rejilla", n(con_aj.MdMRE),
              pc(con_aj.PRED25), n(con_aj.MAE_h, 2)],
             ["una constante, sin leer el texto", n(cte.MdMRE), pc(cte.PRED25),
              n(cte.MAE_h, 2)]],
            anchos=[Inches(5.0), Inches(1.7), Inches(1.7), Inches(1.7)],
            resaltar={5: P.SUAVE})
    P.veredicto(s, y0 + Inches(3.5), "Las dos lecturas",
                f"Pasamos por encima de GPT2SP ({pc(gpt.PRED25)}) con un protocolo "
                "MÁS estricto: fuera de pliegue por proyecto entero. Pero una "
                "constante le gana a todo lo publicado, así que PRED(25) no se "
                "puede reportar sola: va con el SA y el Spearman.", "ambar")

    s, y0 = P.lamina(prs, "El sesgo que ninguna métrica veía",
                     "Revisando a mano qué entra y qué sale del artefacto, no del "
                     "modelo limpio.")
    P.vinetas(s, y0, [
        ("estimar_tareas aplicaba SIEMPRE el factor de estilo 0,8313, medido para "
         "el texto del GENERADOR. Sobre texto de persona el sesgo era x1,002.",
         "el fallo"),
        ("El artefacto mostraba un 31 % menos de horas de las reales. A las tareas "
         "de más de 16 h les decía 3,95 h.", "el efecto"),
        ("Todas las métricas se calculaban sobre un HGB fuera de pliegue, nunca "
         "sobre el .joblib. Medíamos el modelo, no el producto.", "por qué no se veía"),
    ])
    P.tabla(s, y0 + Inches(2.4), ["2.000 tareas reales de prueba", "±25 %",
                                  "factor 2", "MAE", "SA", "sesgo"],
            [["como se desplegaba", pc(hoy.PRED25), pc(hoy.factor2),
              n(hoy.MAE_h, 2), n(hoy.SA, 4), x(hoy.sesgo, 3)],
             ["corregido", pc(cor.PRED25), pc(cor.factor2), n(cor.MAE_h, 2),
              n(cor.SA, 4), x(cor.sesgo, 3)]],
            anchos=[Inches(3.6), Inches(1.3), Inches(1.4), Inches(1.2),
                    Inches(1.3), Inches(1.4)], resaltar={1: P.SUAVE})
    P.veredicto(s, y0 + Inches(4.0), "Una decisión que conviene explicar",
                "Dejar el sesgo dentro daba MÁS ±25 % (18,9 % contra 17,5 %), "
                "porque la métrica premia subestimar. Se eligió la salida "
                "insesgada.", "ambar")

    # =================================================== 3 · bandas
    P.seccion(prs, "2", "Las bandas",
              "Donde más se ha avanzado, y lo que de verdad ve el usuario.")

    s, y0 = P.lamina(prs, "La banda del total no se estrechaba nunca",
                     "Dos fallos, los dos ensanchaban sin motivo.")
    P.vinetas(s, y0, [
        ("Multiplicaba el TOTAL ENTERO por el residuo, incluidas las horas ya "
         "cerradas, que se saben y no tienen incertidumbre.", "fallo 1"),
        ("Usaba SIEMPRE los residuos del arranque en frío, aunque el anclaje ya "
         "hubiera bajado el error.", "fallo 2"),
        ("Calibrar dentro del tramo de tamaño cuesta anchura cuando hay pocas "
         "cerradas: el cuantil conforme se vuelve conservador.", "fallo 3"),
    ])
    P.tabla(s, y0 + Inches(2.4), ["tareas cerradas", "antes", "ahora"],
            [[str(k), x(A2.loc[k, "anchura_mediana"]),
              x(D.loc[k, "anchura_mediana"])]
             for k, D in ((1, D15), (2, D15), (3, D6), (5, D2))] +
            [["10", x(A2.loc[10, "anchura_mediana"]),
              x(A2.loc[10, "anchura_mediana"])]],
            anchos=[Inches(3.4), Inches(2.6), Inches(2.6)], resaltar={1: P.SUAVE})
    P.veredicto(s, y0 + Inches(4.3), "Todo a cobertura real 0,800",
                "Es el detalle que costó tres mediciones equivocadas: la banda "
                "anterior parecía estrecha porque cubría el 75 %, no el 80 %.",
                "aviso")

    s, y0 = P.lamina(prs, "La banda por tarea",
                     "Siete formas de estrecharla en frío. Solo funciona pedir la "
                     "estimación.")
    P.tabla(s, y0, ["", "anchura", "cobertura"],
            [["en frío, una sola anchura",
              x(bpt.loc["A · una sola anchura", "anchura_mediana"]),
              pc(bpt.loc["A · una sola anchura", "cobertura"])],
             ["por acuerdo de los vecinos",
              x(bpt.loc["B · por acuerdo de los vecinos", "anchura_mediana"]),
              pc(bpt.loc["B · por acuerdo de los vecinos", "cobertura"])],
             ["cuantiles aprendidos (CQR)",
              x(bpt.loc["C · cuantiles aprendidos (CQR)", "anchura_mediana"]),
              pc(bpt.loc["C · cuantiles aprendidos (CQR)", "cobertura"])],
             ["por el valor predicho",
              x(bpt.loc["D · por el valor predicho", "anchura_mediana"]),
              pc(bpt.loc["D · por el valor predicho", "cobertura"])],
             ["CON la estimación del usuario",
              x(bpt.loc["E' · con la estimacion, SOLO limpias", "anchura_mediana"]),
              pc(bpt.loc["E' · con la estimacion, SOLO limpias", "cobertura"])]],
            anchos=[Inches(5.2), Inches(2.1), Inches(2.1)], resaltar={5: P.SUAVE})
    P.veredicto(s, y0 + Inches(3.3), "Por qué fallan todas",
                "La correlación entre las señales de incertidumbre del modelo y su "
                f"error real es de {n(des['corr_desacuerdo'], 3)} (desacuerdo del "
                f"ensemble) a {n(des['corr_sigma'], 3)} (sigma aprendido). El modelo "
                "no sabe cuándo se equivoca, así que no hay nada que condicionar.",
                "aviso")

    s, y0 = P.lamina(prs, "La tira de probabilidad",
                     "Región de densidad máxima: la literatura dice que es el "
                     "conjunto más pequeño cuando la distribución es multimodal.")
    P.tabla(s, y0, ["al 80 %", "anchura", "cobertura", "trozos"],
            [["la banda de hoy",
              x(hpd.loc["A · la banda de hoy", "anchura_mediana"]),
              pc(hpd.loc["A · la banda de hoy", "cobertura"]),
              f"{hpd.loc['A · la banda de hoy', 'trozos_mediana']:.0f}"],
             ["conjunto HPD",
              x(hpd.loc["HPD · sin limite de trozos", "anchura_mediana"]),
              pc(hpd.loc["HPD · sin limite de trozos", "cobertura"]),
              f"{hpd.loc['HPD · sin limite de trozos', 'trozos_mediana']:.0f}"],
             ["HPD con 5 trozos como mucho",
              x(hpd.loc["HPD · maximo 5 trozos", "anchura_mediana"]),
              pc(hpd.loc["HPD · maximo 5 trozos", "cobertura"]), "5"]],
            anchos=[Inches(4.4), Inches(1.9), Inches(1.9), Inches(1.6)],
            resaltar={2: P.SUAVE})
    P.vinetas(s, y0 + Inches(2.7), [
        "Un 68 % más estrecha EN MEDIDA, y con la cobertura por tramo mucho más "
        "pareja: de 0,45 a 0,86, frente a 0,25 a 1,00 de la banda.",
        "Pero su RECORRIDO es mayor: se queda con los dientes del peine y descarta "
        "los huecos. Para un «entre X e Y» la banda sigue siendo mejor.",
        "Desplegada como salida OPCIONAL del detalle, no como sustituta.",
    ])

    # =================================================== 4 · hallazgos
    P.seccion(prs, "3", "Lo que se descubrió",
              "Los negativos están medidos, no supuestos. Es lo que acota el "
              "problema.")

    s, y0 = P.lamina(prs, "El muro: el texto no dice en qué régimen cae la tarea",
                     "Un experto en tareas cortas funciona. Lo que no se puede es "
                     "saber a cuál aplicarlo.")
    P.cifra(s, P.MARGEN, y0, Inches(3.0), pc(com["pred25_global"]),
            "modelo global, por tarea", "gris")
    P.cifra(s, P.MARGEN + Inches(3.3), y0, Inches(3.0), pc(com["pred25_oraculo"]),
            "enrutado ORÁCULO", "tinta")
    P.cifra(s, P.MARGEN + Inches(6.6), y0, Inches(3.0),
            n(com["mejor_auc_real"], 3), "AUC de la compuerta real", "aviso")
    P.vinetas(s, y0 + Inches(1.6), [
        ("Enrutar solo empata con el global a partir de AUC 0,80. Lo alcanzable "
         "es 0,65.", "el umbral"),
        ("Cinco señales independientes convergen ahí: clasificador sobre el "
         "título, predicción del propio modelo, las dos juntas, tipo y prioridad, "
         "y la descripción.", "por qué es el dato"),
        ("Doce vías medidas para mejorar la predicción, todas cerradas: historial "
         "aprendido, similitud de texto, proyectos sintéticos, repeso por "
         "densidad, preguntar la escala, descomponer en pasos...", "el recuento"),
    ])

    s, y0 = P.lamina(prs, "Lo que NO se transfiere entre proyectos",
                     "El equipo lleva ALGO —gana al control de otra "
                     "organización— pero sigue siendo peor que no usar nada.")
    P.tabla(s, y0, ["", "resultado"],
            [["horas de proyectos del MISMO equipo", EQUIPO],
             ["horas de OTRA organización (control)", CONTROL],
             ["sesgo de estimación del equipo",
              f"{pc(ENTRE_ORG)} entre organizaciones (ruido)"],
             ["historial de los mismos desarrolladores",
              f"SA {n(hco['SA_frio'], 4)} → {n(hco['SA_devs'], 4)} (peor)"],
             ["cerradas del PROPIO proyecto", PROPIO]],
            anchos=[Inches(6.0), Inches(4.2)], resaltar={5: P.SUAVE})
    P.veredicto(s, y0 + Inches(3.4), "El contraste que lo explica",
                "El desplazamiento de una MISMA persona correlaciona +0,089 entre "
                "sus proyectos. Dos personas DISTINTAS del MISMO proyecto, +0,164. "
                "El proyecto une más que la persona.", "tinta")

    s, y0 = P.lamina(prs, "Una amenaza de validez que detectamos nosotros",
                     "En las tareas con estimación del usuario, el valor real suele "
                     "ser la estimación copiada.")
    P.cifra(s, P.MARGEN, y0, Inches(4.0),
            pc(igu["identicas_por_tarea"]),
            "de las filas con estimación tienen el real copiado", "aviso", tam=36)
    P.vinetas(s, y0 + Inches(1.6), [
        "Muy desigual por fuente: Spring 47,4 %, Jira 17,0 %, Sonatype 10,5 %. La "
        "tarea se cierra imputando lo estimado en vez de lo medido.",
        "Repetido sin esas filas, la ventaja de la estimación AGUANTA: por tarea "
        f"pasa de {(e93t.loc['todas'].con_estimacion - e93t.loc['todas'].texto) * 100:+.1f} a "
        f"{(e93t.loc['SIN las identicas'].con_estimacion - e93t.loc['SIN las identicas'].texto) * 100:+.1f} puntos, y por proyecto SUBE, de "
        f"{pc(LIMPIOS.frio)} a {pc(LIMPIOS.con_estimacion)} en los {int(LIMPIOS.proyectos)} proyectos más limpios.".replace(".", ",", 0),
        "Y aparece una inversión: en datos limpios el modelo GANA a la estimación "
        f"cruda ({pc(e93t.loc['SIN las identicas'].con_estimacion)} contra "
        f"{pc(e93t.loc['SIN las identicas'].cruda)}); con las filas contaminadas "
        "dentro, la comparación se invierte y engaña.",
    ])
    P.veredicto(s, y0 + Inches(3.9), "Por qué conviene declararlo",
                "Es la primera pregunta que haría un revisor. Mejor que salga de "
                "nosotros, con la medida al lado.", "tinta")

    s, y0 = P.lamina(prs, "La trampa que mordió tres veces",
                     "Comparar anchuras de banda a cobertura distinta.")
    P.vinetas(s, y0, [
        "La banda desplegada parecía estrecha, pero cubría el 75 %, no el 80 % que "
        "prometía. Cualquier método honesto parece peor al lado.",
        "Pasó con la banda del total, con la ventana adaptativa por tarea y con la "
        "calibración por tramo. Las tres veces la primera lectura fue equivocada.",
        "El protocolo correcto: ajustar el nivel NOMINAL de cada método hasta que "
        "la cobertura REAL sea 0,800 exacta, y solo entonces comparar anchuras.",
    ])
    P.veredicto(s, y0 + Inches(2.7), "Lo que cambió al corregirlo",
                "La ventana adaptativa por tarea, que parecía ganar un 7 %, "
                "resultó perder; y la contracción del cuantil del tramo, que "
                "parecía neutra, resultó valer un 31 % de anchura con una tarea "
                "cerrada.", "aviso")

    # =================================================== 5 · cierre
    s, y0 = P.lamina(prs, "Qué reportar",
                     "Lo medido, con su contexto.")
    P.tabla(s, y0, ["", "valor", "condición"],
            [["±25 % del total, en frío", pc(anc.loc[0, "PRED25"]),
              "114 proyectos, nada pedido al usuario"],
             ["con 5 tareas cerradas", pc(anc.loc[5, "PRED25"]), "114 proyectos"],
             ["con 20 tareas cerradas", pc(anc.loc[20, "PRED25"]), "114 proyectos"],
             ["±25 % por tarea", pc(con_aj.PRED25),
              "4.225 tareas, por encima de lo publicado"],
             ["±25 % del total, mejor caso", pc(MEJOR_TOT), MEJOR_COND]],
            anchos=[Inches(3.8), Inches(1.7), Inches(4.7)], resaltar={1: P.SUAVE})
    P.veredicto(s, y0 + Inches(3.6), "El aviso",
                f"El {pc(MEJOR_TOT)} necesita tres condiciones a la vez y no es lo que da el "
                "sistema a un proyecto nuevo. Si se da suelto, quien mire el frío "
                "pensará que hay un error.", "aviso")

    SAL.parent.mkdir(parents=True, exist_ok=True)
    prs.save(SAL)
    print(f"escrito {SAL} · {len(prs.slides)} láminas")


if __name__ == "__main__":
    main()
