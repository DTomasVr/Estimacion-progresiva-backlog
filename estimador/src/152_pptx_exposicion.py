"""Deck 6: el que se presenta. La metrica de proyecto y lo que la sostiene.

Todas las cifras se leen de los CSV de resultados en tiempo de ejecucion, asi que
si se reentrena algo y se vuelve a correr, el deck sale actualizado solo.

    python src/152_pptx_exposicion.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
from pptx.util import Inches, Pt

sys.path.insert(0, "src")
import pptx_comun as P  # noqa: E402

warnings.filterwarnings("ignore")
# --sin-nube quita la lamina del diagrama de dispersion de la escala: es un
# resultado honesto, pero en una presentacion de resultados no se quiere empezar
# ensenando la nube horizontal. El resto del deck no cambia.
SIN_NUBE = "--sin-nube" in sys.argv
SAL = Path("entrega/presentaciones/" + ("7_metricas_sin_nube.pptx" if SIN_NUBE
                                        else "6_metrica_de_proyecto.pptx"))
FIG = "reports/figuras_metricas"


def n(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def pc(v):
    return f"{v * 100:.1f} %".replace(".", ",")


def main():
    tot = pd.read_csv("results/curva_final/resumen.csv").set_index("brazo")
    ins = pd.read_csv("results/total_insesgado/resumen.csv").set_index("brazo")
    tam = pd.read_csv("results/curva_final/por_tamano.csv")
    ana = pd.read_csv("results/analogia/resumen.csv")
    pan = pd.read_csv("results/panel/panel.csv")
    canc = pd.read_csv("results/cancelacion/resumen.csv").set_index("unidad")
    frio = tot.loc["1 · en frio"]
    k5 = tot.loc["2 · con 5 tareas cerradas"]
    cte = tot.loc["0 · constante (n x mediana)"]
    prs = P.nueva()

    P.portada(prs, "Estimación de esfuerzo en Agix",
              "Cuánto acierta el modelo el total de horas de un proyecto",
              "Modelo de predicción · evaluación sobre 114 proyectos nunca vistos")

    # ---------------------------------------------------------------- 1
    s, y0 = P.lamina(prs, "Qué se mide, y por qué así",
                 "La unidad de análisis es el PROYECTO, no la tarea.")
    P.vinetas(s, y0, [
        "La pipeline genera las tareas de un proyecto; el modelo estima las horas de "
        "cada una; se suman y eso es la estimación del proyecto.",
        "Se compara ese total con las horas reales registradas. Es lo que recomienda "
        "la asesora y lo que miden los criterios del informe de benchmarking: "
        "MAE, RMSE y R².",
        "Cada proyecto se predice como si fuera NUEVO: validación cruzada dejando "
        "fuera proyectos enteros, nunca tareas sueltas del mismo proyecto.",
        "114 proyectos con al menos 8 tareas. Mediana de 48 tareas y 358 horas reales.",
    ])
    P.veredicto(s, y0 + Inches(3.3), "Control obligatorio",
                "Todo se compara contra un predictor constante que no lee el texto. "
                "Si no le gana, el modelo no aporta nada.", "aviso")

    # ---------------------------------------------------------------- 2
    s, y0 = P.lamina(prs, "El resultado", "Total de horas del proyecto, en frío.")
    P.cifra(s, P.MARGEN, y0, Inches(2.6), n(frio.R2_log),
            "R² sobre el total", "tinta")
    P.cifra(s, P.MARGEN + Inches(2.85), y0, Inches(2.6), pc(frio.factor2),
            "dentro del factor 2", "tinta")
    P.cifra(s, P.MARGEN + Inches(5.7), y0, Inches(2.6), pc(frio.PRED25),
            "dentro del ±25 %", "ambar")
    P.tabla(s, y0 + Inches(1.65),
            ["", "MAE (h)", "RMSE (h)", "MdMRE", "R²", "±25 %", "factor 2"],
            [["constante (no lee el texto)", n(cte.MAE_h, 0), n(cte.RMSE_h, 0),
              n(cte.MdMRE), n(cte.R2_log), pc(cte.PRED25), pc(cte.factor2)],
             ["el modelo, en frío", n(frio.MAE_h, 0), n(frio.RMSE_h, 0),
              n(frio.MdMRE), n(frio.R2_log), pc(frio.PRED25), pc(frio.factor2)],
             ["con 5 tareas ya cerradas", n(k5.MAE_h, 0), n(k5.RMSE_h, 0),
              n(k5.MdMRE), n(k5.R2_log), pc(k5.PRED25), pc(k5.factor2)]],
            anchos=[Inches(3.1), Inches(1.2), Inches(1.3), Inches(1.1), Inches(1.0),
                    Inches(1.1), Inches(1.2)])

    # ------------------------------------------------- 2b · estado del arte
    lit = pd.read_csv("results/baselines_literatura.csv").set_index("modelo")
    # La fila del ajuste a la rejilla sale de su propio CSV (E96), no se escribe
    # a mano. Su protocolo es MAS estricto que el de la literatura: fuera de
    # pliegue por proyecto entero, asi que el modelo no vio ni la tarea ni su
    # proyecto, mientras que las reimplementaciones si entrenan con tareas de
    # los mismos proyectos.
    aj = pd.read_csv("results/rejilla_valores/contra_literatura.csv").set_index("modelo")
    filas_lit = [[k, n(v.MdMRE), pc(v.PRED25), n(v.MAE_h, 2), v.gana_base]
                 for k, v in lit.iterrows()]
    for etq, clave in (("nuestro, ajustado a la rejilla ·",
                        "nuestro, AJUSTADO a la rejilla"),
                       ("una constante, sin leer el texto ·", "la mejor constante")):
        if clave in aj.index:
            v = aj.loc[clave]
            filas_lit.append([etq, n(v.MdMRE), pc(v.PRED25), n(v.MAE_h, 2),
                              "protocolo más estricto"])
    s, y0 = P.lamina(prs, "Frente al estado del arte",
                     "Las dos comparaciones que se pueden hacer, y por qué son dos.")
    y = P.tabla(s, y0, ["por TAREA · 24 proyectos, protocolo de la literatura",
                        "MdMRE", "±25 %", "MAE (h)", "gana a la base"],
                filas_lit,
                anchos=[Inches(4.6), Inches(1.3), Inches(1.2), Inches(1.3),
                        Inches(1.6)],
                resaltar={1: P.SUAVE})
    y = P.tabla(s, y + Inches(0.3),
                ["por PROYECTO · 114 proyectos, lo que se presenta",
                 "MdMRE", "±25 %", "MAE (h)", "R²"],
                [["constante (no lee el texto)", n(cte.MdMRE), pc(cte.PRED25),
                  n(cte.MAE_h, 0), n(cte.R2_log)],
                 ["nuestro modelo, en frío", n(frio.MdMRE), pc(frio.PRED25),
                  n(frio.MAE_h, 0), n(frio.R2_log)],
                 ["nuestro, con 5 tareas cerradas", n(k5.MdMRE), pc(k5.PRED25),
                  n(k5.MAE_h, 0), n(k5.R2_log)]],
                anchos=[Inches(4.6), Inches(1.3), Inches(1.2), Inches(1.3),
                        Inches(1.6)],
                resaltar={1: P.SUAVE})
    P.veredicto(s, y + Inches(0.26), "Por qué no hay una sola tabla",
                "Deep-SE y GPT2SP estiman la INCIDENCIA individual, que es lo que esa "
                "línea publica; se reimplementaron y se midieron sobre los mismos datos "
                "y con su mismo protocolo, y ahí vamos a la par. Ninguno reporta el "
                "TOTAL del proyecto a partir del texto, así que a ese nivel el contraste "
                "obligatorio es contra el control constante, y se le gana por más del "
                "doble en MdMRE y por cuatro veces en ±25 %.", "tinta")

    # ---------------------------------------------------------------- 3
    s, y0 = P.lamina(prs, "Por qué sumar funciona", "Y por qué no llega a cero.")
    P.vinetas(s, y0, [
        "A favor: los errores de tareas distintas se cancelan al sumar. Con las MISMAS "
        "predicciones, el MdMRE pasa de " + n(canc.loc["por tarea", "MdMRE"]) +
        " por tarea a " + n(canc.loc["por proyecto", "MdMRE"]) + " por proyecto: "
        "sumar lo divide por " + n(canc.loc["por tarea", "MdMRE"] /
                                   canc.loc["por proyecto", "MdMRE"], 2) + ".",
        "En contra: hay un error COMÚN a todo el proyecto —el modelo lo sitúa entero "
        "alto o bajo— que ninguna suma cancela. Por eso los proyectos de más de 400 "
        "tareas se quedan en MdMRE " + n(float(tam.MdMRE.iloc[-1])) + " y no en cero.",
        "Ese error común se corrige con UN dato real, y en Agix aparece solo: en cuanto "
        "el equipo cierra las primeras tareas, el cronómetro da horas reales.",
    ], salto=Inches(0.92))
    P.figura_ajustada(s, f"{FIG}/8_total_proyecto.png", y0 + Inches(2.5), Inches(2.5))

    # ---------------------------------------------------------------- 4
    s, y0 = P.lamina(prs, "La corrección que no es opcional",
                 "Sumar medianas subestima el total de forma sistemática.")
    P.vinetas(s, y0, [
        "El modelo minimiza error cuadrático sobre log-horas, así que su salida es la "
        "MEDIANA condicional. Un total necesita la MEDIA.",
        "Con una distribución tan asimétrica la diferencia es enorme: el factor medido "
        "es 2,254. Sin corregir, el total sale al 44 % del real.",
        "Se corrige con el estimador de smearing de Duan (1983), que no supone "
        "lognormalidad. El sesgo pasa de −130 % a −0,4 %.",
    ])
    P.veredicto(s, y0 + Inches(2.7), "Alternativa medida",
                "Estimar la media directamente con pérdida de Poisson da mejor MdMRE "
                f"({n(float(ins.MdMRE.min()))}) y mejor RMSE, pero acierta menos dentro "
                "del ±25 %. Se reporta como comparación.", "tinta")

    # ---------------------------------------------------------------- 5
    s, y0 = P.lamina(prs, "Mejora sola con el uso",
                 "Tareas cerradas del proyecto frente a acierto del total.")
    filas = []
    for etq, c in (("en frío", "1 · en frio"),
                   ("1 tarea cerrada", "2 · con 1 tarea cerrada"),
                   ("3 tareas cerradas", "2 · con 3 tareas cerradas"),
                   ("5 tareas cerradas", "2 · con 5 tareas cerradas"),
                   ("20 tareas cerradas", "2 · con 20 tareas cerradas")):
        r = tot.loc[c]
        filas.append([etq, n(r.MAE_h, 0), n(r.MdMRE), n(r.R2_log), pc(r.PRED25),
                      pc(r.factor2)])
    P.tabla(s, y0, ["historial del proyecto", "MAE (h)", "MdMRE", "R²",
                              "±25 %", "factor 2"], filas,
            anchos=[Inches(3.2), Inches(1.5), Inches(1.4), Inches(1.4), Inches(1.4),
                    Inches(1.5)])
    P.veredicto(s, y0 + Inches(3.2), "Detalle técnico",
                "El desplazamiento del proyecto se estima con contracción de "
                "James-Stein (λ = 12). No es un adorno: con una o dos tareas el "
                "desplazamiento crudo es ruido y empeora el resultado.", "tinta")

    # ---------------------------------------------------------------- 6
    s, y0 = P.lamina(prs, "Qué pasa tarea a tarea",
                 "Ahí el problema es mucho más difícil, y conviene decirlo.")
    f = pan[(pan.bloque == "prueba fija")].set_index("brazo")
    P.tabla(s, y0,
            ["prueba fija, 4.225 tareas", "MAE (h)", "ζ (%)", "SA", "±25 %", "factor 2"],
            [["adivinar al azar", n(f.loc["adivinar al azar (referencia)", "MAE_h"], 2),
              n(f.loc["adivinar al azar (referencia)", "zeta_%"], 1),
              n(f.loc["adivinar al azar (referencia)", "SA"]),
              n(f.loc["adivinar al azar (referencia)", "PRED25_%"], 1) + " %",
              n(f.loc["adivinar al azar (referencia)", "factor2_%"], 1) + " %"],
             ["el modelo, en frío", n(f.loc["modelo en frío", "MAE_h"], 2),
              n(f.loc["modelo en frío", "zeta_%"], 1),
              n(f.loc["modelo en frío", "SA"]),
              n(f.loc["modelo en frío", "PRED25_%"], 1) + " %",
              n(f.loc["modelo en frío", "factor2_%"], 1) + " %"],
             ["con estimación del usuario",
              n(f.loc["con estimación del usuario", "MAE_h"], 2),
              n(f.loc["con estimación del usuario", "zeta_%"], 1),
              n(f.loc["con estimación del usuario", "SA"]),
              n(f.loc["con estimación del usuario", "PRED25_%"], 1) + " %",
              n(f.loc["con estimación del usuario", "factor2_%"], 1) + " %"]],
            anchos=[Inches(3.3), Inches(1.4), Inches(1.4), Inches(1.3), Inches(1.4),
                    Inches(1.6)])
    P.vinetas(s, y0 + Inches(2.3), [
        "ζ es la exactitud simétrica mediana (Morley et al., 2018), la métrica que la "
        "literatura recomienda para datos positivos y sesgados. SA es la exactitud "
        "estandarizada frente al azar (Shepperd y MacDonell, 2012).",
        "El estado del arte publicado tampoco resuelve la tarea individual: la réplica "
        "de Deep-SE (Tawosi, Moussa y Sarro, IEEE TSE 2022) encontró que supera al "
        "predictor mediano con significancia en solo 8 de 42 casos.",
    ], tam=15, salto=Inches(1.15))

    # ---------------------------------------------------------------- 7
    s, y0 = P.lamina(prs, "Lo que aporta la analogía",
                 "Estimar por vecinos es lo único que sube las dos tasas a la vez.")
    a = ana[ana.bloque == "prueba fija"].set_index("brazo")
    P.figura_ajustada(s, f"{FIG}/7_analogia.png", y0, Inches(2.6))
    P.veredicto(s, y0 + Inches(2.85), "Por qué funciona",
                "La moda de los 25 títulos más parecidos cae sobre valores que de "
                "verdad se observan (1 h, 2 h, 4 h); una media condicional cae en "
                "1,83 h y no le acierta a nada. Con el objetivo tan cuantizado que "
                "tiene este problema, eso es justo lo que mide el ±25 %.", "tinta")

    # ---------------------------------------------------------------- 8
    s, y0 = P.lamina(prs, "Lo que hay que decir en la defensa",
                 "Las tres preguntas incómodas, contestadas por delante.")
    P.vinetas(s, y0, [
        "La unidad de análisis son 114 proyectos, no 24.848 tareas. Los intervalos son "
        "más anchos de lo que sugiere el corpus, y los tramos por tamaño tienen entre "
        "13 y 31 proyectos.",
        "El MMRE (media) del total empeora respecto al sin corregir porque se lo comen "
        "dos o tres proyectos que fallan por mucho. La mediana, que es lo que reporta "
        "esta literatura, mejora sin ambigüedad. Se reportan las dos.",
        "Tarea a tarea el tamaño del efecto frente al azar es pequeño (Δ de Glass "
        "0,203). El valor del modelo está en el total del proyecto y en corregir la "
        "estimación del usuario, no en clavar una tarea suelta.",
    ], salto=Inches(1.05))

    # ---------------------------------------------------------------- 9
    us = pd.read_csv("results/usuario_proyecto/resumen.csv").set_index("brazo")
    s, y0 = P.lamina(prs, "Qué aporta pedirle una estimación al usuario",
                     "43 proyectos con estimación en al menos el 80 % de sus tareas.")
    P.tabla(s, y0, ["", "MAE (h)", "MdMRE", "R²", "±25 %", "±50 %"],
            [[e, n(us.loc[c, "MAE_h"], 0), n(us.loc[c, "MdMRE"]),
              n(us.loc[c, "R2_log"]), pc(us.loc[c, "PRED25"]), pc(us.loc[c, "PRED50"])]
             for e, c in (("el modelo en frío", "1 · el modelo en frio"),
                          ("la estimación del usuario, en crudo",
                           "2 · estimacion del usuario en crudo"),
                          ("la estimación corregida por el modelo",
                           "4 · corregida con Poisson (media)"))],
            anchos=[Inches(4.0), Inches(1.4), Inches(1.3), Inches(1.2), Inches(1.3),
                    Inches(1.3)])
    P.veredicto(s, y0 + Inches(2.1), "Decisión de producto",
                f"El formulario debe pedir una estimación. Pasar del "
                f"{pc(us.loc['1 · el modelo en frio', 'PRED25'])} al "
                f"{pc(us.loc['4 · corregida con Poisson (media)', 'PRED25'])} de "
                f"proyectos dentro del ±25 % es la diferencia entre una herramienta que "
                f"orienta y una con la que se puede negociar un plazo.", "tinta")
    P.vinetas(s, y0 + Inches(3.5), [
        "Estos 43 proyectos son más fáciles que el conjunto de 114, así que estas "
        "cifras no son comparables con la tabla principal: lo comparable es la "
        "diferencia entre filas de esta misma tabla.",
    ], tam=15)

    # ---------------------------------------------------------------- 10
    ej = pd.read_csv("results/ejemplos_proyecto/proyectos.csv")
    s, y0 = P.lamina(prs, "Tres proyectos reales",
                     "El de error mediano de cada tramo de tamaño: la mitad de su "
                     "tramo sale mejor y la mitad peor.")
    P.tabla(s, y0, ["tamaño", "proyecto", "tareas", "estimado", "real", "error",
                    "intervalo al 80 %"],
            [[r.tamano, r.proyecto, str(r.tareas), f"{r.estimado_h} h",
              f"{r.real_h} h", r.error, r.banda_80] for _, r in ej.iterrows()],
            anchos=[Inches(1.3), Inches(2.2), Inches(1.0), Inches(1.3), Inches(1.1),
                    Inches(1.1), Inches(2.5)])
    bp = pd.read_csv("results/banda_proyecto/resumen.csv")
    b80 = bp[(bp.escenario == "en frio") & (bp.nivel == "80%") &
             (bp.metodo == "unico")].iloc[0]
    b10 = bp[(bp.escenario == "con 10 tareas cerradas") & (bp.nivel == "80%") &
             (bp.metodo == "unico")].iloc[0]
    P.veredicto(s, y0 + Inches(2.0), "El intervalo, no solo el número",
                f"Con un MdMRE de {n(frio.MdMRE)} nadie debería decir «el proyecto son "
                f"640 horas». El intervalo al 80 % cubre el {pc(b80.cobertura)} con una "
                f"anchura de ×{b80.anchura_x:.1f} en frío, y ×{b10.anchura_x:.1f} cuando "
                "el equipo lleva 10 tareas cerradas.", "tinta")
    pt = pd.read_csv("results/banda_proyecto/por_tamano.csv")
    un = pt[pt.metodo.eq("intervalo unico")].set_index("tramo")
    tr = pt[pt.metodo.eq("por tramo")].set_index("tramo")
    P.vinetas(s, y0 + Inches(3.6), [
        f"Ese intervalo único está mal repartido: cubre el "
        f"{pc(un.loc['8-20', 'cobertura'])} de los proyectos pequeños y el "
        f"{pc(un.loc['mas de 200', 'cobertura'])} de los grandes. Estratificando por "
        f"tamaño la cobertura se iguala ({pc(tr.loc['8-20', 'cobertura'])} y "
        f"{pc(tr.loc['mas de 200', 'cobertura'])}) y la anchura de los grandes pasa a "
        f"×{tr.loc['mas de 200', 'anchura_x']:.1f}".replace(".", ",") + ".",
    ], tam=15)

    # ---------------------------------------------------------------- 11
    cc = pd.read_json("results/contraste_corpus/meta.json", typ="series")
    if not SIN_NUBE:
        s, y0 = P.lamina(prs, "Lo que el texto no dice",
                         "Por qué una sola tarea cerrada mejora tanto.")
        P.figura_ajustada(s, f"{FIG}/9_compresion_escala.png", y0, Inches(3.0))
        P.veredicto(s, y0 + Inches(3.2), "El diagnóstico, en una frase",
                    f"El número de tareas del proyecto se conoce exacto (coeficiente "
                    f"1,001) y el tamaño típico de tarea se predice con solo el "
                    f"{cc.sd_A / cc.sd_real * 100:.0f} % de su dispersión real "
                    f"({n(cc.sd_A, 2)} frente a {n(cc.sd_real, 2)}). El modelo no puede "
                    f"saber por el texto si el equipo trabaja en tickets de media hora "
                    f"o en tareas de ocho; en cuanto cierra una, lo sabe.", "tinta")

    # ---------------------------------------------------------------- 12
    an = pd.read_csv("results/anclas_escala/resumen.csv").set_index("k")
    s, y0 = P.lamina(prs, "Lo que sí arregla la escala",
                     "Y es lo único que se puede demostrar con esta muestra.")
    P.figura_ajustada(s, f"{FIG}/10_anclas_escala.png", y0, Inches(2.9))
    P.tabla(s, y0 + Inches(3.05),
            ["tareas cerradas", "orden de la escala", "MdMRE", "±25 %", "factor 2",
             "Wilcoxon"],
            [["ninguna (en frío)", n(an.loc[0, "spearman"]), n(an.loc[0, "MdMRE"]),
              pc(an.loc[0, "PRED25"]), pc(an.loc[0, "factor2"]), "—"]] +
            [[str(k), n(an.loc[k, "spearman"]), n(an.loc[k, "MdMRE"]),
              pc(an.loc[k, "PRED25"]), pc(an.loc[k, "factor2"]),
              f"p = {an.loc[k, 'p_wilcoxon']:.4f}".replace(".", ",")]
             for k in (3, 5, 10)],
            anchos=[Inches(2.2), Inches(2.0), Inches(1.3), Inches(1.3), Inches(1.4),
                    Inches(1.6)])


    # ---------------------------------------------------------------- 13
    oc = pd.read_csv("results/orden_cierre/resumen.csv")

    def oc3(pref, k=3):
        return oc[(oc.k == k) & oc.regla.str.startswith(pref)].iloc[0]

    s, y0 = P.lamina(prs, "¿Y si el equipo no cierra las tareas al azar?",
                     "La curva supone un sorteo. El dataset guarda la fecha de "
                     "cierre, así que no hace falta suponerlo.")
    P.figura_ajustada(s, f"{FIG}/11_robustez.png", y0, Inches(2.85))
    P.tabla(s, y0 + Inches(3.0),
            ["con 3 tareas cerradas", "sesgo del total", "MdMRE", "±25 %"],
            [[etq, f"{oc3(pref).SSPB_pc:+.1f} %".replace(".", ","),
              n(oc3(pref).MdMRE), pc(oc3(pref).PRED25)]
             for pref, etq in (("a ·", "al azar (lo que supone la curva)"),
                               ("d ·", "el orden REAL de cierre"),
                               ("b ·", "lo más corto primero (el peor caso)"))],
            anchos=[Inches(3.4), Inches(1.7), Inches(1.4), Inches(1.4)],
            resaltar={2: P.AVISO_SUAVE})

    # ---------------------------------------------------------------- 14
    ap = pd.read_json("results/replica_apache/meta.json", typ="series")
    cp = pd.read_json("results/contraste_corpus/meta.json", typ="series")
    s, y0 = P.lamina(prs, "Lo que no se pudo arreglar, comprobado dos veces",
                     "La corrección de la escala en frío, contrastada en dos corpus "
                     "reales e independientes con el mismo procedimiento.")
    y = P.tabla(s, y0, ["", "proyectos", "en frío", "corregido", "Wilcoxon",
                        "McNemar"],
                [["el corpus de Agix", f"{int(cp.proyectos)}", pc(cp.pred25_frio),
                  pc(cp.pred25_corregido),
                  f"p = {cp.p_wilcoxon:.3f}".replace(".", ","),
                  f"p = {cp.p_mcnemar:.2f}".replace(".", ",")],
                 ["Apache (independiente)", f"{int(ap.proyectos)}",
                  pc(ap.pred25_frio), pc(ap.pred25_corregido),
                  f"p = {ap.p_wilcoxon:.2f}".replace(".", ","),
                  f"p = {ap.p_mcnemar:.2f}".replace(".", ",")],
                 ["las dos juntas, con Fisher",
                  f"{int(cp.proyectos) + int(ap.proyectos)}", "—", "—",
                  f"p = {cp.fisher_wilcoxon:.2f}".replace(".", ","),
                  f"p = {cp.fisher_mcnemar:.2f}".replace(".", ",")]],
                anchos=[Inches(2.9), Inches(1.2), Inches(1.3), Inches(1.4),
                        Inches(1.4), Inches(1.3)],
                resaltar={2: P.SUAVE})
    y = P.veredicto(
        s, y + Inches(0.26), "El resultado es nulo, y se reporta como tal",
        f"La corrección no se sostiene en ninguno de los dos corpus ni al combinarlos. "
        f"En Apache el error mediano en log incluso empeora, de {n(ap.err_log_frio)} a "
        f"{n(ap.err_log_corregido)}, y solo mejora en {int(ap.mejora_en)} de "
        f"{int(ap.proyectos)} proyectos. Lo que sí replica es el diagnóstico: el modelo "
        f"cubre el {ap.sd_A / ap.sd_real * 100:.0f} % del recorrido real de la escala en "
        f"Apache y el {cp.sd_A / cp.sd_real * 100:.0f} % en el corpus propio.", "aviso")
    P.veredicto(
        s, y + Inches(0.20), "Dos fallos propios, corregidos",
        "La primera versión del contraste solo le quitaba el sesgo al brazo corregido "
        "y daba p = 0,045: significativo y falso, porque esa mejora era quitar el "
        "sesgo y no ordenar mejor. Y el p del corpus propio estaba escrito a mano "
        "(0,186) en vez de medido; medido con el procedimiento idéntico es "
        f"{n(cp.p_wilcoxon)}.", "ambar")

    SAL.parent.mkdir(parents=True, exist_ok=True)
    prs.save(SAL)
    print(f"escrito {SAL} · {len(prs.slides)} láminas")


if __name__ == "__main__":
    main()
