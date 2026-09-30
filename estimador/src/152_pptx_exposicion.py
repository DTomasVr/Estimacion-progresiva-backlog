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
SAL = Path("entrega/presentaciones/6_metrica_de_proyecto.pptx")
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
    frio = tot.loc["1 · en frio"]
    k5 = tot.loc["2 · con 5 tareas cerradas (cociente de sumas)"]
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

    # ---------------------------------------------------------------- 3
    s, y0 = P.lamina(prs, "Por qué sumar funciona", "Y por qué no llega a cero.")
    P.vinetas(s, y0, [
        "A favor: los errores de tareas distintas se cancelan al sumar. El MdMRE pasa "
        "de 0,741 por tarea a " + n(frio.MdMRE) + " por proyecto.",
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
                   ("1 tarea cerrada", "2 · con 1 tarea cerrada (cociente de sumas)"),
                   ("3 tareas cerradas", "2 · con 3 tareas cerradas (cociente de sumas)"),
                   ("5 tareas cerradas", "2 · con 5 tareas cerradas (cociente de sumas)"),
                   ("20 tareas cerradas", "2 · con 20 tareas cerradas (cociente de sumas)")):
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
                "El formulario debe pedir una estimación. Pasar del 44 % al 70 % de "
                "proyectos dentro del ±25 % es la diferencia entre una herramienta que "
                "orienta y una con la que se puede negociar un plazo.", "tinta")
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
             (bp.metodo == "cuantiles del residuo")].iloc[0]
    b10 = bp[(bp.escenario == "con 10 tareas cerradas") & (bp.nivel == "80%") &
             (bp.metodo == "cuantiles del residuo")].iloc[0]
    P.veredicto(s, y0 + Inches(2.0), "El intervalo, no solo el número",
                f"Con un MdMRE de {n(frio.MdMRE)} nadie debería decir «el proyecto son "
                f"640 horas». El intervalo al 80 % cubre el {pc(b80.cobertura)} con una "
                f"anchura de ×{b80.anchura_x:.1f} en frío, y ×{b10.anchura_x:.1f} cuando "
                "el equipo lleva 10 tareas cerradas.", "tinta")
    P.vinetas(s, y0 + Inches(3.6), [
        "Ese intervalo único está mal repartido: cubre el 70 % de los proyectos "
        "pequeños y el 96 % de los grandes. Estratificando por tamaño la cobertura se "
        "iguala y la anchura pasa a ×2,7 para los de más de 200 tareas.",
    ], tam=15)

    SAL.parent.mkdir(parents=True, exist_ok=True)
    prs.save(SAL)
    print(f"escrito {SAL} · {len(prs.slides)} láminas")


if __name__ == "__main__":
    main()
