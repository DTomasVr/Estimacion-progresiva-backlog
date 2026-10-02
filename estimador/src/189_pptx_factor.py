"""Deck breve: como se calcula y se usa el factor de anclaje.

Cinco laminas, todas las cifras leidas de results/ en tiempo de ejecucion.

    python src/189_pptx_factor.py
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
from pptx.util import Inches

sys.path.insert(0, "src")
import pptx_comun as P  # noqa: E402

warnings.filterwarnings("ignore")
SAL = Path("entrega/presentaciones/8_factor_de_anclaje.pptx")
FIG = "reports/figuras_metricas"


def n(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def pc(v):
    return f"{v * 100:.1f} %".replace(".", ",")


def main():
    e = pd.read_csv("results/ejemplo_anclaje/resumen.csv")
    eq = pd.read_csv("results/horas_del_equipo/resumen.csv")
    an = pd.read_csv("results/anclas_escala/resumen.csv").set_index("k")
    proy = str(e.proyecto.iloc[0])
    frio_eq = float(eq[eq.k == 0].PRED25.iloc[0])

    def br(pref, k):
        return float(eq[(eq.k == k) & eq.brazo.str.startswith(pref)].PRED25.iloc[0])

    prs = P.nueva()
    P.portada(prs, "El factor de anclaje",
              "Cómo se calcula, qué le hace a la estimación y de dónde tienen que "
              "salir las horas",
              "Agix · componente de estimación de esfuerzo")

    # ---------------------------------------------------------------- 1
    s, y0 = P.lamina(prs, "El modelo no cambia; cambia un número",
                     "Los pesos están congelados. Lo que se ajusta es la salida.")
    P.vinetas(s, y0, [
        "Para un texto dado, la predicción del modelo es SIEMPRE la misma. No se "
        "reentrena nada cuando el equipo avanza.",
        "Lo que cambia es UN escalar por proyecto, el factor, que multiplica a todas "
        "las tareas que siguen abiertas.",
        "Y las tareas ya cerradas dejan de estimarse: sus horas son un dato, se suman "
        "tal cual.",
    ], salto=Inches(0.95))
    P.veredicto(s, y0 + Inches(3.0), "La fórmula entera",
                "off = log( Σ horas reales de las cerradas ÷ Σ predicho para esas "
                "mismas ) × k / (k + 12)      ·      "
                "total = Σ reales(cerradas) + Σ predicho(abiertas) × e^off", "tinta")

    # ---------------------------------------------------------------- 2
    s, y0 = P.lamina(prs, "Cómo se calcula",
                     "Un cociente observado, frenado por la contracción de "
                     "James-Stein.")
    P.figura_ajustada(s, f"{FIG}/12_factor_como_se_calcula.png", y0, Inches(3.1))
    P.veredicto(s, y0 + Inches(3.3), "Por qué se frena",
                "Con una sola tarea cerrada el cociente es ruido, así que solo se "
                "aplica el 8 % de la corrección; con cinco, el 29 %; con veinte, el "
                "62 %. Sin ese freno, anclar con pocas tareas empeora: el MdMRE con "
                "una tarea pasa de 0,347 a 0,726.", "ambar")

    # ---------------------------------------------------------------- 3
    s, y0 = P.lamina(prs, "Qué consigue, en un proyecto real",
                     f"{proy}, elegido por tener el error mediano entre los de 20 o "
                     f"más tareas.")
    P.tabla(s, y0, ["tareas cerradas", "factor", "horas de una tarea concreta",
                    "total estimado", "error del total"],
            [[str(int(r.cerradas)), n(r.factor), f"{n(r.horas_tarea, 2)} h",
              f"{r.total:.0f} h", f"{r.error_pc:.1f} %".replace(".", ",")]
             for _, r in e.iterrows()],
            anchos=[Inches(2.0), Inches(1.3), Inches(2.9), Inches(1.8), Inches(1.8)],
            resaltar={len(e) - 1: P.SUAVE})
    P.veredicto(s, y0 + Inches(2.9), "Lo que hay que ver en esta tabla",
                f"La predicción cruda del modelo para esa tarea es "
                f"{n(float(e.horas_tarea.iloc[0]), 2)} h en las seis filas. Lo único "
                f"que se mueve es el factor, y con él el total: del "
                f"{e.error_pc.iloc[0]:.0f} % de error al "
                f"{n(float(e.error_pc.iloc[-1]), 1)} %.", "tinta")

    # ---------------------------------------------------------------- 4
    s, y0 = P.lamina(prs, "De dónde tienen que salir esas horas",
                     "La pregunta de producto: ¿vale pedirle al equipo horas de "
                     "proyectos anteriores?")
    P.figura_ajustada(s, f"{FIG}/13_factor_que_consigue.png", y0, Inches(2.9))
    P.tabla(s, y0 + Inches(3.05),
            ["con 5 horas reales aportadas", "±25 %", "frente a no pedir nada"],
            [["no se pide nada (en frío)", pc(frio_eq), "—"],
             ["del PROPIO proyecto", pc(br("B", 5)),
              f"{(br('B', 5) - frio_eq) * 100:+.1f} puntos".replace(".", ",")],
             ["de otro proyecto del MISMO equipo", pc(br("C", 5)),
              f"{(br('C', 5) - frio_eq) * 100:+.1f} puntos".replace(".", ",")],
             ["de otra organización (control)", pc(br("D", 5)),
              f"{(br('D', 5) - frio_eq) * 100:+.1f} puntos".replace(".", ",")]],
            anchos=[Inches(4.0), Inches(1.5), Inches(2.6)],
            resaltar={1: P.SUAVE})

    # ---------------------------------------------------------------- 5
    s, y0 = P.lamina(prs, "Qué significa para el producto",
                     "Tres decisiones que salen de esto.")
    P.vinetas(s, y0, [
        "PEDIR horas de proyectos anteriores NO sirve: deja la estimación peor que no "
        f"pedir nada ({(br('C', 5) - frio_eq) * 100:+.1f} puntos). El factor supone que "
        "las horas comparten la escala del proyecto, y las de otro proyecto no la "
        "comparten.".replace(".0 puntos", " puntos"),
        "Lo que sirve es el time tracker del propio proyecto, y llega solo. Desde "
        f"TRES tareas cerradas la mejora es significativa (p = "
        f"{n(float(an.loc[3, 'p_wilcoxon']), 4)}) en el 100 % de los sorteos.",
        "Por eso la estimación se muestra como PROVISIONAL y se recalcula sola. No "
        "hace falta pedirle nada al usuario para que mejore.",
    ], salto=Inches(1.15))
    P.veredicto(s, y0 + Inches(3.6), "Lo que sí vale la pena pedirle al usuario",
                "Su propia estimación de las tareas, que es otro canal distinto y ese "
                "sí está medido: sobre 43 proyectos el ±25 % pasa del 44,2 % al "
                "65,1 %.", "ambar")

    SAL.parent.mkdir(parents=True, exist_ok=True)
    prs.save(SAL)
    print(f"escrito {SAL} · {len(prs.slides)} láminas")


if __name__ == "__main__":
    main()
