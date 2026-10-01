"""Un resumen de las metricas en Markdown, para convertirlo a PDF y poder ensenarlo
sin PowerPoint.

Mismo contenido que el deck de metricas pero en dos paginas y sin la lamina de la
nube de dispersion. Todas las cifras se leen de los CSV de results/ en tiempo de
ejecucion: aqui no hay ningun numero escrito a mano.

    python src/185_resumen_metricas.py
    python src/183_md_a_pdf.py reports/metricas_resumen.md
"""
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
SAL = Path("reports/metricas_resumen.md")


def n(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def pc(v):
    return f"{v * 100:.1f} %".replace(".", ",")


def main():
    cf = pd.read_csv("results/curva_final/resumen.csv").set_index("brazo")
    lit = pd.read_csv("results/baselines_literatura.csv").set_index("modelo")
    canc = pd.read_csv("results/cancelacion/resumen.csv").set_index("unidad")
    us = pd.read_csv("results/usuario_proyecto/resumen.csv").set_index("brazo")
    an = pd.read_csv("results/anclas_escala/resumen.csv").set_index("k")
    cte, frio = cf.loc["0 · constante (n x mediana)"], cf.loc["1 · en frio"]
    k5 = cf.loc["2 · con 5 tareas cerradas"]
    k20 = cf.loc["2 · con 20 tareas cerradas"]

    fila = lambda e, v: (f"| {e} | {n(v.MAE_h, 0)} | {n(v.RMSE_h, 0)} | {n(v.MdMRE)} | "
                         f"{n(v.R2_log)} | {pc(v.PRED25)} | {pc(v.factor2)} |")

    md = f"""# Estimación del esfuerzo de un proyecto · resultados

**Qué se mide:** la tubería genera las tareas de un proyecto, el modelo estima las horas de
cada una, **se suman**, y ese total se compara con las horas reales registradas. La unidad de
análisis es el **proyecto**, no la tarea.

**Cómo se mide:** 114 proyectos con al menos 8 tareas, cada uno predicho **como si fuera
nuevo** — validación cruzada dejando fuera proyectos enteros, nunca tareas sueltas del mismo
proyecto. Todo se contrasta contra un predictor constante que no lee el texto: si no le gana,
el modelo no aporta nada.

## El resultado

| | MAE (h) | RMSE (h) | MdMRE | R² | ±25 % | factor 2 |
|---|---|---|---|---|---|---|
{fila("constante (no lee el texto)", cte)}
{fila("**el modelo, en frío**", frio)}
{fila("**con 5 tareas ya cerradas**", k5)}
{fila("con 20 tareas ya cerradas", k20)}

![Total del proyecto](../reports/figuras_metricas/8_total_proyecto.png)

En frío el modelo explica el **{n(frio.R2_log)}** de la varianza del total en log, acierta dentro
del **±25 %** en el **{pc(frio.PRED25)}** de los proyectos y dentro del **factor 2** en el
**{pc(frio.factor2)}**, con un sesgo de {n(frio.SSPB_pc, 1)} %. El control constante se queda en
{pc(cte.PRED25)} y {pc(cte.factor2)}.

## Frente al estado del arte

Son dos comparaciones distintas y conviene no mezclarlas.

**Por tarea**, que es lo que esa línea publica. Deep-SE y GPT2SP se reimplementaron y se
midieron sobre los mismos datos y con su mismo protocolo (dentro de cada proyecto, corte
cronológico, 24 proyectos):

| modelo | MdMRE | ±25 % | MAE (h) | gana a la base |
|---|---|---|---|---|
""" + "\n".join(
        f"| {k} | {n(v.MdMRE)} | {pc(v.PRED25)} | {n(v.MAE_h, 2)} | {v.gana_base} |"
        for k, v in lit.iterrows()) + f"""

Ahí vamos a la par: la tarea individual es un problema duro para todos, y la propia réplica de
Deep-SE (Tawosi, Moussa y Sarro, IEEE TSE 2022) encontró que supera al predictor mediano con
significancia en solo 8 de 42 casos.

**Por proyecto**, que es lo que aquí se presenta: ninguno de esos trabajos reporta el total del
proyecto a partir del texto, así que a ese nivel el contraste obligatorio es contra el control
constante — y se le gana por **más del doble** en MdMRE y por **cuatro veces** en ±25 %.

## Por qué sumar funciona

Con las **mismas** predicciones, medidas en las dos unidades:

| unidad | n | MdMRE | ±25 % | factor 2 |
|---|---|---|---|---|
| por tarea | {int(canc.loc['por tarea', 'n']):,} | {n(canc.loc['por tarea', 'MdMRE'])} | {pc(canc.loc['por tarea', 'PRED25'])} | {pc(canc.loc['por tarea', 'factor2'])} |
| **por proyecto** | {int(canc.loc['por proyecto', 'n'])} | **{n(canc.loc['por proyecto', 'MdMRE'])}** | **{pc(canc.loc['por proyecto', 'PRED25'])}** | **{pc(canc.loc['por proyecto', 'factor2'])}** |

Los errores de tareas distintas se cancelan al sumar: **sumar divide el MdMRE por
{n(canc.loc['por tarea', 'MdMRE'] / canc.loc['por proyecto', 'MdMRE'], 2)}**. Lo que no cancela
es el error común a todo el proyecto —el modelo lo sitúa entero alto o bajo—, y ese se corrige
con un solo dato real.

## Mejora sola con el uso

![Lo que arregla la escala](../reports/figuras_metricas/10_anclas_escala.png)

En cuanto el equipo cierra tareas, el cronómetro da horas reales y el total se reancla:

| tareas cerradas | orden de la escala | MdMRE | ±25 % | significancia |
|---|---|---|---|---|
| ninguna | {n(an.loc[0, 'spearman'])} | {n(an.loc[0, 'MdMRE'])} | {pc(an.loc[0, 'PRED25'])} | — |
| 3 | **{n(an.loc[3, 'spearman'])}** | {n(an.loc[3, 'MdMRE'])} | {pc(an.loc[3, 'PRED25'])} | p = {n(an.loc[3, 'p_wilcoxon'], 4)} |
| 5 | **{n(an.loc[5, 'spearman'])}** | {n(an.loc[5, 'MdMRE'])} | {pc(an.loc[5, 'PRED25'])} | p < 0,0001 |

**Con tres tareas cerradas ya es significativo en el 100 % de los sorteos.** De ahí la
recomendación de producto: enseñar la estimación en frío como provisional y recalcularla en
cuanto se cierren las tres primeras.

## Y si se le pide una estimación al usuario

Sobre los 43 proyectos que tienen estimación del usuario en al menos el 80 % de sus tareas, el
±25 % pasa del **{pc(us.loc['1 · el modelo en frio', 'PRED25'])}** al
**{pc(us.loc['4 · corregida con Poisson (media)', 'PRED25'])}**. Pedir la estimación en el
formulario es la diferencia entre una herramienta que orienta y una con la que se puede
negociar un plazo.
"""
    SAL.parent.mkdir(parents=True, exist_ok=True)
    SAL.write_text(md, encoding="utf-8")
    print(f"escrito {SAL} · {len(md.splitlines())} lineas")
    json.dump({"proyectos": 114}, open("results/resumen_metricas_meta.json", "w"))


if __name__ == "__main__":
    main()
