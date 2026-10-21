"""Anade a DISENO_PIPELINE.md la seccion de vias medidas y descartadas.

Para quien implemente en produccion vale tanto como el diseno: evita gastar
semanas en caminos que ya estan cerrados con medida, y da los numeros para
defender las decisiones cuando alguien pregunte "¿y por que no probasteis...?".

Se inserta antes de B.11 Operacion y se renumera lo que venga detras.
"""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
DOC = Path("DISENO_PIPELINE.md")

SECCION = """## B.11 · Lo que se midió y se descartó

Esta sección existe para que nadie gaste tiempo en vías que ya están cerradas **con medida**.
Cada línea tiene su experimento detrás.

### Para mejorar la predicción — 12 vías

| vía | resultado |
|---|---|
| corrector de historial aprendido | pierde contra la regla fija |
| ponderar el ajuste por similitud de texto | gana 0 de 5 comparaciones |
| proyectos sintéticos | la dispersión de escala no se mueve |
| repeso por densidad (DenseWeight, SMOGN) | sin ganancia |
| preguntar al usuario la escala del equipo | no |
| estimar unas pocas tareas | no: hace falta ≥80 % del backlog |
| horas de proyectos del mismo equipo | de −1,8 a −3,7 puntos |
| sesgo de estimación del equipo | 11,6 % entre organizaciones: ruido |
| experto en tareas cortas | arregla la escala, no ordena (ρ = 0,058) |
| tipo, prioridad y descripción como rasgos | AUC de la compuerta 0,646 → 0,659 |
| entrenar para ordenar (learning to rank) | peor: ρ intra 0,286 contra 0,339 |
| descomponer la tarea en pasos (WBS) | ρ(nº de pasos, horas) = +0,012 |

**El denominador común.** El texto de una tarea no permite saber en qué régimen cae. Para
distinguir «≤2 h» de «>2 h» harían falta **AUC 0,80** solo para empatar con el modelo global, y
**seis señales independientes** convergen entre **0,644 y 0,662**: clasificador sobre el título,
la propia predicción del modelo, las dos combinadas, tipo y prioridad, la descripción, y todas
juntas.

La única señal que cruza el umbral es la **estimación del usuario** (AUC 0,915) — pero ahí la
mezcla de expertos tampoco hace falta: meter la estimación como una variable más del regresor da
41,3 % contra 38,4 % de enrutar. El regresor aprende el cambio de régimen solo.

### Para estrechar la banda por tarea en frío — 7 vías

acuerdo de los vecinos · cuantiles aprendidos (CQR) · quintil del valor predicho · intervalo
asimétrico · conjunto HPD contiguo · ventana más corta de la densidad condicional · desacuerdo
del ensemble.

**Todas negativas, y la causa está medida:** la correlación entre las señales de incertidumbre
del modelo y su error real va de **0,022** (desacuerdo del ensemble) a **0,042** (dispersión de
los vecinos). **El modelo no sabe cuándo se equivoca**, así que no hay nada sobre lo que
condicionar. No es mala calibración: la anchura es exactamente la que implica la dispersión del
residuo.

### Sobre transferir información entre proyectos — 4 medidas

| | resultado |
|---|---|
| horas de proyectos del mismo equipo | de −1,8 a −3,7 puntos |
| horas de otra organización (control) | de −2,8 a −8,1 puntos |
| sesgo de estimación del equipo | 11,6 % entre organizaciones |
| historial de los mismos desarrolladores, otro proyecto | SA 0,0299 → 0,0287 |
| **cerradas del propio proyecto** | **SA 0,0299 → 0,0355** |

El equipo **sí lleva algo** de señal —gana claramente al control de otra organización— pero
sigue siendo peor que no usar nada. Y el contraste que lo explica: el desplazamiento de una
**misma persona** correlaciona **+0,089** entre sus proyectos, mientras que **dos personas
distintas del mismo proyecto** correlacionan **+0,164**. **El proyecto une más que la persona.**

Por eso el diseño no guarda nada entre proyectos: cada uno arranca en frío y se calibra solo con
lo suyo. Simplifica el producto y evita prometer algo que no se cumple.

---

## B.12 · Dos trampas de medición que conviene no repetir

Las dos mordieron en este proyecto y las dos son fáciles de repetir.

### 1 · Comparar bandas a cobertura distinta

Una banda que cubre el 75 % parece más estrecha que una honesta del 80 %. Pasó **tres veces**:
con la banda del total, con la ventana adaptativa por tarea y con la calibración por tramo. Las
tres veces la primera lectura fue equivocada.

**Protocolo correcto:** ajustar el nivel *nominal* de cada método hasta que la cobertura *real*
sea 0,800 exacta, y solo entonces comparar anchuras. Al corregirlo, la ventana adaptativa que
parecía ganar un 7 % resultó perder, y la contracción del cuantil del tramo, que parecía neutra,
resultó valer un 31 % de anchura con una tarea cerrada.

### 2 · El artefacto de medición de `HoursEstimate`

En el **34,7 %** de las tareas con estimación del usuario, las horas reales son **la estimación
copiada**: la tarea se cierra imputando lo estimado en vez de lo medido. Está muy desigual por
fuente (Spring 47,4 %, Jira 17,0 %, Sonatype 10,5 %).

Hay que quitar **las dos** contaminaciones, no solo una:

- **al evaluar**, porque acertar una fila copiada no mide nada;
- **al entrenar**, porque el corrector aprende «devuelve la estimación» de esas mismas filas.

Hecho bien, la palanca **aguanta**: sobre lo que queda abierto, 68,3 % contra 43,7 % del
anclaje. El inflado era de 5,5 puntos, no más. Y aparece una inversión que conviene conocer: en
datos limpios **el modelo gana a la estimación cruda** (23,8 % contra 21,8 %); con las filas
contaminadas dentro la comparación se invierte y engaña.

> Un detalle con el que no hay que confundirse: el brazo «en frío sobre limpias» sale *por
> debajo* del «en frío sobre todas». Las tareas con el real copiado son también las más fáciles
> para el modelo en frío. Quitarlas endurece el problema **para todos los brazos por igual**, así
> que la comparación sigue siendo justa.

---

"""


def main():
    s = io.open(DOC, encoding="utf-8").read()
    assert "## B.11 · Operación" in s, "no esta B.11 Operacion"
    # renumerar lo que venia detras
    s = s.replace("## B.11 · Operación", "## B.13 · Operación", 1)
    s = s.replace("## B.12 · Resumen de decisiones de diseño",
                  "## B.14 · Resumen de decisiones de diseño", 1)
    s = s.replace("## B.13 · Operación", SECCION + "## B.13 · Operación", 1)
    io.open(DOC, "w", encoding="utf-8").write(s)
    print(f"anadidas B.11 y B.12 · {DOC}: {len(s):,} caracteres")


if __name__ == "__main__":
    main()
