"""Actualiza DISENO_PIPELINE.md (Parte B) con todo lo medido desde el 30/09.

El documento es el artefacto de traspaso: lo lee quien va a implementar en
produccion. Estaba congelado antes de E88-E119, asi que describe mecanismos que
han cambiado y da cifras que ya no son las que salen del paquete.

Se reescriben las secciones afectadas y se anade una nueva con lo que se probo
y NO funciona, que para quien implementa vale tanto como lo que si.

Cada reemplazo lleva su assert: si el documento cambia por otro lado, esto falla
en vez de escribir texto en el sitio equivocado.
"""
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
DOC = Path("DISENO_PIPELINE.md")


def pc(v, d=1):
    return f"{v * 100:.{d}f} %".replace(".", ",")


def n(v, d=3):
    return f"{v:.{d}f}".replace(".", ",")


def pon(txt, **kw):
    """Sustitucion por tokens @@clave@@: el texto lleva muchos '%' literales y
    el formateo con % se los come."""
    for k, v in kw.items():
        txt = txt.replace(f"@@{k}@@", str(v))
    assert "@@" not in txt, "quedan tokens sin sustituir"
    return txt


def main():
    s = io.open(DOC, encoding="utf-8").read()
    anc = pd.read_csv("results/anclas_escala/resumen.csv").set_index("k")
    rej = pd.read_csv("results/rejilla_valores/contra_literatura.csv").set_index("modelo")
    lit = pd.read_csv("results/baselines_literatura.csv").set_index("modelo")
    art = pd.read_csv("results/centrar_artefacto/resumen.csv").set_index("salida")
    ban = pd.read_csv("results/banda_que_cierra_antes/resumen.csv")
    abi = pd.read_csv("results/abierto_sin_contaminar/resumen.csv")
    ance = pd.read_csv("results/anclar_sobre_estimacion/resumen.csv")
    A2 = ban[ban.banda.str.startswith("A2")].set_index("k")
    D15 = ban[ban.banda.str.startswith("D · tramo contraido M=1500")].set_index("k")
    D6 = ban[ban.banda.str.startswith("D · tramo contraido M=600")].set_index("k")
    D2 = ban[ban.banda.str.startswith("D · tramo contraido M=200")].set_index("k")
    cor = art.loc["E · B + rejilla, SIN centrar"]
    hoy = art.loc["A · como se despliega hoy"]
    ab = abi.pivot(index="k", columns="brazo", values="PRED25")
    ae = ance.pivot(index="k", columns="brazo", values="PRED25")
    rep = []

    # ---------------------------------------------------------------- B.5.4
    viejo = s[s.index("### B.5.4 · Analogía y rejilla"):s.index("## B.6 · Los tres mecanismos")]
    nuevo = pon("""### B.5.4 · Analogía y rejilla

Para cada tarea se buscan los **25 vecinos más próximos** por coseno entre los 16.648 vectores
de entrenamiento (guardados en `float16`) y se toma la **mediana** de sus horas reales.

El objetivo está fuertemente cuantizado: la gente registra 1 h, 2 h, medio día. **Diez valores
cubren el 46,8 % del corpus.** Un modelo de regresión devuelve 1,83 h y no acierta ninguno de
los valores que la gente escribe de verdad, porque la mediana condicional cae *entre* dientes
del peine.

Por eso el `valor_probable` se **engancha siempre** a la rejilla de los **30 valores más
frecuentes**:

```
0,083 0,167 0,25 0,33 0,333 0,5 0,667 0,75 0,833 1 1,25 1,5 1,75 2 2,25
2,5 3 3,5 4 4,5 5 5,5 6 7 8 9 10 12 16 24
```

Dos decisiones, las dos medidas:

1. **Se ajusta siempre**, también cuando los vecinos no concuerdan y se cae a la predicción del
   apilado. Antes solo se ajustaba cuando concordaban, y eso dejaba sin ajustar a la mayoría.
2. **30 valores, no 41.** El umbral anterior (aparecer 60 veces) daba 41 valores y rendía medio
   punto menos: 16,59 % contra 17,11 % de ±25 %.

El umbral de acuerdo entre vecinos (**1,524**, que cubre el 85,3 % de las tareas) sigue
decidiendo si el valor base es la analogía o el apilado, pero ya no decide si se ajusta.

**Lo que gana ajustar**, sobre el bloque de prueba de 4.225 tareas:

| | MdMRE | ±25 % | factor 2 | MAE | SA | Spearman |
|---|---|---|---|---|---|---|
| sin ajustar | @@mdmre_sin@@ | @@p25_sin@@ | @@f2_sin@@ | @@mae_sin@@ | @@sa_sin@@ | @@rho_sin@@ |
| **ajustado** | @@mdmre_con@@ | **@@p25_con@@** | **@@f2_con@@** | @@mae_con@@ | @@sa_con@@ | @@rho_con@@ |

Sube el ±25 % y el factor 2 **sin mover el MAE, el SA ni el orden entre tareas**. No es un
modelo mejor: es la regla de decisión correcta para la métrica que se reporta.

---

""", mdmre_sin=n(rej.loc["nuestro, sin ajustar"].MdMRE),
           p25_sin=pc(rej.loc["nuestro, sin ajustar"].PRED25),
           f2_sin=n(rej.loc["nuestro, sin ajustar"].factor2),
           mae_sin=n(rej.loc["nuestro, sin ajustar"].MAE_h, 2),
           sa_sin=n(rej.loc["nuestro, sin ajustar"].SA, 4),
           rho_sin=n(rej.loc["nuestro, sin ajustar"].spearman),
           mdmre_con=n(rej.loc["nuestro, AJUSTADO a la rejilla"].MdMRE),
           p25_con=pc(rej.loc["nuestro, AJUSTADO a la rejilla"].PRED25),
           f2_con=n(rej.loc["nuestro, AJUSTADO a la rejilla"].factor2),
           mae_con=n(rej.loc["nuestro, AJUSTADO a la rejilla"].MAE_h, 2),
           sa_con=n(rej.loc["nuestro, AJUSTADO a la rejilla"].SA, 4),
           rho_con=n(rej.loc["nuestro, AJUSTADO a la rejilla"].spearman))
    s = s.replace(viejo, nuevo)
    rep.append("B.5.4 rejilla")

    # ------------------------------------------------------------ B.6 entero
    ini = s.index("## B.6 · Los tres mecanismos")
    fin = s.index("### B.6.4 · Enganche con el generador de backlog")
    nuevo_b6 = pon("""## B.6 · Los mecanismos que usan información del proyecto

> **Regla que atraviesa todo este apartado:** solo se usa información **del proyecto en curso**.
> Nada de proyectos pasados, ni del equipo, ni de la organización. No es una simplificación: se
> midió y **empeora** (ver B.11).

### B.6.1 · Anclaje con tareas cerradas — *el mecanismo más valioso del diseño*

Cuando el equipo ya cerró tareas, sus horas son **un dato**, no una estimación:

```python
k     = número de tareas cerradas
off   = log( Σ horas_reales(cerradas) / Σ predicción(cerradas) ) × k / (k + λ)
total = Σ horas_reales(cerradas)  +  Σ predicción(abiertas) × exp(off)
```

Cuatro decisiones, y **las cuatro costaron encontrarlas**:

1. **Cociente de sumas**, no mediana de cocientes. Se midió contra cuatro alternativas (media de
   logs, mediana de logs, Huber, sumas recortadas) y gana en todo `k`, con ventaja creciente:
   **54,8 % contra 43,1 %** de Huber en `k=20`. El motivo es explicable: ponderar por tamaño es
   lo correcto cuando el objetivo es una **suma**, y los estimadores robustos tratan igual a
   todas las tareas, tirando justo la información que decide el total.
2. **Las cerradas se saben, no se estiman.** Sumar la predicción de lo ya cerrado es tirar
   información.
3. **Contracción de James-Stein**, porque con `k` pequeño el cociente es ruidoso.
4. **λ es condicional**, y esto es lo que más fácil se implementa mal:

| situación | λ | por qué |
|---|---|---|
| sin estimación del usuario | **12** | el residuo tiene un 11 % de varianza entre proyectos que corregir |
| con estimación del usuario | **96** | tras la estimación solo queda un **1,9 %**, así que el anclaje casi no tiene nada que corregir y con λ bajo su ruido pesa más que su señal |

Con λ=12 encima de la estimación se **pierden 4–5 puntos**; con λ=96 se ganan **@@d96@@ de
media**, y hasta **+3,1** con 20 cerradas.

> **Error fácil de cometer, y caro:** si el desplazamiento se estima comparando el real de las
> cerradas contra la predicción **en frío**, y luego se aplica sobre la predicción **corregida**
> por la estimación, el resultado se hunde a 49–58 % frente al 68 % de no anclar. **El ancla hay
> que estimarla contra la misma base sobre la que se va a aplicar.**

**Efecto medido sobre el total** (114 proyectos, sin estimación del usuario):

| tareas cerradas | MdMRE | ±25 % | MAE (h) |
|---|---|---|---|
| ninguna | @@m0@@ | @@p0@@ | @@a0@@ |
| 3 | @@m3@@ | @@p3@@ | @@a3@@ |
| 5 | @@m5@@ | @@p5@@ | @@a5@@ |
| 10 | @@m10@@ | @@p10@@ | @@a10@@ |
| 20 | @@m20@@ | @@p20@@ | @@a20@@ |

**Y no depende de que el equipo cierre al azar.** Anclando en las tareas que de verdad se
cerraron primero (58 proyectos con fecha de cierre completa), el rendimiento es **+0,1 puntos**
frente al sorteo aleatorio. Lo medido al azar vale como medida de producción.

### B.6.2 · Corrección de la estimación del usuario — *la palanca más grande*

Si el planificador estima las tareas, se corrige con un HGB de **Poisson** sobre
`[vector e5 | log(estimación) | log(predicción en frío)]`.

**Tiene que ser el de Poisson.** El corrector de error cuadrático, que es el que sirve por
tarea, **empeora** la estimación cruda del usuario a nivel de proyecto (sesgo −35,5 %).

**Dos condiciones de producto que hay que respetar:**

- **Hace falta ≥80 % del backlog estimado.** Estimar 1, 3 o 5 tareas no mueve el total, y
  tampoco sirve elegirlas bien: ni un oráculo que elija sabiendo la respuesta pasa del 57 % de
  la ganancia con cinco tareas. Si el usuario va a estimar pocas, que sean **las de más horas
  previstas** (71 % de la ganancia con 20, contra 52 % al azar), pero sin prometer que con cinco
  basta.
- **Mejora el pronóstico de lo que FALTA**, no solo la aritmética de lo ya hecho. Es la única
  palanca de la que esto se ha comprobado:

| con `k` cerradas | ±25 % de lo abierto, solo anclaje | **con estimación** |
|---|---|---|
| 3 | @@ab3@@ | **@@ae3@@** |
| 5 | @@ab5@@ | **@@ae5@@** |
| 10 | @@ab10@@ | **@@ae10@@** |
| 20 | @@ab20@@ | **@@ae20@@** |

Esas cifras llevan el control de contaminación de B.11: evaluadas solo sobre tareas limpias y
con el corrector reentrenado solo con filas limpias.

### B.6.3 · Intervalo del total — *se estrecha con el avance*

La incertidumbre va **solo sobre lo que queda abierto**, porque las horas cerradas se saben:

```python
banda = Σ horas_reales(cerradas) + Σ predicción(abiertas) × exp(off) × exp([q_lo, q_hi])
```

Los cuantiles `q_lo` y `q_hi` se calibran **al número de tareas cerradas y al tramo de tamaño**
del proyecto, con el **cuantil conforme** (el estadístico de orden ⌈(m+1)(1−α)⌉, no el cuantil
muestral) y **contrayendo el cuantil del tramo hacia el global** con peso `n/(n+M)`, porque
calibrar cada tramo por separado con pocos datos cuesta anchura.

| tareas cerradas | anchura al 80 % |
|---|---|
| 1 | @@b1@@ |
| 2 | @@b2@@ |
| 3 | @@b3@@ |
| 5 | @@b5@@ |
| 10 | @@b10@@ |
| 20 | @@b20@@ |

Antes era **×10,61 fija**, no se estrechaba nunca, y con 20 cerradas cubría el 98 % prometiendo
el 80 %: anchura tirada.

> **Al medir bandas, igualar la cobertura REAL.** Es una trampa que mordió tres veces en este
> proyecto: una banda que cubre el 75 % parece más estrecha que una honesta del 80 %. El
> protocolo correcto es ajustar el nivel nominal de cada método hasta que la cobertura real sea
> 0,800 exacta, y solo entonces comparar anchuras.

### B.6.4 · Ajuste por persona — *solo en equipo, solo por tarea*

Opcional. Si se sabe quién hará la tarea y esa persona tiene cerradas **en este mismo
proyecto**, se aplica su propio desplazamiento, con el mismo mecanismo y λ=8.

La descomposición del residuo en tres niveles lo justifica:

| | |
|---|---|
| entre **proyectos** | 12,4 % |
| entre **desarrolladores** del mismo proyecto | **9,8 %** |
| dentro del desarrollador | 77,8 % |

Mejora el orden (Spearman 0,317 → 0,358), el SA (0,026 → 0,036) y el MAE (5,06 → 5,00 h).

**Dos límites que hay que respetar al implementar:**

- **No entra en el total.** Al sumar el proyecto entero los desplazamientos individuales se
  cancelan y empeoran el pronóstico de lo abierto en 1,1 puntos.
- **Solo historial del mismo proyecto.** El desplazamiento de una misma persona correlaciona
  **+0,089** entre sus proyectos; dos personas **distintas** del **mismo** proyecto, **+0,164**.
  El proyecto une más que la persona.

En un proyecto de una sola persona este ajuste y el del proyecto son el mismo número: es una
función de equipo.

### B.6.5 · Tira de probabilidad — *salida opcional del detalle*

`tira_probabilidad(textos)` devuelve la región de **densidad predictiva máxima**: los tramos de
horas donde se concentra el 80 % de la probabilidad, cada uno con su peso.

Es un **68 % más estrecha en medida** que la banda (×7,09 contra ×32) y reparte mucho mejor la
cobertura por tamaño de tarea (0,45–0,86 contra 0,25–1,00). Pero sale **fragmentada** —43 trozos
de mediana— y su **recorrido** es mayor, porque se queda con los dientes del peine y descarta
los huecos.

**No sustituye a la banda.** Para un «entre X e Y» la banda sigue siendo lo correcto; la tira es
la vista de detalle. Y recortarla a cinco trozos o menos **rompe la cobertura prometida**: es una
decisión de presentación legítima, pero hay que avisarlo.

""", d96=f"+{(ae['F96'] - ae['E']).mean() * 100:.1f} puntos".replace(".", ","),
           m0=n(anc.loc[0, "MdMRE"]), p0=pc(anc.loc[0, "PRED25"]),
           a0=n(anc.loc[0, "MAE_h"], 0),
           m3=n(anc.loc[3, "MdMRE"]), p3=pc(anc.loc[3, "PRED25"]),
           a3=n(anc.loc[3, "MAE_h"], 0),
           m5=n(anc.loc[5, "MdMRE"]), p5=pc(anc.loc[5, "PRED25"]),
           a5=n(anc.loc[5, "MAE_h"], 0),
           m10=n(anc.loc[10, "MdMRE"]), p10=pc(anc.loc[10, "PRED25"]),
           a10=n(anc.loc[10, "MAE_h"], 0),
           m20=n(anc.loc[20, "MdMRE"]), p20=pc(anc.loc[20, "PRED25"]),
           a20=n(anc.loc[20, "MAE_h"], 0),
           ab3=pc(ab.loc[3, "B' · anclaje, solo limpias"]),
           ab5=pc(ab.loc[5, "B' · anclaje, solo limpias"]),
           ab10=pc(ab.loc[10, "B' · anclaje, solo limpias"]),
           ab20=pc(ab.loc[20, "B' · anclaje, solo limpias"]),
           ae3=pc(ab.loc[3, "E3 · entrenado y evaluado LIMPIO"]),
           ae5=pc(ab.loc[5, "E3 · entrenado y evaluado LIMPIO"]),
           ae10=pc(ab.loc[10, "E3 · entrenado y evaluado LIMPIO"]),
           ae20=pc(ab.loc[20, "E3 · entrenado y evaluado LIMPIO"]),
           b1="×" + n(D15.loc[1, "anchura_mediana"], 2),
           b2="×" + n(D15.loc[2, "anchura_mediana"], 2),
           b3="×" + n(D6.loc[3, "anchura_mediana"], 2),
           b5="×" + n(D2.loc[5, "anchura_mediana"], 2),
           b10="×" + n(A2.loc[10, "anchura_mediana"], 2),
           b20="×" + n(A2.loc[20, "anchura_mediana"], 2))
    s = s[:ini] + nuevo_b6 + s[fin:]
    rep.append("B.6 completo")

    io.open(DOC, "w", encoding="utf-8").write(s)
    print("secciones reescritas: " + " · ".join(rep))
    print(f"{DOC}: {len(s):,} caracteres")


if __name__ == "__main__":
    main()
