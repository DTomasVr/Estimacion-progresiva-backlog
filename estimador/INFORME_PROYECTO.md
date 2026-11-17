# La métrica de presentación: el TOTAL de horas del proyecto

**Estado al 30 de septiembre, 05:30.** Esta es la cifra que va a la exposición.
Sustituye a la sección equivalente de `INFORME_METRICAS.md`, que quedó anticuada
la misma noche.

Lo que recomendó la asesora y lo que miden los criterios del informe de
benchmarking (**MAE, RMSE, R²**): sumar las horas predichas de todas las tareas que
la pipeline genera para un proyecto y comparar ese total con el real.

Figura: `reports/figuras_metricas/8_total_proyecto.png` · Deck:
`entrega/presentaciones/6_metrica_de_proyecto.pptx`

---

## El resultado

**114 proyectos, todos predichos como si fueran nuevos** (validación cruzada
dejando fuera proyectos enteros, sin fuga). Mediana de 48 tareas y 358 h reales
por proyecto.

| historial del proyecto | MAE (h) | RMSE (h) | MdMRE | **R² log** | **±25 %** | ±50 % | **factor 2** | sesgo |
|---|---|---|---|---|---|---|---|---|
| constante (no lee el texto) | 886 | 2046 | 0,694 | 0,364 | 8,8 % | 22,8 % | 24,6 % | −203 % |
| **en frío** | **455** | **1090** | **0,355** | **0,817** | **36,8 %** | 60,5 % | **72,8 %** | −0,3 % |
| con 1 tarea cerrada | 428 | 995 | 0,344 | 0,847 | 36,9 % | 64,4 % | 75,6 % | −3,3 % |
| con 3 tareas cerradas | 394 | 954 | 0,326 | 0,882 | 39,7 % | 70,1 % | 80,3 % | −2,5 % |
| **con 5 tareas cerradas** | **379** | 991 | **0,291** | **0,911** | **43,7 %** | 75,7 % | **85,5 %** | +0,2 % |
| con 10 tareas cerradas | 336 | 891 | 0,266 | 0,917 | 47,9 % | 77,5 % | 87,7 % | +2,1 % |
| con 20 tareas cerradas | **291** | **795** | **0,219** | 0,908 | **54,3 %** | 78,4 % | 87,5 % | +1,5 % |

Las filas con tareas cerradas se promedian sobre 60 sorteos de **cuáles** fueron las
tareas que se cerraron primero. La dispersión entre sorteos es de ±0,025 en MdMRE y
±0,037 en el ±25 %: qué tareas toquen importa unos cuatro puntos.

Por tamaño de proyecto, en frío:

| tareas | proyectos | MdMRE | ±25 % | factor 2 | MdMRE del constante |
|---|---|---|---|---|---|
| 8–15 | 29 | 0,510 | 27,6 % | 62,1 % | 0,712 |
| 16–40 | 18 | 0,438 | 38,9 % | 72,2 % | 0,762 |
| 41–120 | 31 | 0,290 | 45,2 % | 58,1 % | 0,738 |
| 121–400 | 23 | 0,358 | 26,1 % | 95,7 % | 0,656 |
| más de 400 | 13 | **0,236** | **53,8 %** | 92,3 % | 0,607 |

Tres proyectos reales, el de **error mediano** de cada tramo (la mitad de su tramo
sale mejor y la mitad peor), con su intervalo al 80 %:

| tamaño | proyecto | tareas | estimado | real | error | intervalo al 80 % | dentro |
|---|---|---|---|---|---|---|---|
| pequeño | RedHat:WFCORE | 8 | 43 h | 87 h | −51 % | 12 a 113 h | sí |
| mediano | RedHat:FUSEDOC | 37 | 261 h | 183 h | +43 % | 100 a 495 h | sí |
| grande | JSWCLOUD | 226 | 1885 h | 1410 h | +34 % | 1008 a 2750 h | sí |

---

## Las tres cosas que hicieron falta para llegar aquí

### 1 · Entrenar un modelo PARA el total, no reciclar el de tareas (E52)

Toda la cadena estaba entrenada para acertar una tarea suelta: error cuadrático
sobre log-horas, que da la **mediana** condicional. Un total necesita la **media**.

Durante un tiempo eso se resolvió de segunda mano, empujando la mediana con el
estimador de smearing de Duan (1983) — factor 2,254, sin el cual el total salía al
44 % del real. Funcionaba, pero E45 avisó de que un solo árbol con pérdida de
Poisson le ganaba el MdMRE al apilado entero de cuatro canales. Cuando un modelo
suelto entrenado para el objetivo correcto le gana a un conjunto entrenado para el
objetivo equivocado, el problema es el objetivo.

El modelo del total es el mismo conjunto pero con **pérdida de Poisson sobre horas**:
bolsa de 5, canal de TF-IDF y canal de analogía, apilados también con Poisson.
Mejora MdMRE (0,438 → 0,355), MAE, RMSE, ±50 % y factor 2 a la vez, y deja el
±25 % prácticamente igual.

### 2 · Anclar bien, que tiene tres trampas (E53)

Cuando el proyecto ya tiene tareas cerradas, el cronómetro da horas reales y se
puede corregir el desplazamiento del proyecto. Pero **el estadístico importa**: si
se estima con la mediana de log(real/estimado), y el estimador predice la media,
esa mediana es sistemáticamente negativa con una distribución asimétrica. Anclando
así, cuantas más tareas cerradas **peor**: con 20 tareas el sesgo llegaba a −45,7 %
y el ±25 % caía a 26,3 %.

Para un total lo que corresponde es el **cociente de sumas**.

**Segunda trampa: las tareas ya cerradas se saben, no se estiman.** Si el equipo
cerró tres tareas, sus horas son un dato. El total es lo conocido más la estimación
corregida de lo que queda; sumar la estimación de lo ya cerrado es tirar información.

**Tercera trampa, y la que más costaba: hay que promediar la MÉTRICA, no la
predicción.** Cuáles son las tres tareas que se cierran primero es un sorteo, así que
hay que repetirlo. Pero promediar las *predicciones* de 60 sorteos y medir sobre esa
media cancela la varianza del sorteo y da lo que se conseguiría con 60 × k tareas, no
con k. Con ese fallo, λ = 0 y 3 tareas daban un MdMRE de 0,08 y un ±25 % del 92 %,
que es imposible; ahí se detectó. Corregido, la curva baja entre 4 y 9 puntos
respecto a la que llegué a escribir.

**La contracción de James-Stein (λ = 12) está bien calibrada.** Se barrió λ de 0 a 24
(E62): sin contracción el MdMRE con una tarea cerrada es 0,726 en vez de 0,347, y
elegir λ dejando fuera cada proyecto es **peor** que dejarlo en 12 para todos los k.
El valor heredado del anclaje por tarea resultó ser el correcto también aquí.

### 3 · Por qué sumar funciona, y por qué no llega a cero

**A favor:** los errores de tareas distintas se cancelan al sumar. Con las **mismas**
predicciones (E73, `src/181_cancelacion_al_sumar.py`), el MdMRE pasa de **1,370 por
tarea a 0,355 por proyecto**: sumar lo divide por 3,86. Que la cabeza de la media
salga mal por tarea es deliberado —la media está por encima de la mediana en una
distribución asimétrica— y es justo lo que obliga a llevar dos cabezas.

**En contra:** no son independientes. Hay un error **común a todo el proyecto** —el
modelo lo sitúa entero alto o bajo— que ninguna suma cancela. Por eso los proyectos
de más de 400 tareas se quedan en MdMRE 0,236 y no en cero, y por eso conocer una
sola tarea real ya mejora tanto.

---

## Dónde falla, y por qué (E61)

El MdMRE de 0,355 es una mediana: hay proyectos que fallan por factores de cinco.
No están repartidos al azar. Agrupando por el **tamaño típico de tarea** del
proyecto:

| mediana de horas por tarea | proyectos | MdMRE | ±25 % | factor 2 | sesgo |
|---|---|---|---|---|---|
| menos de 0,5 h | 6 | **4,36** | 0 % | 0 % | ×5,36 |
| 0,5–1 h | 18 | 0,708 | 16,7 % | 61,1 % | ×1,71 |
| 1–2 h | 34 | 0,373 | 35,3 % | 76,5 % | ×1,31 |
| **2–4 h** | 29 | **0,193** | **65,5 %** | **93,1 %** | ×0,89 |
| más de 4 h | 27 | 0,386 | 29,6 % | 70,4 % | ×0,61 |

Los ocho proyectos que peor salen son **todos** de JiraPublico, con medianas de
0,5 a 2 h por tarea y hasta un 37 % de tareas por debajo de 0,25 h. Un proyecto
que registra 57 tareas con mediana de 0,5 h no está midiendo esfuerzo de
desarrollo.

**La lectura correcta no es "el dato es malo", es que el modelo comprime la escala
del proyecto**: a los que van a 0,5 h por tarea les predice ~3 h, y a los que van a
7 h les predice ~4 h. Es la misma compresión que se ve por tarea, ahora visible por
proyecto. Y explica por qué una sola tarea cerrada ya mejora tanto: es la única
información sobre la escala que el texto no da.

Quitando los 14 proyectos con menos de 1 h por tarea, el MdMRE pasa de 0,355 a
0,327 y el ±25 % del 36,8 % al 41,0 %. **Eso no es una mejora del modelo**, es medir
sobre otro subconjunto; sirve solo para saber cuánto de la cifra se lleva el ruido
de registro.

---

## Lo que sí arregla la escala: tareas cerradas (E67–E68)

**Esta es la conclusión operativa de toda la sección.** Figura:
`reports/figuras_metricas/10_anclas_escala.png`

Anclar con tareas ya cerradas ataca directamente lo que estaba roto — el orden de
la escala — y, a diferencia de todo lo demás, **se puede demostrar**:

| tareas cerradas | orden de la escala (Spearman) | dispersión | pendiente | MdMRE | ±25 % | factor 2 | Wilcoxon vs frío |
|---|---|---|---|---|---|---|---|
| ninguna | 0,444 | 0,271 | 1,374 | 0,355 | 36,8 % | 72,8 % | — |
| 1 | 0,549 | 0,338 | 1,362 | 0,346 | 37,6 % | 75,3 % | p = 0,015 (73 % de sorteos) |
| 2 | 0,622 | 0,399 | 1,291 | 0,333 | 38,4 % | 78,0 % | p = 0,0026 (92 %) |
| **3** | **0,700** | 0,448 | 1,268 | 0,319 | 40,7 % | 80,8 % | **p = 0,0003 (100 %)** |
| **5** | **0,769** | 0,519 | 1,197 | 0,292 | 43,2 % | 84,4 % | **p < 0,0001 (100 %)** |
| 10 | 0,780 | 0,553 | 1,140 | 0,260 | 48,4 % | 87,4 % | p < 0,0001 (100 %) |

*(la dispersión real es 0,791; "% de sorteos" es en cuántos de los 60 sorteos de qué
tareas se cerraron primero el contraste sale por debajo de 0,05)*

**Con cinco tareas cerradas el orden pasa de 0,44 a 0,77.** No queda perfecto —la
contracción deja algo de compresión a propósito, y la pendiente se queda en 1,20—
pero es otra cosa. Y desde **dos o tres tareas** la mejora del total es significativa
en prácticamente todos los sorteos.

**Recomendación de producto:** mostrar la estimación como provisional y recalcularla
en cuanto el equipo cierre las primeras tres tareas. No hace falta esperar a veinte.

### Y una mejora que no se puede demostrar (E68)

Contraer el anclaje hacia el **modelo de escala** de E66 en vez de hacia cero —
James-Stein hacia un centro informado en lugar de hacia el origen— es mejor en casi
todo y tiene sesgo cero por construcción:

| tareas cerradas | MdMRE hacia cero → hacia el modelo | ±25 % | MAE (h) |
|---|---|---|---|
| ninguna | 0,355 → 0,355 | 36,8 % → **42,1 %** | 455 → **423** |
| 1 | 0,346 → **0,335** | 37,6 % → 40,1 % | 423 → 404 |
| 3 | 0,325 → **0,310** | 38,9 % → 42,1 % | 396 → 383 |
| 5 | 0,287 → 0,288 | 44,1 % → 44,2 % | 377 → 370 |

Consistente en dirección en los siete valores de k, pero **no significativa en
ninguno por separado**: en frío, Wilcoxon p = 0,186 y McNemar del ±25 % p = 0,263
(13 mejoran, 7 empeoran), con IC 95 % [−0,026, +0,123]. Se documenta, no se
presenta como mejora.

---

## ¿Se puede arreglar la compresión de la escala sin tareas cerradas? (E65–E66)

Sí se puede corregir la escala. **No se nota en el total.**

El orden no es malo: el Spearman entre la escala predicha y la real es **0,444**. El
modelo sí sabe qué proyecto es más grande; solo se queda corto de recorrido. Eso
abre tres vías, todas de la literatura:

| método | dispersión | error de escala | pendiente |
|---|---|---|---|
| sin tocar | 0,271 | 0,382 | 1,374 |
| mapeo de cuantiles (corrección de sesgo, climatología) | **0,781** | 0,447 | 0,428 |
| calibración lineal en log | 0,369 | **0,347** | 0,934 |
| cambio de etiqueta (BBSE, Lipton et al. 2018) | 1,828 | 1,181 | 0,181 |
| *(real, referencia)* | *0,791* | — | *1,0* |

El **mapeo de cuantiles** deja la dispersión clavada (0,781 frente a 0,791) pero
empeora el error puntual: es el canje conocido entre calibrar la distribución y
acertar cada caso. El **cambio de etiqueta** se pasa de largo — con 8 a 20 tareas por
proyecto, el reparto observado de predicciones es demasiado ruidoso para invertir la
matriz de confusión.

Mejor camino: **predecir la escala como problema propio**, con rasgos de proyecto
(vector medio de sus tareas + agregados), en vez de promediar predicciones por tarea
ya aplastadas:

| | error de escala | dispersión | **Spearman** | R² | pendiente |
|---|---|---|---|---|---|
| la media de las predicciones | 0,382 | 0,271 | 0,444 | 0,182 | 1,374 |
| vector medio + agregados | 0,382 | 0,346 | 0,469 | 0,224 | 1,085 |
| **+ la media como rasgo** | **0,369** | 0,356 | **0,481** | **0,251** | **1,121** |

Mejora el orden y la pendiente, que es justo lo que estaba roto.

**Y aun así, en el total no se nota:** MdMRE 0,355 → 0,336, MAE 455 → 410,
±25 % 36,8 % → 39,5 %. El contraste dice que eso es ruido:

- Wilcoxon pareado **p = 0,441**, con mejora en 62 de 114 proyectos
- McNemar del ±25 % **p = 0,711** (16 mejoran, 13 empeoran)
- IC 95 % de la diferencia de error **[−0,094, +0,105]**, que cruza el cero

Con 114 proyectos no hay potencia para ver una mejora de ese tamaño. **No se
presenta como mejora.** Lo que sí está establecido, y con significancia, es que
**una sola tarea cerrada arregla la escala**: es la única información sobre ella que
el texto no da.

*(Nota metodológica: la primera versión ajustaba el SVD del vector de proyecto con
los 114 proyectos incluido el evaluado. Con el SVD dentro del pliegue, el Spearman
baja de 0,521 a 0,481 y el ±25 % de 43,0 % a 39,5 %.)*

### Y tampoco ayuda entrenar la escala con Apache (E69)

Tenía buena pinta: Apache aporta **186 proyectos más** (300 en total) y un recorrido
de escala **más ancho** (sd 1,024 frente a 0,791), que es justo lo que le falta a un
modelo comprimido.

| | error de escala | dispersión | **Spearman** | R² | pendiente |
|---|---|---|---|---|---|
| la media de las predicciones | 0,382 | 0,271 | 0,444 | 0,182 | 1,374 |
| modelo de proyecto, **sin** Apache | 0,399 | 0,350 | **0,487** | 0,235 | 1,128 |
| modelo de proyecto, **con** Apache | 0,504 | 0,514 | 0,315 | −0,006 | 0,502 |
| con Apache + la media como rasgo | 0,546 | 0,553 | 0,373 | 0,058 | 0,564 |

Apache **ensancha la dispersión** (0,35 → 0,51) pero **destroza el orden** (0,487 →
0,315). La relación entre el texto y la escala es distinta allí: sus convenciones de
registro no transfieren. Misma conclusión que E38 sacó para la predicción por tarea.

### ¿Y si las tareas que se cierran primero no son una muestra al azar? (E70)

Figura: `reports/figuras_metricas/11_robustez.png`

Toda la curva de anclaje supone que las *k* tareas ya cerradas son un sorteo
aleatorio. En un equipo real eso es dudoso: se suele empezar por lo fácil, y si las
primeras en cerrarse fuesen sistemáticamente las más cortas, el desplazamiento
estimado saldría bajo y el total se quedaría corto. Sería un sesgo que la evaluación
no ve y el producto sí sufriría.

No hace falta suponer nada, porque el dataset guarda la fecha de cierre:
**82 proyectos** la tienen en al menos el 80 % de sus tareas, así que se puede
usar el **orden real**.

| con 3 tareas cerradas | sesgo del total (SSPB) | MdMRE | ±25 % |
|---|---|---|---|
| al azar (lo que supone la curva) | −0,3 % | 0,359 | 36,7 % |
| **el orden REAL de cierre** | **−1,6 %** | **0,335** | **40,2 %** |
| las que el modelo cree cortas | +5,9 % | 0,414 | 39,0 % |
| lo más corto primero (el peor caso) | −80,5 % | 0,506 | 19,5 % |
| lo más largo primero | +60,4 % | 0,604 | 26,8 % |

**El orden real no es una amenaza.** Con tres cerradas el sesgo sigue siendo
despreciable (−1,6 %, frente a −0,3 % del sorteo: menos de dos puntos de diferencia)
y la exactitud **mejora**: el MdMRE baja de 0,359 a 0,335 y el ±25 % sube del
36,7 % al 40,2 %. Los equipos de este corpus no despachan lo más corto primero; al
contrario, el orden con que cierran de verdad ayuda algo.

**Pero el escenario que sí lo rompería queda medido**, y hay que decirlo: si un equipo
cerrase de verdad las más cortas primero, el total saldría un −80,5 % con tres cerradas
y hasta −166,4 % con diez. Es la única condición bajo la cual el anclaje engaña, y es
observable en producción: basta vigilar si las primeras tareas cerradas de un
proyecto quedan sistemáticamente por debajo de la mediana del equipo. La variante
realista —cerrar primero lo que el modelo *cree* corto, que es cómo elige un equipo,
por tamaño percibido y no por el real— da +5,9 %, inofensivo.

---

### Y la réplica independiente lo cierra (E71)

Con 114 proyectos la potencia es del 20 %, así que "no significativo" podía
significar simplemente "muestra pequeña". Para resolverlo **no hay que generar
proyectos sintéticos** —heredarían las relaciones que el modelo ya aprendió y
darían p-valores buenos y falsos— sino replicar en datos reales independientes: los
**186 proyectos de Apache**, con su propio pipeline montado de cero.

Para que el Fisher signifique algo, los dos p-valores tienen que salir del mismo
procedimiento, así que el del corpus propio se vuelve a medir con el guion espejo
(**E72**, `src/180_contraste_corpus.py`), que reproduce exactamente la métrica en
frío publicada (MAE 455,30 · MdMRE 0,3552 · ±25 % 36,84 %).

| corpus | proyectos | ±25 % en frío → corregido | McNemar | error en log | Wilcoxon |
|---|---|---|---|---|---|
| el corpus de PBE (E72) | 114 | 36,0 % → 36,8 % | p = 1,00 (12-11) | 0,394 → **0,397** | p = 0,083 |
| Apache, independiente (E71) | 186 | 23,7 % → 26,9 % | p = 0,31 (15-9) | 0,541 → **0,590** | p = 0,77 |
| **las dos juntas, con Fisher** | **300** | | **p = 0,67** | | **p = 0,24** |

**No hay efecto**, ni por separado ni combinando. En los dos corpus el error mediano
en log empeora al corregir, y en Apache solo mejora en 95 de 186 proyectos: la
mitad justa. Y esto cierra la cuestión mucho mejor que "no se pudo demostrar con 114
proyectos": se probó en dos corpus reales e independientes y no aparece.

*(Aquí hubo **dos fallos propios**, los dos detectados al revisar y los dos corregidos.
El primero: la primera versión del contraste le quitaba el sesgo solo al brazo
corregido, y en Apache el brazo en frío tiene un sesgo del +41 %. Así salía Wilcoxon
p = 0,045 y Fisher combinado p = 0,048 —significativo—, pero esa "mejora" era quitar
el sesgo, no ordenar mejor; con los dos brazos centrados, nada. El segundo: el
p-valor del corpus propio que entraba en el Fisher estaba escrito a mano dentro de
E71 (0,1862) en vez de leído de un fichero, y además era el de otro contraste
distinto —el de E68, encoger hacia el modelo—, así que el Fisher combinaba dos cosas
que no eran comparables. Medido con el procedimiento idéntico al de Apache sale
0,0833, el Fisher pasa de 0,42 a 0,24, y el cálculo vive ahora en un solo sitio
que lee los dos ficheros.)*

**En cambio la compresión sí se replica, y es peor:**

| corpus | sd predicha | sd real | razón | Spearman |
|---|---|---|---|---|
| sin Apache (114 proyectos) | 0,271 | 0,791 | 34 % | 0,444 |
| **Apache (186 proyectos)** | 0,192 | 1,024 | **19 %** | **0,108** |

En Apache el modelo apenas puede ordenar los proyectos por escala. El diagnóstico
es general y más severo allí, lo que encaja con que Apache mezcle equipos con
prácticas de registro mucho más dispares.

**Con esto, la vía de arreglar la compresión sin tareas cerradas queda cerrada:** se
probaron mapeo de cuantiles, calibración lineal, cambio de etiqueta, un modelo de
escala propio, triplicar la muestra con Apache y una réplica independiente en un
segundo corpus. Lo único que funciona y se puede demostrar son las tareas cerradas.

---

## El hilo que recorre toda la sección: mediana frente a media

Tres veces, en tres sitios distintos, el mismo fallo:

| dónde | el componente predecía | lo que hacía falta | lo que costaba |
|---|---|---|---|
| sumar las tareas (E43) | la mediana condicional | la media | el total salía al 44 % del real |
| anclar con tareas cerradas (E53) | la mediana de los cocientes | el cociente de sumas | con 20 tareas, sesgo −45,7 % |
| corregir la estimación del usuario (E58) | la mediana del factor | la media | sesgo −31,2 % y peor que no corregir |

**La regla, en una frase: en cuanto algo se va a SUMAR, el componente que lo
produce tiene que estimar la media, no la mediana.** Todo el modelo estaba
entrenado con error cuadrático sobre log-horas, que da la mediana, porque estaba
pensado para acertar una tarea suelta. Pasar a nivel de proyecto obliga a revisar
cada pieza de la cadena, y las tres que no se revisaron fallaron por el mismo
motivo.

---

## Qué aporta pedirle la estimación al usuario (E58)

Medido sobre los **43 proyectos** que tienen estimación de usuario en al menos el
80 % de sus tareas:

| | MAE (h) | MdMRE | R² | **±25 %** | ±50 % | sesgo |
|---|---|---|---|---|---|---|
| constante | 1040 | 0,738 | 0,171 | 4,7 % | 16,3 % | −282 % |
| el modelo en frío | 586 | 0,332 | 0,921 | 44,2 % | 69,8 % | −5,9 % |
| estimación del usuario en crudo | 302 | **0,167** | 0,957 | 62,8 % | 86,0 % | +12,1 % |
| corregida con error cuadrático | 446 | 0,273 | 0,933 | 39,5 % | 90,7 % | −35,5 % |
| **corregida con Poisson** | **301** | 0,183 | **0,958** | **65,1 %** | **95,3 %** | −13,1 % |

**Decisión de producto:** el formulario debe pedir una estimación. Pasar de 44,2 %
a 65,1 % de proyectos dentro del ±25 %, y de 69,8 % a 95,3 % dentro del ±50 %, es
la diferencia entre una herramienta que orienta y una con la que se puede negociar
un plazo.

**Aviso obligado:** estos 43 proyectos son más fáciles que el conjunto de 114 (el
modelo en frío saca R² 0,921 aquí y 0,817 allí), así que **estas cifras no son
comparables con la tabla principal**. Lo comparable es la diferencia entre brazos
dentro de esta misma tabla.

---

## Lo que aporta la estimación descendente (E48)

De Jørgensen (2004): con equipos de profesionales, la estrategia **ascendente**
(descomponer y sumar) fue casi el doble de precisa que la **descendente** — MRE
medio 23 % y 31 % frente a 45 % y 55 %. Pero la condición para que la descendente
sirva es *recordar proyectos muy parecidos ya terminados*, y eso es justo lo que
sabe hacer un vector.

Representando cada proyecto por la media de los vectores de sus tareas y estimando
su total con los 5 proyectos más parecidos, y combinando las dos vías con pesos
elegidos dejando fuera cada proyecto:

| | MAE | RMSE | MdMRE | R² | ±25 % | factor 2 |
|---|---|---|---|---|---|---|
| en frío (solo ascendente) | 455 | 1090 | 0,355 | 0,817 | 36,8 % | 72,8 % |
| **combinada con historial** | 468 | 1194 | **0,352** | **0,829** | 36,8 % | 71,1 % |

**Supuesto que hay que declarar:** la descendente usa los totales **reales** de otros
proyectos. Es realista para una organización con historial, pero el corpus mezcla
organizaciones, así que es una cota optimista de lo que daría dentro de una sola
empresa. En el arranque de PBE, sin proyectos entregados, solo vale la ascendente.

---

## Robustez: ¿y si la pipeline genera el número equivocado de tareas? (E49)

La métrica se apoya en sumar sobre las tareas que genera la otra pipeline, y esa
pipeline no sabe cuántas tendrá el proyecto de verdad. Ese error **no está** en
ninguna de las cifras de arriba, porque en la evaluación las tareas son las reales.

Resultado, y es tranquilizador:

- Con horas **perfectas** y un 20 % de tareas de menos, el **95,6 %** de los
  proyectos sigue dentro del ±25 %: un error del 20 % en el conteo cabe dentro de la
  ventana.
- Con el modelo real y el conteo **exacto**, solo el 36,8 %.

**El error del modelo por tarea pesa mucho más que el del conteo.** La pipeline de
generación tendría que equivocarse en más de un 25 % del número de tareas para que
el conteo fuese el problema principal. Entre 0,9 y 1,1 veces el número correcto, el
factor 2 ni se mueve.

---

## Las cuatro preguntas incómodas, contestadas por delante

1. **La unidad de análisis son 114 proyectos, no 24.848 tareas.** Los intervalos son
   mucho más anchos de lo que sugiere el corpus, y los tramos por tamaño tienen entre
   13 y 31 proyectos. La no monotonía del tramo 121–400 es casi seguro ruido.
2. **El MMRE (media) es peor que el MdMRE** porque se lo comen dos o tres proyectos
   que fallan por mucho. Se reportan los dos.
3. **Tarea a tarea el problema sigue siendo difícil**: el tamaño del efecto frente al
   azar es pequeño (Δ de Glass 0,203). El valor del modelo está en el total del
   proyecto y en corregir la estimación del usuario, no en clavar una tarea suelta.
4. **El codificador ajustado de extremo a extremo no aporta al total.** Aporta a la
   predicción por tarea (E35 brazo C: R² del apilado 0,199 → 0,204), pero no se podía
   meter en la validación cruzada por proyecto sin reajustarlo por pliegue, porque vio
   las etiquetas de los proyectos que cada pliegue deja fuera. Se reajustó por pliegue
   (E54, 90 minutos de GPU) y el resultado es que **empeora** el total: MdMRE 0,355 →
   0,434. Así que la cifra en frío de arriba no tiene asteriscos ni está esperando una
   mejora escondida.

---

## Contexto de literatura

- **Sousa et al. (IEEE Access, 2023)**, *Applying Machine Learning to Estimate the
  Effort and Duration of Individual Tasks in Software Projects*. **Es el aval
  directo de este enfoque, y ya está en la bibliografía del informe de
  benchmarking** (es el que se cita para Random Forest). Estiman tareas
  individuales con ocho algoritmos sobre tres conjuntos de datos y obtienen MMRE
  por tarea de entre 0,11 y 9,45; pero **incluso en el peor conjunto, al agregar
  al nivel de proyecto obtienen MMRE 0,23**, y lo atribuyen explícitamente a la
  compensación de errores. O sea: que las tareas sueltas fallen y el total acierte
  no es una excusa nuestra, es un resultado conocido.

  Nuestro MMRE por proyecto en frío es 0,81 y el MdMRE 0,355, peores que su 0,23.
  La diferencia esperable es el corpus: ellos usan proyectos de organizaciones
  concretas con prácticas de registro homogéneas, y aquí hay 114 proyectos de
  fuentes distintas, algunos con tareas registradas a 0,05 h.

- **Jørgensen (2004)**, *Top-Down and Bottom-Up Expert Estimation*: la ascendente
  casi dobla en precisión a la descendente con equipos humanos; y la precondición de
  la descendente es recordar proyectos muy parecidos.
- **Tawosi, Moussa y Sarro (IEEE TSE, 2022)**, réplica de Deep-SE: el aprendizaje
  profundo supera al predictor mediano con significancia en solo **8 de 42 casos**.
  Su conclusión es que la similitud semántica no basta para distinguir historias por
  sus puntos.
- **Shepperd y MacDonell (2012)**: MMRE y PRED están sesgados; recomiendan MAE, la
  exactitud estandarizada SA frente al azar, y un tamaño del efecto.
- **Morley, Brito y Welling (2018)**: para datos positivos y sesgados, la exactitud
  simétrica mediana ζ y el sesgo porcentual simétrico SSPB.
- **Duan (1983)**: el estimador de smearing para retransformar de log a nivel.

---

## Scripts

| script | qué hace |
|---|---|
| `src/158_modelo_del_total.py` | el modelo entrenado para el total (Poisson apilado) |
| `src/159_curva_final.py` | la curva de tareas cerradas, con el anclaje correcto |
| `src/154_descendente.py` | estimación descendente por proyectos parecidos |
| `src/156_combinar_total.py` | la combinación de los seis estimadores |
| `src/155_sensibilidad_conteo.py` | robustez al número de tareas generadas |
| `src/148_total_proyecto.py` | la primera versión, con smearing |
| `src/141_figuras_metricas.py` | las ocho figuras |
| `src/152_pptx_exposicion.py` | el deck |
