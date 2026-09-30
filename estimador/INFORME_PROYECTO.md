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
| **en frío** | **459** | **1104** | **0,372** | **0,815** | **36,8 %** | 60,5 % | **71,9 %** | +3,7 % |
| con 1 tarea cerrada | 423 | 922 | 0,338 | 0,842 | 41,2 % | 65,8 % | 74,6 % | +0,2 % |
| con 3 tareas cerradas | 366 | 824 | 0,283 | 0,881 | 44,7 % | 71,9 % | 80,7 % | −2,7 % |
| **con 5 tareas cerradas** | **329** | **748** | **0,263** | **0,909** | **48,3 %** | 77,2 % | **85,1 %** | −0,5 % |
| con 10 tareas cerradas | 284 | 717 | 0,223 | 0,919 | 51,8 % | 80,7 % | 88,6 % | +0,1 % |
| con 20 tareas cerradas | **169** | **336** | **0,174** | 0,912 | **63,2 %** | 79,8 % | 87,7 % | +1,3 % |

Por tamaño de proyecto, en frío:

| tareas | proyectos | MdMRE | ±25 % | factor 2 | MdMRE del constante |
|---|---|---|---|---|---|
| 8–15 | 29 | 0,495 | 31,0 % | 62,1 % | 0,712 |
| 16–40 | 18 | 0,458 | 33,3 % | 66,7 % | 0,762 |
| 41–120 | 31 | 0,321 | 45,2 % | 61,3 % | 0,738 |
| 121–400 | 23 | 0,360 | 26,1 % | 91,3 % | 0,656 |
| más de 400 | 13 | **0,232** | **53,8 %** | **92,3 %** | 0,607 |

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
Mejora MdMRE (0,438 → 0,372), MAE, RMSE, ±50 % y factor 2 a la vez, y deja el
±25 % prácticamente igual.

### 2 · Anclar con el cociente de sumas, no con la mediana de cocientes (E53)

Cuando el proyecto ya tiene tareas cerradas, el cronómetro da horas reales y se
puede corregir el desplazamiento del proyecto. Pero **el estadístico importa**: si
se estima con la mediana de log(real/estimado), y el estimador predice la media,
esa mediana es sistemáticamente negativa con una distribución asimétrica. Anclando
así, cuantas más tareas cerradas **peor**: con 20 tareas el sesgo llegaba a −45,7 %
y el ±25 % caía a 26,3 %.

Para un total lo que corresponde es el **cociente de sumas**. Con eso, 20 tareas
cerradas dan ±25 % del 63,2 % en vez del 26,3 %.

La contracción de James-Stein (λ = 12) sigue siendo imprescindible: con 1 o 2
tareas el desplazamiento crudo es ruido.

### 3 · Por qué sumar funciona, y por qué no llega a cero

**A favor:** los errores de tareas distintas se cancelan al sumar. El MdMRE pasa de
0,741 por tarea a 0,372 por proyecto.

**En contra:** no son independientes. Hay un error **común a todo el proyecto** —el
modelo lo sitúa entero alto o bajo— que ninguna suma cancela. Por eso los proyectos
de más de 400 tareas se quedan en MdMRE 0,232 y no en cero, y por eso conocer una
sola tarea real ya mejora tanto.

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
| en frío (solo ascendente) | 459 | 1104 | 0,372 | 0,815 | 36,8 % | 71,9 % |
| **combinada con historial** | 468 | 1194 | **0,352** | **0,829** | 36,8 % | 71,1 % |

**Supuesto que hay que declarar:** la descendente usa los totales **reales** de otros
proyectos. Es realista para una organización con historial, pero el corpus mezcla
organizaciones, así que es una cota optimista de lo que daría dentro de una sola
empresa. En el arranque de Agix, sin proyectos entregados, solo vale la ascendente.

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
4. **La cifra en frío es conservadora.** El canal del codificador ajustado de extremo
   a extremo, que sí aporta (E35 brazo C), no entra en la validación cruzada por
   proyecto sin reajustarlo por pliegue, porque vio las etiquetas de los proyectos que
   cada pliegue deja fuera. El modelo desplegado lo lleva.

---

## Contexto de literatura

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
