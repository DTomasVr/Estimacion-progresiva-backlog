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
| con 1 tarea cerrada | 421 | 914 | 0,341 | 0,844 | 39,5 % | 64,9 % | 75,4 % | −4,6 % |
| con 3 tareas cerradas | 365 | 816 | 0,281 | 0,882 | 45,6 % | 71,1 % | 79,8 % | −5,7 % |
| **con 5 tareas cerradas** | **326** | **736** | **0,256** | **0,910** | **48,3 %** | 74,6 % | **82,5 %** | −2,5 % |
| con 10 tareas cerradas | 281 | 702 | 0,219 | 0,920 | 51,8 % | 79,8 % | 88,6 % | −1,4 % |
| con 20 tareas cerradas | **170** | **336** | **0,186** | 0,913 | **63,2 %** | 79,0 % | 87,7 % | +0,5 % |

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
0,741 por tarea a 0,355 por proyecto.

**En contra:** no son independientes. Hay un error **común a todo el proyecto** —el
modelo lo sitúa entero alto o bajo— que ninguna suma cancela. Por eso los proyectos
de más de 400 tareas se quedan en MdMRE 0,236 y no en cero, y por eso conocer una
sola tarea real ya mejora tanto.

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
