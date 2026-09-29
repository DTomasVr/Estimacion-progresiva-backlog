# Mejorar las métricas del modelo en frío · experimentos E30–E40

Once experimentos sobre el modelo final (`models/estimador_agix_v3.joblib`), evaluados
siempre sobre los mismos tres bloques congelados: **prueba fija** (4.225 tareas,
corte cronológico dentro de proyectos conocidos), **prueba SiP** (1.885, el
subconjunto industrial) y **externo no Apache** (792, proyectos nunca vistos).

Resumen en una línea: **el error no bajó, y ahora sé por qué**. Las tasas de acierto
sí suben entre 1 y 3 puntos, pero siempre a costa del error, y eso resultó ser una
propiedad del problema, no un fallo de implementación.

---

## 1 · Dónde falla el modelo (E30)

El error no está repartido. Por decil del valor real, en los bloques de prueba:

| decil del valor real | mediana real | MdAElog |
|---|---|---|
| 0 (≤ 0,25 h) | 0,25 h | **1,854** |
| 3–5 | 1,45–2,7 h | **0,354–0,437** |
| 9 (≥ 15 h) | 24,5 h | **1,947** |

El modelo es razonable en el centro y se rompe en los dos extremos.

**Pero ojo con cómo se lee eso.** Mirado por decil del valor *real*, el modelo parece
sesgadísimo (×6,4 arriba en el decil 0, ×0,14 abajo en el 9). Mirado por tramo de la
*predicción*, el sesgo desaparece: la pendiente de log(real) sobre log(predicho) es
**0,898**, o sea que condicionado a lo que el modelo dice, no hay nada que corregir.
La primera lectura es la trampa clásica de la regresión a la media y **tienta a
expandir las predicciones, que es exactamente lo que no hay que hacer**. Por eso los
experimentos de expansión y calibración de noches anteriores no dieron nada: no
había nada que ganar ahí.

Figura: `reports/figuras_metricas/1_la_trampa_del_sesgo.png`

## 2 · Los dos extremos no son iguales (E31)

Un clasificador binario que solo lee el título:

| objetivo | prevalencia | AUC |
|---|---|---|
| microtarea ≤ 0,25 h | 6,6 % | **0,617** |
| microtarea ≤ 0,5 h | 17,6 % | 0,653 |
| tarea grande ≥ 16 h | 8,8 % | **0,740** |
| tarea grande ≥ 40 h | 2,6 % | **0,759** |

**Las tareas grandes se ven en el texto; las microtareas no.** Mirando los peores
casos (`results/diagnostico/residuos.csv`), las microtareas son tickets tipo *"Central
sync for project makumba"* con 0,1 h registradas: en buena parte son artefactos de
registro, no tareas que alguien pudiera haber estimado.

La descripción, que no usábamos, aporta **+0,015 de R²** donde existe (53 % de las
tareas), contra su control permutado. Real pero pequeño.

Figura: `reports/figuras_metricas/2_senal_en_los_extremos.png`

## 3 · Lo que se intentó, y qué dio

| # | intento | resultado |
|---|---|---|
| E32 | estimador distribucional (softmax sobre la rejilla de valores) | PRED25 0,167 → **0,186**, factor 2 0,446 → **0,467**; R² 0,199 → 0,093 |
| E33 | separar localización (apilado) y forma (distribución del residuo), residuo fuera de muestra con 5 pliegues | PRED25 en SiP 0,163 → **0,193** (McNemar p = 0,0046); R² → 0,12 |
| E34 | seis funciones de pérdida (absoluta, cuantil, gamma, pesos de regresión desequilibrada) | **negativo**: la cuadrática gana también el MdAElog |
| E37 | intervalos conformes | adaptativos 13 % más estrechos a igual cobertura; **la garantía se rompe** fuera de validación |
| E38 | recuperar las 65.003 tareas de Apache separando nivel de proyecto y desviación | **negativo**: baja el Spearman de 0,314 a 0,252 |
| E39 | apilar las once vistas disponibles | **negativo**: igual que el apilado de cuatro |
| E40 | elegir los pesos del apilado para la métrica que se reporta | mixto: gana en SiP, pierde en proyectos nuevos |

**Tres negativos con explicación, que valen tanto como los positivos:**

- **E34.** La teoría dice que el MdAElog quiere la mediana condicional y por tanto
  pérdida absoluta. No funciona porque los residuos en log **ya son casi simétricos**
  (SSPB −1,2 %): media y mediana coinciden, y la pérdida absoluta solo añade varianza
  de estimación. Los pesos de regresión desequilibrada (Yang et al. 2021) hunden el R²
  de 0,146 a 0,047, como cabía esperar: ese remedio corrige un desajuste entre la
  distribución de entrenamiento y la de prueba, y aquí son la misma.
- **E38.** Centrar por proyecto elimina justo la señal que el texto sí lleva sobre la
  escala del proyecto (jerga, nombres de componentes), y la estructura interna de
  Apache no transfiere.
- **E39.** Once canales no superan a cuatro. El brazo de control reproduce el modelo
  actual cifra por cifra, lo que confirma que la reimplementación es fiel.

## 4 · El hallazgo sólido: nada domina

Figura: `reports/figuras_metricas/3_frontera_del_canje.png`

Puestos los dieciséis brazos en el plano R² contra PRED(25), **ninguno cae arriba y a
la derecha del modelo actual**. Es una frontera, no un fallo. Y aparece por cuatro vías
independientes (E27, E28, E32, E33), lo que descarta que sea un artefacto de una
implementación concreta.

La razón es la de Gneiting (2011): un pronóstico puntual solo es óptimo respecto de
una pérdida. PRED(25) y factor 2 son pérdidas de cero-uno con tolerancia, cuyo
estimador óptimo es el centro de la ventana modal; R² y MdAElog quieren una tendencia
central. Con una distribución tan asimétrica y tan cuantizada como esta, esos dos
puntos están lejos.

**Corolario práctico para la tesis:** hay que decidir *una* métrica principal y
reportar las demás con su brazo correspondiente, en vez de buscar un modelo que gane
todas. Mi recomendación es MdAElog / ζ como principal, porque es la que la literatura
de datos positivos y sesgados avala (Morley et al. 2018) y la que no premia
subestimar.

## 5 · El otro hallazgo: la analogía gana la tasa de acierto (E39–E40)

Figura: `reports/figuras_metricas/4_que_sabe_cada_canal.png`

| canal | R² log | PRED(25) |
|---|---|---|
| analogía k=15 (mediana de los 15 títulos más parecidos) | **0,038** | **0,1775** |
| SVR disperso | 0,108 | 0,1815 |
| e5 ajustado | 0,091 | 0,1804 |
| el apilado de los cuatro | **0,199** | 0,1669 |

La analogía tiene el **mejor PRED(25) en validación y en los tres bloques de prueba**
(0,212 · 0,178 · 0,183 · 0,173) con un R² casi nulo. No es sobreajuste de validación:
se repite en todas partes. El mecanismo es el mismo que el de la regla de Bayes — la
mediana de k vecinos **cae sobre los valores que de verdad se observan**, y una media
condicional no. Esto conecta con la familia clásica de estimación por analogía (ABE)
de la literatura de esfuerzo, y explica por qué esos métodos sobrevivieron tanto
tiempo evaluados con PRED(25).

**Y explica el efecto incómodo de E40: el apilado empeora la tasa de acierto de sus
propios componentes**, porque promediar acerca a la media condicional y eso aleja del
valor más probable.

## 6 · Salida probabilística (E36–E37)

Figura: `reports/figuras_metricas/5_intervalos.png`

Ahora hay una distribución condicional por tarea, no solo un número. Evaluada con las
reglas propias del pronóstico:

| | CRPS log | cobertura del 80 % | anchura del 80 % |
|---|---|---|---|
| climatología (el constante probabilístico) | 0,766 | 0,822 | ×42 |
| lognormal sobre el apilado, σ 1,10 | **0,698** | 0,700 | ×11 |
| distribucional (E32) | 0,719 | 0,769 | ×16 |

Las distribuciones aprendidas son **mucho más nítidas** que la climatología, pero
salen **sobreconfiadas**. Los intervalos conformes adaptativos son un 13 % más
estrechos que los de anchura constante a igual cobertura — poco.

**Limitación que hay que escribir en la tesis:** la garantía conforme no se cumple
aquí. Al 80 % cubren 0,776 en la prueba fija y 0,710 en proyectos nuevos, porque la
partición es cronológica y calibración y prueba **no son intercambiables**, que es
justo el supuesto que pide el método. Calibrar con el tramo más reciente del
entrenamiento empeora (0,662): ese tramo es más fácil, no más parecido.

## 7 · Dónde queda el modelo frente a la literatura

Lo más útil que encontré es que **el estado del arte publicado tampoco le gana al
predictor trivial**:

- Tawosi, Moussa y Sarro (*IEEE TSE*, 2022) replican Deep-SE, el trabajo seminal de
  aprendizaje profundo para estimación ágil, y encuentran que **supera al predictor
  mediano con significancia estadística en solo 8 de 42 casos**. Su conclusión
  literal es que la similitud semántica no basta para distinguir historias de usuario
  por sus puntos.
- La misma réplica sobre GPT2SP (2022) encontró un fallo en el cálculo del MAE que
  inflaba sus cifras; corregido, gana al predictor mediano en 6 de 16 proyectos, con
  significancia en 3.

Nuestro modelo **sí le gana al constante con significancia** en PRED(25) (p = 0,0046
en la prueba fija, p = 0,0023 en SiP) y en factor 2 (p = 1,8 × 10⁻⁸ en SiP) tras
aplicar la regla de Bayes, y en MAE con SA 0,431 contra 0,393 del predictor mediano.
Eso es más de lo que consiguen los trabajos que replicaron.

**Pero hay que ser honesto en dos puntos:**
1. El **tamaño del efecto** (Δ de Glass) es **0,203**, que es *pequeño* según Cohen.
2. **A factor 2, el constante iguala o gana al modelo en frío** (45,1 % contra 44,6 %
   en la prueba fija; 48,5 % contra 41,7 % en externo). Ese es el argumento más fuerte
   a favor de pedirle la estimación al usuario, donde sube a 57,8 % y 75,5 %.

## 8 · Qué queda por probar

- **Reajustar el codificador con mejor objetivo** (E35, corriendo). El codificador
  ajustado pesa 0,390 en el apilado, más que la bolsa de siete árboles, y se ajustó
  con error cuadrático y una cabeza escalar. Se prueban una cabeza de clasificación
  sobre la rejilla y una pérdida de ordenación por pares. Es la única palanca viva que
  podría mover las métricas de error.
- **Un codificador base más fuerte** (`multilingual-e5-large-instruct`,
  `jina-embeddings-v3`). E4 comparó cinco en 2025 pero no los de instrucciones.

## Archivos

| script | qué hace |
|---|---|
| `src/130_diagnostico.py` | los ocho cortes del error |
| `src/131_extremos_y_descripcion.py` | AUC de los extremos, valor de la descripción |
| `src/132_distribucional.py` | softmax sobre la rejilla, un estimador por métrica |
| `src/133_forma_residual.py` | localización y forma separadas, residuo fuera de muestra |
| `src/134_perdidas.py` | seis funciones de pérdida |
| `src/136_probabilistico.py` | CRPS, PIT y cobertura |
| `src/137_conformal.py` | intervalos con garantía |
| `src/138_transferencia.py` | recuperar Apache |
| `src/139_apilado_total.py` | las once vistas |
| `src/140_pesos_por_metrica.py` | pesos elegidos para la métrica |
| `src/141_figuras_metricas.py` | las seis figuras |
