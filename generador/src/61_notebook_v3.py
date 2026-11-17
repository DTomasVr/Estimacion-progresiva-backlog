"""Quinto notebook: los roles y la coherencia historia->tarea (tanda v3).

Reutiliza los ayudantes de src/60_notebooks.py sin volver a escribirlos: se
ejecuta solo su cabecera, hasta el primer guardar(), que es donde estan md(),
code(), guardar() y CABECERA.

  python src/61_notebook_v3.py
"""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = open(f"{BASE}/src/60_notebooks.py", encoding="utf-8").read()
_cab = _src[:_src.index("# ============================================================ 1. datos")]
_ns = {"__name__": "ayudantes"}
exec(compile(_cab, "60_notebooks_cabecera", "exec"), _ns)
md, code, guardar = _ns["md"], _ns["code"], _ns["guardar"]
CABECERA = _ns["CABECERA"]

guardar("5_roles_coherencia.ipynb", [
    md("""# 5 · Los roles y la coherencia historia→tarea

La tanda **v3**. Dos defectos que se ven al leer la salida y que la métrica principal no registra,
más la métrica nueva que sí los ve.

El documento largo es `entrega/ROLES_Y_COHERENCIA.md`; la revisión de literatura sobre métricas
está en `entrega/METRICAS.md`."""),
    code(CABECERA),

    md("""## 5.1 El diagnóstico, y por qué mi primera hipótesis era falsa

Se venía afirmando que el modelo **colapsaba a un solo rol**. Se midió y es falso: el que colapsaba
era el **profesor** (63,5 % de sus backlogs con un solo rol), no el modelo (mediana 6 roles por
backlog, solo el 10 % con un rol único).

El defecto real es que la mitad de arriba **rellena el hueco del rol con un sustantivo cualquiera**
de la descripción: *As a service appointment, I want to schedule my first day of repairs*. En un
caso de taller de coches, 7 de 8 historias tenían un rol que no es una persona.

Y solo aparece **fuera de la taxonomía**: sobre las 40 descripciones reservadas del profesor el
100 % de los roles son válidos, porque ahí la descripción nombra un público que sí es un rol y el
modelo lo copia. Para ver el defecto hay que usar descripciones de consumo nuevas."""),
    code('correr("52_coherencia.py", "profesor", mostrar=14)'),
    md("""**La causa raíz cabe en una línea** de `41_dominios_sinteticos.py`:

```python
publico=random.choice(publicos)      # UN solo publico por descripcion
```

Cada descripción sintética nombraba una única audiencia, así que el profesor escribía las 12
historias para ese rol y el modelo aprendió que *el rol es un sustantivo que se copia de la
entrada*."""),

    md("""## 5.2 La métrica que faltaba: F1 de **arista**

El F1 suave de conjunto se calcula por nivel y por separado, así que **no ve** si las tareas cuelgan
de la historia equivocada. La prueba: se toman 25 backlogs correctos y se reparten sus tareas entre
historias equivocadas, sin quitar ni añadir ninguna."""),
    code('correr("57_f1_arista.py", "--sanidad", mostrar=12)'),
    md("""| caso | F1 historias | F1 de nodo | F1 de arista |
|---|---|---|---|
| idéntico | 1,000 | 1,000 | 1,000 |
| mitad desordenado | 1,000 | 1,000 | 0,684 |
| todo desordenado | 1,000 | **1,000** | **0,257** |

El F1 de nodo no se mueve ni una milésima. La unidad de evaluación tiene que ser la **arista**
(historia, tarea), con emparejamiento óptimo de historias primero (asignación húngara).

Respaldo bibliográfico: Node/Edge/Ancestor F1 de inducción de taxonomías, *relation generation* de
Wiseman, Shieber y Rush (EMNLP 2017), hF de clasificación jerárquica, PARENT (ACL 2019) para
referencias divergentes, AlignScore (ACL 2023) para fidelidad sin referencia, y el marco QUS/AQUSA
de Lucassen y **Dalpiaz** (REJ 2016), el mismo grupo del test humano.

**Dos trampas declaradas.** Primera: no reordenar las tareas con el mismo codificador con el que se
mide, porque sería optimizar el termómetro. Segunda: los filtros de inferencia hacen circulares las
medidas de eco y anclaje, así que van siempre con su ablación al lado.

Un dato de calibración: en un backlog **perfecto** la mala atribución no es 0 % sino 24,4 %. Incluso
entre historias correctas del mismo producto hay tareas que se parecen más a una hermana. Así que la
atribución del modelo se lee contra el **75,6 % del propio profesor**, no contra 100."""),

    md("""## 5.3 Los datos del profesor v3

| cambio | antes | ahora |
|---|---|---|
| públicos nombrados por descripción | 1 | **3** |
| plantel de actores por sector, dentro del prompt | no existía | **40 sectores, 136 actores** |
| sectores | 30 | **40** (10 nuevos de consumo) |
| backlogs / historias / tareas | 528 / 6.330 / 18.990 | **621 / 7.426 / 25.947** |
| se exige en el prompt | formato | formato **+ ≥4 roles del plantel + capas de tarea + frases prohibidas** |
| validación de lo que devuelve | ninguna | **rechaza y reintenta** |

Las tareas se piden **por capas**: qué datos guardar, qué hace la pantalla o el endpoint, qué regla
o validación aplica, y qué pasa cuando falla o hay que avisar a alguien. Eso es lo que hace que las
tres o cuatro tareas de una historia sean complementarias en vez de tres paráfrasis."""),
    code('''# correr("53_dominios_v3.py")                  # descripciones v3, ~1 min
# correr("54_generar_v3.py", "qwen2.5:7b")     # el profesor, 107 min
correr("52_coherencia.py", "profesor_v3", mostrar=14)'''),

    md("""## 5.4 Además: la variante sin público

Las descripciones de entrenamiento **nombran** a los actores; las que llegan a PBE a menudo no. Si
todos los ejemplos los nombran, el modelo aprende a copiar. Por eso cada descripción sintética
entrena **también en su versión despublicada**, con las mismas historias como objetivo:

```
CON : We are building an online store for shoppers, sellers and store managers. It must support ...
SIN : We are building an online store. It must support ...
```

Eso enseña la invariancia: el rol no se copia de la entrada, se elige según el dominio."""),

    md("""## 5.5 Dos métodos para el rol, comparados

**A · filtrar.** Se muestrean 30 candidatas y se descartan las que no tienen una persona en el hueco
del rol. El léxico sale de las **propias fuentes de entrenamiento** —plantel v3, roles del corpus
real y roles del profesor v1—: 935 roles y 262 núcleos de persona. Se acepta si el rol está listado
o si su núcleo (la última palabra) es de persona, así que `membership manager` pasa y
`service appointment` no.

**B · construir.** Un planificador decide primero **quién** es el actor y lo pasa como ficha:

```
feature -> user story | role: mechanic | A car workshop system for service appointments, ...
```

El plantel sale del sector más parecido de la taxonomía. Si la descripción no se parece a ninguno
(similitud < 0,70) **no fuerza nada** y trabaja el método A, para no poner *As a parking inspector*
en un servicio OAuth2. El umbral está medido: cuando el sector acierta la similitud es 0,79-0,84 y
cuando se equivoca 0,34-0,62."""),
    code('correr("55_lexico_roles.py", mostrar=6)'),

    md("""## 5.6 Los resultados

Tres columnas medidas con **un solo generador abajo**, para que la comparación sea pareada: con el
conjunto de dos se mezclaría el modelo nuevo con el viejo."""),
    code('''# la cadena entera, unas 2 h:  bash src/run_v3.sh
correr("59_tabla_v3.py", mostrar=45)'''),
    code('''for v in ["base", "v3a", "v3r"]:
    try:
        d = pd.read_csv(f"{BASE}/results/57_arista_{v}.csv")
        print(f"{v:5s}  nodo {d.nodo_F1.mean():.4f}  ARISTA {d.arista_F1.mean():.4f}  "
              f"mala atribucion {100*d.mala_atrib.mean():.1f} %")
    except FileNotFoundError:
        print(f"{v:5s}  todavia no medido")'''),
    md("""**Solapamiento declarado.** De los 6 proyectos del test humano, dos —`school` y
`supermarket`— tienen un sector hecho a medida entre los 10 nuevos de v3 (`colegio` y
`supermercado`). No hay fuga de textos: las descripciones sintéticas se escriben desde la taxonomía,
nunca desde las historias del test. Pero sí hay solapamiento de **dominio**, así que la cifra a
defender es la de los **4 proyectos limpios** (`fish_chips`, `ticket`, `grocery`, `sports`), cuyos
dominios ya estaban cubiertos igual desde la tanda v1."""),

    md("""## 5.7 Leer las salidas, que es lo que decide

La métrica no basta: fue leyendo como se encontraron el eco, el relleno de plantilla y las fugas de
Jira. Estas dos celdas imprimen las salidas de los dos métodos sobre los mismos casos."""),
    code('correr("45_leer_salidas.py", "nuevo", mostrar=60)'),

    md("""## 5.8 Lo que sigue pendiente

* **Dar la descripción del producto también a la mitad de abajo.** Ayudaría a la coherencia, pero el
  corpus real solo tiene 39 descripciones de proyecto: sería entrenar con un campo casi siempre
  ausente.
* **Sobremuestrear lo sintético abajo**: hoy son 6.106 filas contra 22.871 reales. Puede mejorar el
  estilo y empeorar la no-regresión, así que hay que tocarlo con la medida delante.
* **Calibrar la longitud** hacia ratio 1,0, que sigue pendiente de antes."""),
])
