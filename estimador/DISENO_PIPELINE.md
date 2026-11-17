# Diseño de la tubería completa de PBE

**De una descripción de producto a un backlog con horas estimadas.**

Son **dos tuberías encadenadas**, cada una con sus modelos, sus datos y su evaluación, y este
documento las reúne en uno solo:

| | qué hace | modelos | estado |
|---|---|---|---|
| **Tubería 1 · Generación** | descripción → historias de usuario con sus tareas | 3 × mT5-small podado (121 M) + SBERT propio + Marian | desplegado 28/09/2026, revisado 21/10/2026 |
| **Tubería 2 · Estimación** | tareas → horas, y el total del proyecto | e5-large congelado (ONNX int8, 397 MB) + dos apilados | `estimador_pbe_v7`, 21/10/2026 |

> **Alcance:** solo el diseño vigente de las dos tuberías. El camino de los experimentos
> descartados está en `ROL_FUNCION_LITERATURA.md`, `AUMENTO_DATOS_LITERATURA.md` (generación) y
> `INFORME_PROYECTO.md` (estimación). La demo web no se documenta aquí: es una envoltura, no
> parte de la tubería.

Los diagramas aparecen en **Mermaid** (se ven en GitHub y en VS Code) y, los de la generación,
también como **figura exportada** en SVG vectorial y PNG a 300 dpi para pegar en la memoria.

---

## Índice

**Parte 0 · La tubería completa**
- [0.1 De punta a punta](#01--de-punta-a-punta)
- [0.2 Figura A — Diagrama de flujo simple](#02--figura-a--diagrama-de-flujo-simple)
- [0.3 Figura B — La tubería completa en detalle](#03--figura-b--la-tubería-completa-en-detalle)
- [0.4 El contrato entre las dos tuberías](#04--el-contrato-entre-las-dos-tuberías)

**Parte A · Generación de historias y tareas** — [A.1](#a1-qué-hace-y-qué-decide-el-diseño) ·
[A.2 las siete decisiones](#a2-las-siete-decisiones-de-diseño) · [A.3 flujo](#a3-figura-1--flujo-de-extremo-a-extremo) ·
[A.4 etapas](#a4-detalle-de-cada-etapa) · [A.5 las dos mitades](#a5-figura-2--las-dos-mitades) ·
[A.6 plan de roles](#a6-figura-3--el-plan-de-roles-y-las-fichas-de-control) ·
[A.7 procedencia](#a7-figura-4--procedencia-de-datos-y-modelos) · [A.8 artefactos](#a8-artefactos-que-se-cargan-en-ejecución) ·
[A.9 configuración](#a9-configuración-desplegada) · [A.10 reproducir](#a10-cómo-reproducir-desde-cero) ·
[A.11 evaluación](#a11-protocolo-de-evaluación) · [A.12 resultados](#a12-resultados) ·
[A.13 lo que queda mal](#a13-lo-que-queda-mal-y-las-vías-cerradas) · [A.14 archivos](#a14-mapa-de-archivos)

**Parte B · Estimación de esfuerzo** — [B.1](#b1--qué-hace-y-en-una-tabla) ·
[B.2 dos cabezas](#b2--la-decisión-que-estructura-todo-dos-cabezas) · [B.3 diagramas](#b3--diagramas-de-flujo) ·
[B.4 datos](#b4--los-datos) · [B.5 arquitectura](#b5--arquitectura-interna) ·
[B.6 mecanismos de proyecto](#b6--los-tres-mecanismos-que-usan-información-del-proyecto) ·
[B.7 contrato](#b7--contrato-de-la-interfaz) · [B.8 artefactos y rendimiento](#b8--artefactos-y-rendimiento-medido) ·
[B.9 calidad](#b9--calidad-medida) · [B.10 límite conocido](#b10--límite-conocido-la-compresión-de-la-escala) ·
[B.11 descartes](#b11--lo-que-se-midió-y-se-descartó) · [B.12 trampas](#b12--dos-trampas-de-medición-que-conviene-no-repetir) · [B.13 operación](#b13--operación) · [B.14 decisiones](#b14--resumen-de-decisiones-de-diseño)

---

## 0.1 · De punta a punta

**Entrada**: una descripción de producto en español o en inglés, de una o dos frases.
**Salida**: **12 historias de usuario** en formato «Como X, quiero Y, para Z», cada una con sus
**tareas en español**, cada tarea con sus **horas**, y el **total del proyecto con un intervalo**.

| | coste medido |
|---|---|
| Tubería 1 · generar el backlog | ~26 s en RTX 5060 · ~33 s en CPU |
| Tubería 2 · estimar las horas | **5 s en CPU** para unas 35 tareas |
| carga de los modelos, una sola vez | ~20 s + ~2 s |

Ninguna de las dos llama a un servicio externo: todo corre en la máquina.

---

## 0.2 · Figura A — Diagrama de flujo simple

Lo mínimo que hay que entender: **qué entra y qué sale de cada tubería**, y que lo que la segunda
consume es el **inglés** de la primera, no su traducción.

```mermaid
flowchart LR
    IN[/"Descripción del producto<br/>una o dos frases · español o inglés"/]

    subgraph T1["TUBERÍA 1 · Generación del backlog"]
        direction TB
        G1[["mT5 One2One<br/>descripción → historia"]]
        G2[["mT5 One2Seq<br/>historia → tareas"]]
        G1 --> G2
    end

    subgraph T2["TUBERÍA 2 · Estimación del esfuerzo"]
        direction TB
        E1[["Cabeza A · MEDIANA<br/>horas de una tarea"]]
        E2[["Cabeza B · MEDIA<br/>horas que se pueden sumar"]]
    end

    IN --> T1
    T1 --> BK[/"BACKLOG EN INGLÉS<br/>12 historias · unas 35 tareas"/]
    BK --> T2
    BK --> TR["Traducir al español<br/>Marian en→es"]
    TR --> O1[/"Backlog en español<br/>lo que lee la persona"/]
    T2 --> O2[/"Horas por tarea · subtotal por historia<br/>total del proyecto con banda al 80 %"/]
    O1 --> FIN(["Backlog estimado"])
    O2 --> FIN

    style T1 fill:#f2e9d5,stroke:#7d5a17
    style T2 fill:#dcebe9,stroke:#12645d
    style BK fill:#f4e2dc,stroke:#a4442a
```

**Las formas dicen qué es cada cosa**, y se mantienen en las dos figuras:

| forma | significa | ejemplo |
|---|---|---|
| `[/paralelogramo/]` | **datos** que entran o salen | la descripción, el backlog, las horas |
| `[[subrutina]]` | un **modelo** entrenado | los dos mT5, las dos cabezas |
| `[rectángulo]` | un **paso de proceso** sin modelo | traducir, deduplicar, sumar |
| `{rombo}` | una **decisión** | ¿está en español?, ¿hay tareas cerradas? |
| `[(cilindro)]` | un **almacén** que se consulta | los corpus de vecinos |
| `([estadio])` | principio y **final** | el backlog estimado |

**El punto que más importa de esta figura**: el nodo rojo del centro es a la vez **la salida de la
tubería 1 y la entrada de la tubería 2**, y está **en inglés**. La traducción al español es una
rama lateral, solo para mostrar. Pasarle el español al estimador no fallaría; simplemente
acertaría menos, porque su contrato pide inglés.

---

## 0.3 · Figura B — La tubería completa en detalle

Las mismas dos tuberías con todas sus etapas reales.

```mermaid
flowchart TB
    IN[/"Descripción del producto<br/>español o inglés"/]

    subgraph T1["TUBERÍA 1 · GENERACIÓN DEL BACKLOG"]
        direction TB
        D1{"¿está<br/>en español?"}
        D2["Leer los ACTORES del español<br/>léxico de 57 actores"]
        D3["Traducir la entrada<br/>Marian es→en"]
        P1["Planificar ROLES<br/>4 fuentes, nunca vacío<br/>plantel solo con respaldo"]
        P2["Planificar CLASES de función<br/>2 por rol · 6 clases"]
        V1["Recuperar vecinos · k=2<br/>procedencia, no van al prompt"]
        M1[["mT5 One2One · 121M<br/>muestreo top-p 0,95<br/>6 muestras por par (rol, clase)"]]
        B1[/"Banco de ~36 candidatas"/]
        ANC["ANCLAS · una historia por<br/>función descrita, prefijo forzado"]
        S1["SELECCIÓN por cobertura de rasgos<br/>voraz submodular + 6 castigos de forma"]
        FU["FUSIÓN · anclas primero, hasta k/2<br/>y la selección sin duplicarlas"]
        H1[/"12 historias elegidas"/]
        V2["Recuperar vecinos por historia<br/>procedencia, no van al prompt"]
        M2[["2 × mT5 One2Seq · 121M en conjunto<br/>haz de 4, sin repetir 4-gramas"]]
        DD["Filtrar no-desarrollo · deduplicar<br/>tope de arranques · reasignar cada tarea"]
        R1[("corpus o2o<br/>pipeline_emb.npz")]
        R2[("corpus hist<br/>s2s_hist_train")]
        D1 -->|sí| D2 --> D3 --> P1
        D1 -->|no| P1
        P1 --> P2 --> V1 --> M1 --> B1 --> S1 --> FU --> H1 --> V2 --> M2 --> DD
        P1 --> ANC --> FU
        R1 -.-> V1
        R2 -.-> V2
    end

    IN --> D1
    DD --> BK[/"BACKLOG EN INGLÉS<br/>12 historias · unas 35 tareas"/]

    subgraph T2["TUBERÍA 2 · ESTIMACIÓN DEL ESFUERZO"]
        direction TB
        C1["Codificar cada tarea<br/>e5-large congelado 1024d<br/>+ TF-IDF caracteres y palabras + SVD"]
        AN["Analogía: 25 vecinos más próximos<br/>ajuste a la rejilla de 30 valores"]
        CA[["Cabeza A · MEDIANA<br/>apilado RidgeCV de 5 canales<br/>× factor de estilo 0,831"]]
        CB[["Cabeza B · MEDIA<br/>apilado Poisson de 4 canales<br/>× factor de estilo 0,727"]]
        SU["Sumar las medias"]
        AC{"¿el equipo ya cerró<br/>alguna tarea?"}
        AK["ANCLAR: cociente de sumas<br/>contraído con James-Stein k/(k+12)"]
        IV["Intervalo al 80 %<br/>solo sobre lo abierto, por k y tramo"]
        E3[("16.648 vectores<br/>de entrenamiento")]
        C1 --> AN
        C1 --> CA
        C1 --> CB
        AN --> CA
        AN --> CB
        CB --> SU --> AC
        AC -->|no| IV
        AC -->|sí| AK --> IV
        E3 -.-> AN
    end

    BK --> C1
    BK --> TR["Traducir la salida<br/>Marian en→es + reparaciones en español"]
    TR --> O1[/"Backlog en español"/]
    CA --> O2[/"Horas por tarea<br/>la mediana"/]
    IV --> O3[/"Total del proyecto<br/>con banda al 80 %"/]
    SU --> O4[/"Subtotal por historia<br/>la media, que sí se suma"/]
    O1 --> FIN(["BACKLOG ESTIMADO"])
    O2 --> FIN
    O3 --> FIN
    O4 --> FIN

    style T1 fill:#f2e9d5,stroke:#7d5a17
    style T2 fill:#dcebe9,stroke:#12645d
    style BK fill:#f4e2dc,stroke:#a4442a
```

Tres cosas que esta figura hace explícitas y que son todo el diseño:

1. **La tubería 1 genera; el pipeline decide.** El modelo de arriba produce ~36 candidatas y la
   selección por cobertura elige 12. La calidad no sale del modelo, sale de la elección. Lo único
   que se fuerza es que cada función que la descripción nombra tenga su historia (las anclas,
   A.2.6), porque lo que el modelo no genera no se puede elegir.
2. **La tubería 2 tiene dos cabezas y no una.** La de la mediana contesta «cuánto tarda esta
   tarea» y la de la media contesta «cuánto suma esto». No son el mismo número y no se pueden
   intercambiar.
3. **Las dos se juntan por el inglés**, y cada una aplica su propio factor de estilo porque el
   texto del generador no se parece a un título de Jira.

---

## 0.4 · El contrato entre las dos tuberías

Lo que una entrega y la otra espera. Está implementado en `src/34_estimacion.py` (en el
repositorio de la generación).

| | |
|---|---|
| **qué se pasa** | `h["tareas"]`, la lista de tareas **en inglés**, antes de traducir |
| **qué NO se pasa** | `h["tareas_es"]`: el estimador se entrenó con títulos de Jira en inglés |
| **qué devuelve** | horas por tarea, subtotal por historia, total con intervalo |
| **orden** | el factor de estilo se aplica **antes** del anclaje con tareas cerradas |

**El factor de estilo del total no es el de la mediana.** La cabeza A multiplica por 0,8313; la
cabeza B no aplicaba nada, y el factor de la cabeza A no vale porque son dos modelos con pérdidas
distintas. Medido (E74) sobre las 800 tareas reales reescritas al estilo del generador, todas
fuera del entrenamiento, con cociente de sumas:

| texto de entrada | suma predicha / suma real |
|---|---|
| título original de Jira | **1,0018** — calibrado |
| el mismo contenido en estilo del generador | **1,3775** |
| **factor que lo corrige** | **0,7273** |

Sin esa corrección **el total del proyecto sale un 37 % alto**. Con ella, la media por tarea de un
backlog generado queda en 6,3 h, que es la media del corpus (6,11 h).

**Por qué se aplica antes del anclaje**: el anclaje es un cociente de sumas, así que un factor
constante se cancelaría solo si no hubiera contracción. Con James-Stein (`k/(k+12)`) la
cancelación es parcial, y la predicción corregida es el mejor punto hacia el que contraer.

---

---

# PARTE A · Generación de historias y tareas

> La primera tubería: de la descripción del producto al backlog. Estado desplegado a
> 21/10/2026: el diseño del 28/09 más los arreglos de la revisión de extremo a extremo
> (A.2.6, A.2.7 y A.12). Los experimentos descartados están en `ROL_FUNCION_LITERATURA.md` y
> `AUMENTO_DATOS_LITERATURA.md`.

---

## A.1 Qué hace y qué decide el diseño

**Entrada**: una descripción de producto en español o en inglés, de una o dos frases.
**Salida**: **12 historias de usuario** en formato «Como X, quiero Y, para Z», cada una con sus
**tareas**, en español.

**Coste**: ~26 s por descripción en una RTX 5060, de los cuales ~20 s son cargar los modelos una sola
vez. Tres mT5-small podados de 121 M parámetros cada uno —uno arriba y dos en conjunto abajo—;
entran de sobra en 8 GB de VRAM. Las anclas de A.2.6 añaden ~0,4 s en GPU y 1,3–2,3 s en CPU.

El problema que el diseño tiene que resolver no es generar texto plausible —un modelo pequeño lo hace—
sino tres cosas que un modelo pequeño **no** hace solo:

1. que el backlog tenga **varios actores** y que sean los del producto descrito;
2. que no sean doce historias de **pedir ver cosas**;
3. que las tareas pertenezcan a **su** historia y no se repitan;
4. que **cada función que la descripción nombra** esté en el backlog, con su actor.

Las cuatro se resuelven **fuera del modelo**: condicionando la generación con fichas de control y
eligiendo después entre muchas candidatas. El modelo genera; el pipeline decide.

---

## A.2 Las siete decisiones de diseño

Cada una con el número que la sostiene. Son las que hay que poder defender.

### A.2.1 Dos modelos encadenados, no uno

Un solo modelo que fuera de la descripción al backlog completo tendría que aprender a la vez a
inventar historias y a descomponerlas. Se separa en **descripción → una historia** (One2One) y
**historia → sus tareas** (One2Seq), cada uno con sus datos y su decodificación. Permite además
muestrear arriba (hacen falta muchas candidatas para elegir) y usar haz abajo (ahí se quiere la mejor
descomposición, no variedad).

### A.2.2 Fichas de control al estilo CTRL

A la entrada de la mitad de arriba se le anteponen dos fichas: `| role: X |` y `| funcion: Y |`.

- **Ficha de rol.** Sin ella el backlog se concentra en un actor. Medido: con plantel de actores la
  obediencia al rol es **100 %**; sin plantel, **7 %**.
- **Ficha de función.** El modelo se concentraba en pedir *ver* cosas: el 42 % de las historias eran de
  clase `view`, contra el 22 % de las humanas. Con la ficha, la obediencia es del **99 %** y la
  entropía de clases pasa de 2,22 a **2,39** sobre un máximo de 2,58, con las humanas en 2,45.

### A.2.3 Los actores se leen del ESPAÑOL, antes de traducir

El plan de roles se construía sobre la descripción ya traducida, y la traducción destruye los actores:
«apoderados» → «**agents**», «avisa al equipo» → «alerts the **computer**». Y es **inestable**: la
misma palabra sale «proxy» en una frase corta y «agents» en la descripción completa, así que un
diccionario sobre el inglés no puede arreglarlo.

Medición que cierra el diagnóstico: una sonda directa de obediencia, 8 muestras por petición y 32
peticiones, da **8/8 en 31 de 32**, incluidos `guardian`, `vendor` y `team member`. **El modelo obedece;
el rol llegaba mal.** Por eso se lee del español, que es la fuente.

### A.2.4 Selección por cobertura de rasgos, no por relevancia

De las ~36 candidatas generadas se eligen 12 maximizando la **cobertura de los rasgos que la
descripción enumera** (localización de instalaciones, voraz submodular), no la relevancia individual.
Medido sobre 30 descripciones reservadas: los rasgos cubiertos pasan del 81,1 % al **87,8 %**, y los
backlogs que los cubren todos del 53,3 % al **66,7 %**. Cuesta 2,6 puntos de atribución, y se acepta:
un rasgo sin cubrir es trabajo que no existe en el backlog.

**Los rasgos se leen de la descripción entera** (`rasgos_con_sujeto`). El extractor anterior
cortaba por el primer «with» y tiraba la cabeza de la frase, que es justo donde va la función
central: «A web application to book home cleaning services, with…» daba solo lo de detrás del
«with», y «graduates apply with their CV» daba «their CV». La selección llevaba tiempo cubriendo
los rasgos equivocados. Ahora se parte por comas, punto y coma y dos puntos, y cada rasgo guarda
su **sujeto** («companies» en «companies publish offers»), que A.2.6 usa para elegir el actor.

Sobre esa selección van **seis castigos de forma** (circularidad, autorreferencia, eco, rol-función,
largo, beneficio vacío) y tres términos de cobertura (rol, clase de función, beneficio literal).

### A.2.5 El aumento de datos imita las condiciones de inferencia

El modelo se entrenaba con descripciones en inglés **escritas a mano**, pero en producción la entrada
es español que pasa por Marian. Ese inglés no habla igual: el viaje de ida y vuelta cambia el **38 %
de las palabras**. Se añadieron al conjunto los mismos ejemplos con la descripción retrotraducida por
los **mismos** modelos que el producto usa. Resultado pareado con tres semillas: **+0,0118 de sF1**,
las tres a favor, con desviación 0,0017.

### A.2.6 Una historia ancla por función descrita

En la revisión de extremo a extremo, dos de seis proyectos salían **sin su función central**:
reservar una limpieza, reservar una sala de estudio. No era la selección: en salas el banco traía
**1 candidata de reservar sala entre 28**, y en la ferretería **0 de transferencias entre tiendas**.
Lo que no se genera no se puede elegir. Y pedirlo por la entrada no funciona: con «A mobile app:
find and book study rooms» el modelo sigue escribiendo lecciones, porque la ficha de función son
seis verbos sin objeto y el decodificador copia poco objeto de la entrada.

Lo que el modelo **sí obedece** es un **prefijo forzado** en el decodificador. Por cada función que
la descripción enumera (A.2.4) se construye «As a student, I want to find and book study rooms on
campus in real time, so that» y el modelo escribe solo el beneficio (6 muestras, elige el
reordenador de beneficios). La capacidad sale de `capacidad_de_rasgo`: el rasgo tal cual si empieza
por verbo, o con un verbo ligero si es un sustantivo («low stock alerts» → «receive low stock
alerts»). El actor sale del sujeto del rasgo, con una tabla para los colectivos que el SBERT no
casa con su rol («companies» quedaba más cerca de «graduate» que de «recruiter»).

Las anclas salen de las funciones de la descripción y del plan de **roles** (no del de
clases), y se **fusionan después** de la selección: entran **siempre**, primero, hasta k/2, y el
resto de lo seleccionado completa el backlog sin duplicarlas. Es un chequeo de
cobertura, pero en el lado de la **generación**, que es donde estaba el hueco. `PBE_ANCLAR=0` lo
apaga para comparar.

Según el informe del chat de generación (`RESPUESTA_REVISION_EXTREMO_A_EXTREMO.md`), de la mejora total de sF1 de la revisión
(+0,0317), **+0,0261 son de las anclas** (ablación, IC95 [+0,018, +0,034]). **Al citarlo:** la
capacidad sale literal de la descripción, que es la entrada y no la referencia, así que no hay
fuga; pero el sF1 premia reutilizar la descripción, y parte de la ganancia es eso.

### A.2.7 El plantel del sector solo entra con respaldo

El umbral de sector (0,65) decide si **hay** plantel, pero no si es el **correcto**. Limpieza a
domicilio caía en *atención domiciliaria* con 0,691 contra 0,682 del sector de limpieza —gana por
0,009— y el backlog salía entero de enfermeras. Salas de estudio nombra solo «estudiante», y el
plantel de docencia metía instructor y asistente, y con ellos asistencia y lecciones. Los sectores
acertados están en 0,75 o más (taller 0,755, clínica 0,827, delivery 0,853, banca 0,920); los
equivocados, entre 0,69 y 0,71.

Por eso, **por debajo de 0,75** (`UMBRAL_PLANTEL_LIBRE`) un rol del plantel solo entra si la
descripción lo **respalda**: comparte con ella una palabra de dominio, por raíz de cinco letras
(«cleaner» con «cleaning», «store manager» con «stores»; `rol_respaldado`). Las palabras de rol que
no dicen nada del dominio —manager, agent, member, administrator…— no cuentan como respaldo.
Calibrado sobre 9 puntos, no más.

Además se leen del español «egresado», «recepción» y, solo en un portal de empleo, «empresa». Al
empleador se le pide *recruiter* y no *employer*: en una sonda de 16 muestras «employer» se obedece
11 veces y el resto sale «lawyer»; «recruiter», 16 de 16.

---

## A.3 Figura 1 — Flujo de extremo a extremo

![Flujo de extremo a extremo](reports/figuras_generacion/fig1_pipeline.png)

*Archivos: `reports/figuras_generacion/fig1_pipeline.svg` · `reports/figuras_generacion/fig1_pipeline.png`*

```mermaid
flowchart TD
    A["Descripción del producto<br/>(español o inglés)"] --> B{"¿está en español?"}
    B -->|sí| B1["Leer los ACTORES del español<br/>actores_es_nombrados"]
    B1 --> B2["Traducir a inglés<br/>Marian opus-mt-es-en"]
    B -->|no| C
    B2 --> C["Planificar ROLES<br/>PlanificadorRoles<br/>plantel con respaldo si sector &lt; 0,75"]
    C --> D["Planificar CLASES de función<br/>plan_funciones"]
    D --> E["Recuperar vecinos del corpus<br/>k=2, tabla o2o<br/>procedencia, no van al prompt"]
    E --> F["MITAD DE ARRIBA<br/>descripción + fichas → 1 historia<br/>muestreo, 6 por par (rol, clase)"]
    F --> G["Banco de candidatas<br/>filtros de forma y de léxico"]
    C --> AN["ANCLAS: una historia por función descrita<br/>funciones de la descripción + plan de roles<br/>prefijo forzado, el modelo escribe el beneficio"]
    G --> H["SELECCIÓN por cobertura de rasgos<br/>seleccion_por_rasgos, voraz"]
    H --> FU["FUSIÓN: las anclas primero, hasta k/2<br/>y la selección sin duplicarlas"]
    AN --> FU
    FU --> I["Recuperar vecinos por historia<br/>tabla hist · procedencia"]
    I --> J["MITAD DE ABAJO<br/>historia → conjunto de tareas<br/>haz de 4"]
    J --> K["Filtrar y deduplicar tareas<br/>no-desarrollo · global 0,90<br/>intrahistoria 0,95 y 0,95 sin el verbo"]
    K --> L["Tope de arranques repetidos<br/>máximo 2 por backlog"]
    L --> M["Reasignar la tarea a su historia<br/>y piso de pertenencia 0,30"]
    M --> N["Traducir al español<br/>Marian opus-mt-en-es"]
    N --> O["Reparaciones en español<br/>glosario · glosario con fuente · glosario general<br/>restos en inglés · unificar roles<br/>verbo en infinitivo · eco por flexión"]
    O --> P["Backlog final<br/>12 historias con tareas"]
```

Las etapas que el propio pipeline cronometra, en orden: `traducir_entrada`, `recuperar_arriba`,
`generar_historias`, `deduplicar_historias`, `recuperar_abajo`, `generar_tareas`, `deduplicar_tareas`,
`reasignar_tareas`, `traducir`, `total`.

---

## A.4 Detalle de cada etapa

| # | etapa | qué hace | dónde está |
|---|---|---|---|
| 1 | Actores del español | 57 actores del español al rol inglés que el generador espera. Solo si la entrada viene en español | `actores_es_nombrados` |
| 2 | Traducir la entrada | Marian `opus-mt-es-en` | `Pipeline.__call__` |
| 3 | Plan de roles | cuatro fuentes en orden de fiabilidad (figura 3) | `PlanificadorRoles` |
| 4 | Plan de clases | 2 clases por rol según P(clase \| familia del rol), con piso 0,10 | `plan_funciones` |
| 5 | Vecinos de arriba | 2 ejemplos del corpus real por similitud del SBERT propio. **No van en el prompt** (`EJEMPLOS_ARRIBA = False`: con ellos el sF1 bajaba 0,022); quedan para procedencia y orden | `_vecinos` |
| 6 | Generación de arriba | una tanda de muestreo **por par (rol, clase)**, mínimo 6 muestras cada una | `_generar` |
| 7 | Filtros del banco | formato de historia, rol válido (persona, no sustantivo cualquiera), léxico de Jira | `rol_valido`, `degenerada_texto` |
| 7b | Anclas | una historia por función enumerada, a partir de las funciones de la descripción y del plan de roles, con prefijo forzado y el beneficio elegido entre 6 | `_anclas`, `rasgos_con_sujeto`, `capacidad_de_rasgo` |
| 8 | Selección y fusión | cobertura de rasgos + 3 términos de reparto + 6 castigos de forma; después, las anclas primero (hasta k/2) y lo seleccionado sin duplicarlas | `seleccion_por_rasgos`, `__call__` |
| 9 | Vecinos de abajo | ejemplos para cada historia elegida; **tampoco van en el prompt** (`EJEMPLOS_ABAJO = False`, empate con y sin ellos) | `_vecinos` |
| 10 | Generación de abajo | haz de 4, sin repetir 4-gramas, penalización 1,3 | `_generar` |
| 11 | Filtrar y deduplicar tareas | fuera lo que no es desarrollo (verbo como *attend*, *proceed*, *discuss*, *please*; «prepare for»; frases de historia como «I would like») · 0,90 entre historias · 0,95 dentro · 0,95 **sin el verbo inicial** | `tarea_degenerada`, `__call__` |
| 12 | Tope de arranques | máximo 2 tareas por backlog con el mismo arranque de dos palabras | `arranque` |
| 13 | Reasignar | cada tarea va a la historia a la que de verdad pertenece; piso 0,30. Ninguna historia se queda vacía, y el rescate prefiere una tarea que no esté ya en otra historia | `__call__` |
| 14 | Traducir la salida | Marian `opus-mt-en-es` | `__call__` |
| 15 | Reparaciones | glosario, glosario con fuente inglesa, **glosario general** disparado por el inglés de origen (stock → existencias, job listing → oferta de empleo, Implement → Implementar, apply → postular en empleo, grade → calificación, appointment → cita, branches → sucursales fuera de un repositorio, test case → caso de prueba), **restos en inglés** (real-time, follow-up, Scheduler, HR…), unificar nombres de rol, verbo en infinitivo, eco por flexión | `glosario`, `glosario_fuente`, `glosario_general`, `unificar_roles_es`, `normalizar_verbo_es`, `eco_por_raiz` |

**Por qué el tope de arranques está en la salida y no en los datos**: los datos de entrenamiento ya
están planos (3,0 % de concentración en los cinco arranques más frecuentes, contra 2,6 % humano) y el
modelo los amplifica ×14,6 hasta el 43,8 %. Es una propiedad del decodificador, así que se corrige en
inferencia.

---

## A.5 Figura 2 — Las dos mitades

![Las dos mitades](reports/figuras_generacion/fig2_dos_mitades.png)

*Archivos: `reports/figuras_generacion/fig2_dos_mitades.svg` · `reports/figuras_generacion/fig2_dos_mitades.png`*

```mermaid
flowchart LR
    subgraph ARRIBA["Mitad de arriba: One2One"]
        direction TB
        A1["feature -> user story<br/>| role: X | funcion: Y |<br/>&lt;descripción&gt; | examples:"]
        A2["Una historia de usuario<br/>As a X, I want ..., so that ..."]
        A1 --> A2
    end
    subgraph ABAJO["Mitad de abajo: One2Seq"]
        direction TB
        B1["decompose story into tasks<br/>&lt;historia&gt; | examples:"]
        B2["Conjunto de tareas<br/>t1 ; t2 ; t3 ..."]
        B1 --> B2
    end
    ARRIBA -->|"12 historias elegidas"| ABAJO
```

Formato exacto de entrada de cada mitad:

```
arriba:  feature -> user story | role: courier | funcion: track | <descripción> | examples:
abajo:   decompose story into tasks: <historia> | examples:
```

El campo `examples:` va **vacío** en las dos mitades. Los modelos se entrenaron con ese formato,
pero medido en `24_calidad_arriba.py` los ejemplos recuperados no ayudan: arriba el sF1 sube 0,022
**sin** ellos y no hay fuga de vocabulario de los vecinos; abajo, empate. La recuperación sigue
corriendo para la procedencia (`vecinos_proyecto`) y para ordenar, no para el prompt.

| | decodificación | por qué |
|---|---|---|
| arriba | muestreo, top-p 0,95, T 1,0, ~6 muestras por par (rol, clase) | hacen falta muchas candidatas para que la selección pueda elegir |
| abajo | haz de 4, `no_repeat_ngram_size=4`, `repetition_penalty=1,3` | aquí no se quiere variedad, se quiere la mejor descomposición |

---

## A.6 Figura 3 — El plan de roles y las fichas de control

![Plan de roles](reports/figuras_generacion/fig3_plan_roles.png)

*Archivos: `reports/figuras_generacion/fig3_plan_roles.svg` · `reports/figuras_generacion/fig3_plan_roles.png`*

```mermaid
flowchart TD
    A["Descripción"] --> B["Actores leídos del ESPAÑOL<br/>léxico de 57 actores → rol inglés"]
    B --> C{"¿2 o más?"}
    C -->|sí| D["Se les cree:<br/>NO se consulta la traducción"]
    C -->|no| E["Extractor sobre el inglés<br/>actores_nombrados, léxico de 935 roles"]
    E --> F["Descartar los que son MÁQUINA<br/>rol_infraestructura"]
    F --> G{"¿la descripción se parece a<br/>un sector de la taxonomía?<br/>similitud ≥ 0,65"}
    G -->|sí| H["Plan = actores leídos + plantel del sector<br/>hasta k = 4<br/>bajo 0,75: solo roles respaldados"]
    G -->|no| I{"¿2 o más actores?"}
    I -->|sí| J["Plan = solo los actores leídos"]
    I -->|no| K["Actores COLECTIVOS<br/>the team → team member"]
    K --> L{"¿ya hay 2?"}
    L -->|sí| J
    L -->|no| M["Completar con roles genéricos<br/>administrator, user, support agent"]
    D --> H
    H --> N["plan_funciones:<br/>2 clases por rol, según P(clase|familia del rol)<br/>con piso 0,10"]
    J --> N
    M --> N
    N --> O["Pares (rol, clase)<br/>una tanda de muestreo por par"]
```

**Las seis clases de función** salen de agrupar con el SBERT propio los verbos del medio de 9.077
historias (`src/84_clases_funcion.py`), no de una lista escrita a mano: **receive, view, set, track,
monitor, manage**.

**El plan nunca queda vacío**, y esa es la regla que más basura evita. Cuando lo estaba, el modelo
inventaba el actor: en el dominio de integración continua salían **6 roles fabricados en 12
historias** («hydrator user», «spending collector», «volunteer»).

Dos reglas del plan, las dos falsificadas antes de entrar:

- **Descartar el actor que es máquina** («runs build pipelines **on agents**»): probada contra las 622
  descripciones de la taxonomía, descarta **0 de 1.507** actores, y 1 de 14 en los siete dominios del
  producto, que es justo el que se buscaba.
- **Si el español nombra 2 actores o más, no se consulta la traducción**, porque su entrada es una
  versión con pérdidas de la misma frase.
- **Con similitud de sector por debajo de 0,75, el plantel solo aporta roles respaldados** por la
  descripción (A.2.7). Es la regla que quitó las enfermeras de la limpieza a domicilio.

---

## A.7 Figura 4 — Procedencia de datos y modelos

![Procedencia](reports/figuras_generacion/fig4_procedencia.png)

*Archivos: `reports/figuras_generacion/fig4_procedencia.svg` · `reports/figuras_generacion/fig4_procedencia.png`*

```mermaid
flowchart LR
    M0["google/mt5-small<br/>300M"] --> M1["Poda de vocabulario<br/>250.100 → 75.360<br/>src/10_prune_vocab.py<br/>121M parámetros"]
    M1 --> UA["s2s_o2o<br/>src/17_one2one_arriba.py"]
    UA --> UB["s2s_o2ofmt<br/>formato de historia"]
    UB --> UC["s2s_v9bt<br/>real + profesor + retrotraducción<br/>25.938 ejemplos"]
    UC --> UD["MITAD DE ARRIBA<br/>_o2o_o2ofmt_v9bt_pbe_v1"]
    M1 --> LA["s2s_hist<br/>src/15_datos_historias.py"]
    LA --> LB["s2s_v5<br/>real + profesor curado"]
    LB --> LC["MITAD DE ABAJO<br/>_hist_v5_pbe_v1"]
    S0["paraphrase-multilingual-<br/>mpnet-base-v2"] --> S1["SBERT propio<br/>src/06_finetune_sbert.py<br/>sbert_pbe_v1"]
```

Hiperparámetros de los tres modelos desplegados, que son los que hay que repetir:

| mitad | base | bs × acum | max_out | lr | épocas | val por época |
|---|---|---|---|---|---|---|
| arriba (`v9bt`) | `_o2o_pbe_v1_o2ofmt_pbe_v1` | 32 × 1 | 48 | 5e-4 | 4 | 1,7947 · 1,8017 · **1,7897** · 1,8208 |
| abajo (`v5`) | `_hist_pbe_v1` | 16 × 1 | 192 | 5e-4 | 4 | **3,3210** · 3,3393 · 3,3565 |
| abajo, 2.º del conjunto (`_hist_bs8_lr9.2e-4`) | datos: solo `s2s_hist`, 22.871 ejemplos, sin profesor | 8 × 1 | — | 9,2e-4 | parada en la 5.ª | mejor la 3.ª |

Se guarda la **mejor época por pérdida de validación**, no la última. Datos y épocas leídos de
los registros de entrenamiento (`logs/09_v9bt.log`, `logs/44_train_abajo_v5.log`,
`logs/09_train_hist_hp.log`) del repositorio del generador: `v9bt` con 25.938 ejemplos, `v5`
con 28.274.

**Nota de licencia**: el profesor sintético es `qwen2.5:7b`, Apache-2.0, elegido a propósito. Llama
queda excluido porque su licencia obligaría a poner «Llama» en el nombre del modelo resultante.

**La poda de vocabulario no cuesta calidad**: medido pareado, −0,0019 de sF1 con IC95 [−0,0050,
+0,0011], es decir empate; y sin podar, el lote 16 no cabe en 8 GB.

---

## A.8 Artefactos que se cargan en ejecución

| artefacto | lo produce | para qué |
|---|---|---|
| `models/..._v9bt_pbe_v1` | `09_train_mt5.py` | mitad de arriba |
| `models/..._hist_pbe_v1_v5_pbe_v1` | `09_train_mt5.py` | mitad de abajo |
| `models/mt5-small-pruned_hist_pbe_v1_bs8_lr9.2e-4` | `09_train_mt5.py` | mitad de abajo, 2.º del conjunto: sola empata, pero se equivoca distinto y la unión sube la cobertura |
| `models/sbert_pbe_v1` | `06_finetune_sbert.py` | recuperación, selección, deduplicado |
| `models/reordenador_beneficio.joblib` | `74_reordenador.py` | bonificación por coherencia del beneficio (AUC 0,728) |
| `data/processed/pipeline_emb.npz` | `08_preprocess.py` | embeddings del corpus para recuperar vecinos |
| `data/processed/s2s_o2o_train.parquet` | `17_one2one_arriba.py` | tabla de vecinos de arriba |
| `data/processed/s2s_hist_train.parquet` | `15_datos_historias.py` | tabla de vecinos de abajo |
| `data/processed/roles_lexico.json` | `55_lexico_roles.py` | 935 roles, 262 núcleos de persona, 114 vetados |
| `data/processed/clases_funcion.json` | `84_clases_funcion.py` | las 6 clases y P(clase \| familia de rol) |
| `data/processed/matriz_rol_verbo.json` | `83_matriz_rol_verbo.py` | asociación rol×verbo |
| `data/processed/lexico_jira_estricto.json` | `81_lexico_jira_estricto.py` | 167 palabras de Jira a vetar en las tareas |
| `data/synth/descripciones_v3.json` | `53_dominios_v3.py` | 622 descripciones y el plantel de 40 sectores |
| `models/marian_es_en`, `models/marian_en_es` | copia local de `Helsinki-NLP/opus-mt-*` | traducción de entrada y de salida. Desde carpeta y no desde el caché de HF: con el caché, estando en línea, la librería volvía a resolver `main` y bajaba otra variante de los pesos (+600 MB, y el paquete dejaba de ser offline sin avisar) |

Comprobar que están todos: `python src/91_comprobar_entorno.py`.

---

## A.9 Configuración desplegada

Todo se puede sobreescribir por variable de entorno; estos son los valores por omisión.

| parámetro | valor | qué controla |
|---|---|---|
| `K_HISTORIAS` | 12 | historias del backlog |
| `N_MUESTRAS` | 30 | candidatas pedidas (mínimo 6 por par rol×clase) |
| `K_VECINOS` | 2 | vecinos recuperados por consulta |
| `UMBRAL_SECTOR` | 0,65 | a partir de aquí se usa el plantel del sector |
| `UMBRAL_PLANTEL_LIBRE` | 0,75 | por debajo, un rol del plantel solo entra si la descripción lo respalda |
| `ANCLAR` | 1 | una historia ancla por función descrita |
| `N_BEN_ANCLA` | 6 | beneficios muestreados por ancla |
| `CONJUNTO` | `dos_modelos` | la mitad de abajo con los dos modelos; `no` usa uno solo |
| `MODO_ROLES` | `descripcion` | manda lo que la descripción nombra |
| `MIN_ACTORES_CON_PLANTEL` | 1 | con plantel basta un actor leído |
| `MIN_ACTORES_SIN_PLANTEL` | 2 | sin plantel hacen falta dos |
| `MODO_FUNCION` | `auto` | ficha de función; se activa por el nombre del modelo |
| `CLASES_POR_ROL` | 2 | clases de función por actor |
| `PISO_CLASE` | 0,10 | probabilidad mínima para asignar una clase a un rol |
| `NU_CIRCULAR` · `NU_AUTO` · `NU_ECO` · `NU_ROLFUN` · `NU_VACIO` | 0,50 · 0,40 · 0,40 · 0,45 · 0,45 | castigos por la forma de la historia |
| `NU_CAP` | **0,0** | castigo por capacidad repetida: **apagado, descartado con medición** |
| `UMBRAL_TAREA_DUP` | 0,90 | tareas duplicadas entre historias distintas |
| `UMBRAL_TAREA_DUP_INTRA` | 0,95 | dentro de una misma historia |
| `UMBRAL_TAREA_SINV` | 0,95 | ídem, comparando **sin el verbo inicial** |
| `TOPE_ARRANQUE_SALIDA` | 2 | veces que se repite un arranque de dos palabras |
| `PISO_TAREA` | 0,30 | pertenencia mínima de la tarea a su historia |
| `BEAMS_ABAJO` | 4 | anchura del haz de la mitad de abajo |
| `ECO_RAIZ` | 1 | eco por flexión, también en español |

> **Cuidado al desplegar un modelo nuevo.** Las fichas de rol y de función se activan por el **nombre**
> del modelo, con la lista `CON_FICHA = ("v3r", "v8ff", "v9bt", "v9df")` en `src/19_pipeline.py`. Un
> modelo cuyo nombre no esté ahí corre **sin fichas** —es decir, sin la mitad del diseño— **y sin
> avisar**. `src/92_ver_salida.py` imprime al arrancar si están activas, precisamente por esto.

---

## A.10 Cómo reproducir desde cero

Intérprete: `C:\Users\User\Documents\Programacao\PBE-Dataset\.venv\Scripts\python.exe` o
`F:\PBE\venv\Scripts\python.exe` (los dos tienen torch 2.11 + cu128). **El `python` del PATH no
sirve.** Las rutas del proyecto están al principio de cada script.

```bash
# --- 1. corpus real (Jira / TAWOS) y particion por proyecto
python src/01_extract_jira_full.py
python src/02_extract_tawos_changelog.py
python src/03_build_hierarchy.py
python src/04_split_projects.py

# --- 2. SBERT propio, que usa todo lo demas
python src/05_sbert_pairs.py
python src/06_finetune_sbert.py

# --- 3. datos de las dos mitades y poda del vocabulario
python src/08_preprocess.py
python src/10_prune_vocab.py
python src/15_datos_historias.py      # abajo: historia -> tareas
python src/17_one2one_arriba.py       # arriba: descripcion -> una historia

# --- 4. datos sinteticos del profesor (requiere ollama con qwen2.5:7b)
python src/53_dominios_v3.py          # 622 descripciones sobre 40 sectores
python src/54_generar_v3.py qwen2.5:7b
PBE_SYNTH=backlogs_v3_qwen2.5_7b.jsonl PBE_PREF=v8f python src/43_datos_mixtos.py

# --- 5. el aumento por retrotraduccion (lo que distingue a v9bt)
python src/88_retrotraducir_datos.py 0.5

# --- 6. entrenamiento de las dos mitades
DATOS=v9bt python src/09_train_mt5.py models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1 32 48 0.0005 4
DATOS=v5   python src/09_train_mt5.py models/mt5-small-pruned_hist_pbe_v1 16 192 0.0005 4

# --- 7. recursos que el pipeline carga en ejecucion
python src/55_lexico_roles.py
python src/81_lexico_jira_estricto.py
python src/83_matriz_rol_verbo.py
python src/84_clases_funcion.py
python src/74_reordenador.py

# --- 8. comprobacion
python src/91_comprobar_entorno.py
python src/19_pipeline.py --demo
PBE_SEED=20260926 python src/49_eval_real.py nuevo
```

---

## A.11 Protocolo de evaluación

Tres medidas con propósitos distintos, y conviene no confundirlas.

| medida | contra qué | script | para qué sirve |
|---|---|---|---|
| **sF1 suave de conjunto** | 328 historias escritas por **personas** en 6 proyectos reales | `49_eval_real.py` | **la única que no se explica por imitar al profesor.** Es la que decide |
| atribución tarea→historia | consigo misma | `92_ver_salida.py` | coherencia interna. **Sube al empobrecer el backlog, así que sola engaña** |
| coherencia y cobertura | rasgos enumerados en la descripción | `52_coherencia.py`, `68_cobertura` | si el backlog cubre lo que se pidió |

Cuatro reglas de medición que este proyecto aprendió a la fuerza:

1. **Tres semillas, pareadas.** Una sola corrida tiene un ruido de ~0,012 de sF1, del tamaño de los
   efectos que se buscan. Cada semilla se corre en los **dos** brazos y se compara la diferencia.
2. **La línea base se regenera en la misma sesión.** Con la misma semilla y el mismo código, una
   corrida de dos horas antes **no** se reproduce; dos seguidas sí, byte a byte. Para el «antes» se
   copia el respaldo de `respaldo/19_pipeline_*.py` a `src/` y se carga por ruta.
3. **Leer la salida en texto antes de mirar los números.** `python src/92_ver_salida.py <etiqueta>`
   imprime los 7 dominios de prueba. Varias mejoras aparentes en las métricas resultaron ser el
   backlog empobreciéndose.
4. **El sF1 puntúa solo las historias**, no las tareas (`H = [h["historia"] for h in r["historias"]]`).
   Un cambio que solo toque tareas no puede moverlo, y no hace falta medirlo con semillas.

---

## A.12 Resultados

### Contra historias escritas por personas

| versión | sF1 (media de 3 semillas) |
|---|---|
| modelo anterior (`v8ff`) | 0,6245 |
| **desplegado (`v9bt`)** | **0,6363** |

Diferencia **+0,0118**, medida pareada: las tres semillas a favor, desviación 0,0017.

### Sobre los 7 dominios de prueba (84 historias, ~250 tareas)

| | antes | ahora |
|---|---|---|
| Roles **fabricados** | 12 de 84 | **0** |
| Roles inexistentes por error de traducción | 4 de 84 | **0** |
| El actor que la descripción nombra llega al backlog | 5 de 7 dominios | **7 de 7** |
| Historias casi duplicadas (personas 4,8 %) | 10,7 % | **7,1 %** |
| Beneficio vacío (personas 0 de 235) | 3 de 84 | **0** |
| Tareas gemelas dentro de una historia | 3 pares | **0** |
| Eco por flexión | 3,1 % | **0,8 %** |
| Historias sin ninguna tarea | 4,8 % | **~1 %** |
| Formato «Como… quiero… para…» | — | **100 %** |
| Atribución tarea→historia | 83 % | **86 %** |

### Revisión de extremo a extremo (21/10/2026)

El chat del estimador corrió el sistema completo sobre seis descripciones y revisó a mano las 199
tareas; los fallos se confirmaron después con **3 semillas** y eran sistemáticos. Cifras del informe
del chat de generación (`RESPUESTA_REVISION_EXTREMO_A_EXTREMO.md`), no de un CSV:

| proyecto | antes | después |
|---|---|---|
| Reservas de limpieza | enfermera, cuidador; ninguna reserva | cliente y limpiador; reservar servicios, citas recurrentes, pagos |
| Salas de estudio | instructor, asistente; ninguna sala | «encontrar» y «reservar salas de estudio» siempre; el resto aún tira a lo académico |
| Bolsa de trabajo | faltaban publicar y postular | reclutador «publicar ofertas», egresado «postular con mi CV» |
| Ferretería | sin transferencias; «acciones» | «gestionar transferencias entre tiendas», «existencias» |
| Historias clínicas | bien | igual, y ahora con recepcionista, que la descripción nombra |
| Delivery | bien | publicar menús, pedir, pagar y rutas, cada uno con su actor |

Contra las **328 historias humanas**, pareado en la misma sesión, 3 semillas × 6 proyectos:

| | sF1 |
|---|---|
| antes de la revisión | 0,6354 |
| **después** | **0,6671** |
| diferencia | **+0,0317**, IC95 [+0,025, +0,038] |

Gana en los 6 proyectos y en las 3 semillas; +0,0261 es de las anclas (A.2.6, con su salvedad).
El filtro nuevo de tareas quita 33 de 1.240 (2,7 %) en 36 backlogs, y leídas una a una son todas
ruido. El duplicado literal entre historias es 1 de ~620 tareas (0,2 %).

---

## A.13 Lo que queda mal y las vías cerradas

Escrito para que nadie lo descubra creyendo que es nuevo.

| defecto | tasa medida | estado |
|---|---|---|
| **Fuga de vocabulario de otro sector** («mi **prenda**» en bodas, «mi **scooter**» en técnico, «perfil de la **enfermera**» en un colegio) | ~1 por backlog | **CERRADO como negativo medido.** El veto por sector marca 18 historias y solo 1 es fuga real; y los dominios con fugas son los que no empatan con ningún sector, así que ahí el conjunto de palabras «propias» también está mal. *Solo es seguro donde no hace falta* |
| **Historias casi duplicadas** del mismo actor | 7,1 % (personas 4,8 %) | **CERRADO.** El castigo funciona (→ 0 %) pero cuesta 0,007 de sF1 con las tres semillas en contra, y 0 % se pasa del nivel humano |
| **Plantilla CRUD en las tareas** | — | **CERRADO.** Bajar la plantilla en los datos 4,6 puntos movió la salida 0,3: es del decodificador, no de los datos |
| Beneficio con un objeto ajeno al dominio | ~4 de 84 | abierto; misma familia que la fuga de sector |
| Eco por flexión residual | 0,8 % de las tareas | abierto |
| Términos técnicos que el traductor de salida rompe | corregidos *branch*, *runs*, *pipeline* | abierto para el resto; se amplía en `_GLOSARIO_FUENTE` |
| **Fuera de lo que la descripción nombra**, el modelo rellena con lo típico del sector (salas de estudio, fuera de sus dos anclas, sigue tirando a lo académico) | — | abierto. En la interfaz conviene pedir al usuario quién hace qué (`INTEGRACION.md` §4) |
| Beneficios genéricos frecuentes («para poder ofrecer un mejor servicio») | — | abierto |
| Alguna tarea floja bajo las anclas («Crear un proyecto» para «encontrar salas») | — | abierto |
| El **sustantivo** «Procedure for…» pasa el filtro de no-desarrollo, que atrapa el verbo *proceed* («Proceder para solicitudes recurrentes», visto en la prueba del paquete del 21/10) | 1 en 36 tareas de esa prueba | abierto |
| Diversidad de superficie de las descripciones sintéticas | 81 arranques distintos para 622 | abierto; la vía sería generar con atributos ortogonales al estilo AttrPrompt, lo que exige volver a pasar el profesor |

---

## A.14 Mapa de archivos

### Producción

| archivo | qué es |
|---|---|
| `src/19_pipeline.py` | **el módulo entero.** Todo lo de producción está aquí |
| `src/91_comprobar_entorno.py` | verifica los 13 artefactos y la GPU, sin cargar modelos |
| `src/92_ver_salida.py` | genera y enseña los 7 dominios, con la tabla de cifras y `--comparar` |
| `src/49_eval_real.py` | sF1 contra las 328 historias humanas |
| `src/33_servidor.py` | demostración web en el puerto 8123 |
| `src/90_figuras.py` | regenera las cuatro figuras de este documento |

### Datos y entrenamiento

`09_train_mt5.py`, `43_datos_mixtos.py`, `88_retrotraducir_datos.py`, `89_dedup_historias_profesor.py`,
más los de la sección 10.

### Documentación

| documento | contenido |
|---|---|
| **`DISENO_PIPELINE.md`** | este documento: el diseño completo de las DOS tuberías |
| `ESTADO_Y_PUESTA_EN_MARCHA.md` | qué se consiguió y cómo poner en marcha y probar el módulo |
| `ROL_FUNCION_LITERATURA.md` | los experimentos de control de roles y funciones, **con los descartes y su medición** |
| `AUMENTO_DATOS_LITERATURA.md` | el repaso del aumento de datos contra literatura 2023-2025 y los dos reentrenamientos |
| `reports/figuras_generacion/*.{svg,png}` | las cuatro figuras, vectoriales y a 300 dpi |

### Respaldos

`respaldo/19_pipeline_*.py` guarda el pipeline antes y después de cada cambio. Sirven para regenerar
una línea base en la misma sesión (sección 11, regla 2).

---

# PARTE B · Estimación de esfuerzo

> La segunda tubería: de cada tarea a sus horas, y del backlog entero al total del proyecto.
> `estimador_pbe_v7`, corpus v4 sin Apache, construido el 21/10/2026 sobre el v6 del
> 29/09/2026: el mismo diseño con **un solo codificador** (ver B.5.1). Los experimentos
> descartados están en `INFORME_PROYECTO.md`.

---

## B.1 · Qué hace, y en una tabla

PBE descompone un proyecto en tareas con un LLM. Este componente recibe **el texto de esas
tareas** y responde **cuántas horas**. Es el arranque en frío: en ese momento no existe
historial del proyecto, solo títulos recién generados.

| entrada | salida | cuándo |
|---|---|---|
| lista de títulos | horas por tarea, con banda | al generar el backlog |
| lista de títulos | **total del proyecto** con intervalo al 80 % | al presentar el plan |
| + horas reales de tareas ya cerradas | total **recalculado y anclado** | según avanza el proyecto |
| + estimación del usuario | total corregido | si el formulario la pide |

**Cifras de cabecera** (114 proyectos, todos predichos como si fueran nuevos):

| | MAE | MdMRE | R²log | ±25 % | factor 2 | sesgo |
|---|---|---|---|---|---|---|
| línea base constante | 886 h | 0,694 | 0,364 | 8,8 % | 24,6 % | −203 % |
| **en frío** | **455 h** | **0,355** | **0,817** | **36,8 %** | **72,8 %** | −0,3 % |
| **con 5 tareas cerradas** | **379 h** | **0,291** | **0,911** | **43,7 %** | **85,5 %** | +0,2 % |

---

## B.2 · La decisión que estructura todo: dos cabezas

**Es el punto central del diseño y la razón de que el artefacto lleve dos modelos.**

Las dos preguntas del producto no tienen la misma respuesta estadística:

| pregunta | respuesta correcta | por qué |
|---|---|---|
| ¿cuánto tarda **esta** tarea? | la **mediana** condicional | es el número que se le enseña a una persona; la mediana minimiza el error absoluto |
| ¿cuánto tarda **el proyecto**? | la **media** condicional | el total es una **suma**, y solo las medias suman |

La distribución de horas es muy asimétrica (mediana 2,0 h, media 6,11 h). Sumar medianas deja
el total en el **44 %** de lo que toca. De ahí dos cabezas con **pérdidas distintas** sobre los
mismos rasgos:

- **Cabeza A — por tarea:** error cuadrático sobre `log(horas)` → mediana condicional.
- **Cabeza B — para sumar:** **pérdida de Poisson sobre horas** → media condicional directa.

Antes esto se resolvía empujando la mediana con el estimador de *smearing* de Duan (1983).
Funcionaba, pero entrenar directamente para el objetivo correcto mejora MdMRE, MAE, RMSE,
±50 % y factor 2 **a la vez**. El factor de smearing sigue en el paquete por trazabilidad, pero
**el servidor no lo usa**.

---

## B.3 · Diagramas de flujo

### B.3.1 · Flujo de extremo a extremo (solo la estimación)

```mermaid
flowchart LR
    U(["Usuario"]) -->|describe el proyecto| A["PBE · web"]
    A -->|descompone| L["Generador de tareas · LLM"]
    L -->|lista de titulos| E["Estimador de esfuerzo<br/>(este componente)"]
    E -->|horas por tarea + total + banda| A
    A -->|muestra el plan| U
    U -->|trabaja y registra horas| T[("Time tracker")]
    T -->|horas reales de tareas cerradas| E
    T -->|lote periodico| R["Reentrenamiento"]
    R -.->|artefacto nuevo| E
    style E fill:#dcebe9,stroke:#12645d,stroke-width:2px
```

### B.3.2 · Camino de una petición, con las dos rutas

El diagrama que más importa: **las dos preguntas no recorren el mismo camino**.

```mermaid
flowchart TB
    IN(["lista de titulos"]) --> NORM["normalizar<br/>titulo, ingles, sin descripcion"]
    NORM --> VIS["codificar: 3 vistas del texto"]
    VIS --> RA{"que se pregunta?"}

    RA -->|cuanto tarda ESTA tarea| CA["Cabeza A · MEDIANA<br/>apilado de 3 canales"]
    CA --> EST["x factor de estilo 0,831"]
    EST --> BND["banda x4,45"]
    VIS --> ANA["analogia: 25 vecinos<br/>enganchada a la rejilla"]
    ANA --> VP{"vecinos<br/>de acuerdo?"}
    VP -->|si, el 94,7 %| VPS["valor_probable = rejilla"]
    VP -->|no| VPN["valor_probable = apilado"]
    BND --> OUT1(["horas · valor_probable · banda"])
    VPS --> OUT1
    VPN --> OUT1

    RA -->|cuanto tarda EL PROYECTO| CB["Cabeza B · MEDIA<br/>apilado Poisson de 4 canales"]
    CB --> UE{"hay estimacion<br/>del usuario?"}
    UE -->|si| CORR["corrector de Poisson"]
    UE -->|no| SUM["sumar las tareas"]
    CORR --> SUM
    SUM --> CER{"hay tareas<br/>cerradas?"}
    CER -->|no| TOT["total en frio"]
    CER -->|si, k tareas| ANC["anclaje: cociente de sumas<br/>contraido k/(k+12)"]
    ANC --> TOT
    TOT --> INT["intervalo al 80 %<br/>x10,6 en frio, x2,6 con 10 cerradas"]
    INT --> OUT2(["total_horas · banda_baja · banda_alta"])

    style CA fill:#f2e9d5,stroke:#7d5a17
    style CB fill:#dcebe9,stroke:#12645d
    style ANC fill:#f4e2dc,stroke:#a4442a
```

### B.3.3 · Flujo interno del modelo

```mermaid
flowchart TB
    subgraph COD["1 · Codificacion del titulo"]
        direction LR
        T["titulo"] --> E5["e5-large congelado<br/>ONNX int8 · 1024 dim"]
        T --> TF["TF-IDF caracteres 3-5<br/>+ palabras 1-2"]
        TF --> SVD["SVD 256 + 256"]
    end

    subgraph CA["2a · Cabeza A · horas por TAREA (mediana)"]
        direction TB
        A1["bolsa de 7 HGB<br/>e5 + SVD"] --> AP{{"apilado RidgeCV<br/>ajustado en validacion"}}
        A2["Ridge disperso"] --> AP
        A3["SVR disperso"] --> AP
        AP --> AX["exp(.) x factor de estilo"]
    end

    subgraph CB["2b · Cabeza B · horas para SUMAR (media)"]
        direction TB
        B1["bolsa de 5 HGB Poisson<br/>e5 + SVD"] --> BP{{"apilado HGB Poisson<br/>ajustado en validacion"}}
        B2["HGB Poisson sobre SVD"] --> BP
        B3["HGB Poisson sobre analogia"] --> BP
        B4["Poisson lineal disperso<br/>TF-IDF crudo"] --> BP
    end

    COD --> CA
    COD --> CB
    AX --> S1["horas por tarea + banda"]
    BP --> S2["suma + anclaje + intervalo"]

    style CA fill:#f2e9d5,stroke:#7d5a17
    style CB fill:#dcebe9,stroke:#12645d
```

### B.3.4 · Nivel 1 · Contexto

```mermaid
C4Context
title Nivel 1 · Contexto del componente de estimación

Person(dev, "Desarrollador / líder de proyecto", "Crea proyectos, revisa las tareas generadas y registra sus horas")
Person(tesista, "Responsable del modelo", "Construye el corpus, entrena y evalúa")

System(estimacion, "Componente de estimación de esfuerzo", "Horas por tarea y total del proyecto a partir del texto; ancla con las tareas ya cerradas")

System_Ext(pbe, "PBE · aplicación web", "Gestión de proyectos: backlog, tablero y time tracker")
System_Ext(llm, "Generador de tareas (LLM)", "Descompone el proyecto en tareas")
System_Ext(corpus, "Corpus públicos", "TAWOS, The Public Jira Dataset, SiP (Jones)")
System_Ext(hf, "Hugging Face Hub", "Origen de intfloat/multilingual-e5-large; en servicio va dentro del paquete")

Rel(dev, pbe, "Crea proyectos y registra horas", "HTTPS")
Rel(pbe, llm, "Pide la descomposición en tareas", "HTTP/JSON")
Rel(pbe, estimacion, "Pide horas por tarea y total del proyecto", "HTTP/JSON")
Rel(estimacion, pbe, "Devuelve horas, total e intervalo", "HTTP/JSON")
Rel(pbe, estimacion, "Horas reales de las tareas cerradas", "en cada recálculo")
Rel(tesista, estimacion, "Entrena, selecciona y evalúa", "scripts")
Rel(corpus, estimacion, "Alimentan el entrenamiento", "volcados SQL / mongodump")
Rel(hf, estimacion, "Provee el codificador base congelado", "descarga única, al construir")

UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="2")
```

### B.3.5 · Nivel 2 · Contenedores en ejecución

```mermaid
C4Container
title Nivel 2 · Contenedores en tiempo de ejecución (v7)

Person(dev, "Desarrollador", "Usa PBE")
System_Ext(pbe, "PBE · aplicación web", "Backend + frontend")

Container_Boundary(est, "Componente de estimación de esfuerzo") {
  Container(api, "EstimadorV6", "Python", "Dos entradas: estimar_tareas y estimar_proyecto; carga perezosa del codificador")
  Container(encbase, "Codificador", "ONNX Runtime · multilingual-e5-large CONGELADO, int8, vocabulario podado", "Texto a vector de 1024. 397 MB. Lo usan las DOS rutas")
    Container(cabA, "Cabeza A · mediana", "scikit-learn · apilado RidgeCV de 3 canales", "Horas de UNA tarea, con banda x4,45")
  Container(cabB, "Cabeza B · media", "scikit-learn · apilado HGB de Poisson de 4 canales", "Horas que se pueden SUMAR")
  Container(analogia, "Índice de analogía", "numpy · 16.648 vectores float16", "25 vecinos más próximos, mediana enganchada a la rejilla")
  Container(anclaje, "Anclaje por proyecto", "Python", "Cociente de sumas contraído con James-Stein, lambda = 12")
  Container(corrector, "Corrector de la estimación del usuario", "scikit-learn · HGB de Poisson", "Vector + log(estimación) + log(predicción en frío)")
  Container(intervalo, "Intervalo del total", "Python · cuantiles empíricos", "Residuos en log de los 114 proyectos")
  ContainerDb(artefactos, "Almacén de artefactos", "joblib + carpetas de pesos", "estimador_pbe_v7.joblib de 106 MB, mas models/e5_v7 (ONNX, tokenizador, tabla del vocabulario)")
  ContainerDb(historial, "Historial de tareas", "base de datos de PBE", "Tareas cerradas con horas reales y estimaciones del usuario")
  Container(reentreno, "Reentrenamiento por lotes", "Python", "Reajusta las cabezas; el codificador base NO se toca")
}

Rel(dev, pbe, "Crea el proyecto y registra horas", "HTTPS")
Rel(pbe, api, "estimar_tareas / estimar_proyecto", "JSON")
Rel(api, encbase, "Texto de las tareas", "en proceso")
Rel(encbase, cabA, "Vectores", "numpy")
Rel(encbase, cabB, "Vectores", "numpy")
Rel(encbase, analogia, "Vector normalizado", "numpy")
Rel(analogia, cabA, "Valor de la rejilla", "numpy")
Rel(analogia, cabB, "Canal de analogía", "numpy")
Rel(cabB, corrector, "Predicción en frío como rasgo", "numpy")
Rel(cabB, anclaje, "Horas por tarea para sumar", "numpy")
Rel(historial, anclaje, "Horas reales de las cerradas", "JSON")
Rel(anclaje, intervalo, "Total puntual", "numpy")
Rel(intervalo, api, "Total y banda", "numpy")
Rel(api, artefactos, "Carga al arrancar", "joblib")
Rel(pbe, historial, "Horas reales al cerrar", "SQL")
Rel(reentreno, historial, "Lee tareas cerradas", "SQL")
Rel(reentreno, artefactos, "Escribe el artefacto nuevo", "joblib")

UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

> **Decisión que se ve aquí:** hay **un solo codificador**, congelado, y lo comparten las dos
> rutas. El v6 llevaba además dos codificadores ajustados (2,2 GB cada uno) para la ruta por
> tarea; medidos fuera de pliegue no aportaban y el v7 los quitó (B.5.1). El codificador va en
> ONNX int8 con el vocabulario podado: **397 MB** frente a los 6,5 GB de los tres del v6.

### B.3.6 · Nivel 2b · Tubería de datos y entrenamiento (fuera de línea)

```mermaid
C4Container
title Nivel 2b · Tubería de datos y entrenamiento

System_Ext(tawos, "TAWOS", "Volcado SQL, 458.232 incidencias, 39 proyectos")
System_Ext(jira, "The Public Jira Dataset", "mongodump de 5,87 GB, 2.686.282 incidencias")
System_Ext(sip, "SiP (Jones)", "CSV de una empresa real, horas reales y estimadas")

Container_Boundary(tuberia, "Tubería de datos") {
  Container(extraccion, "Extractores", "Python · jira_archive.py, tawos_sql.py", "Leen los volcados en streaming, sin bases de datos intermedias")
  ContainerDb(interim, "Capa intermedia", "Parquet · data/interim/", "Una tabla por colección, sin limpiar")
  Container(construccion, "Constructor del dataset", "Python · 39_build_v3, 52_build_v4", "Armoniza taxonomías, unifica unidades, descarta filas y arma las particiones")
  ContainerDb(procesado, "Dataset", "Parquet · pbe_effort_v4.parquet", "90.975 filas: train 63.926, val 8.445, test 4.225, externo 13.255")
  Container(codificacion, "Codificación con caché", "Python · 53_encode_v4, 98_codificadores", "Reutiliza vectores calculados; solo codifica el texto nuevo")
  ContainerDb(vectores, "Vectores", "npy · embeddings/codificadores/", "e5-large.npy, alineado por fila (y los ajustados de los experimentos)")
  Container(finetune, "Ajuste de codificadores", "PyTorch · 147_finetune_rapido", "Brazos A y C sobre e5-large. Solo experimentos: el v7 no los usa")
  Container(entrenamiento, "Entrenamiento de las cabezas", "Python · 163_modelo_v5, 171_estimador_v6, 252_estimador_v7", "Ajusta los canales y el apilado; elige SIEMPRE en validación")
  Container(evaluacion, "Evaluación", "Python · GroupKFold por proyecto", "MdMRE, PRED(25), MAE, SA, zeta, SSPB y contrastes pareados")
  ContainerDb(resultados, "Resultados", "CSV/JSON · results/", "Una fila por brazo, escenario y bloque de prueba")
}

Rel(tawos, extraccion, "SQL comprimido", "zip")
Rel(jira, extraccion, "mongodump --gzip --archive", "zip")
Rel(sip, construccion, "CSV cp1252", "fichero")
Rel(extraccion, interim, "Escribe", "parquet")
Rel(interim, construccion, "Lee", "parquet")
Rel(construccion, procesado, "Escribe con particiones", "parquet")
Rel(procesado, codificacion, "Texto de cada tarea", "parquet")
Rel(codificacion, vectores, "Escribe", "npy")
Rel(vectores, finetune, "Punto de partida", "npy")
Rel(finetune, vectores, "Vectores ajustados A y C", "npy")
Rel(vectores, entrenamiento, "Rasgos", "numpy")
Rel(procesado, entrenamiento, "Objetivo y particiones", "parquet")
Rel(entrenamiento, evaluacion, "Predicciones por bloque", "numpy")
Rel(evaluacion, resultados, "Escribe", "csv/json")

UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

> **Garantías de la partición:** `test` es cronológica dentro de cada proyecto, `externo` son
> **150 proyectos enteros** nunca vistos, y los duplicados se agrupan por hash de texto para que
> no crucen particiones. En la evaluación por proyecto se usa siempre `GroupKFold` **por
> proyecto entero**: partir por tarea filtraría la escala del proyecto al entrenamiento.

### B.3.7 · Orden de reconstrucción

**El orden importa.** Si cambian los pesos de un codificador ajustado, cambian sus vectores, y
la cabeza que los consume queda ajustada sobre los antiguos. Servir con un codificador que no
case con su cabeza **no falla: solo acierta menos**.

```mermaid
flowchart LR
    F["147_finetune_rapido.py<br/>reentrena el codificador C"] --> V5["163_modelo_v5.py<br/>apilado por tarea"]
    V5 --> V6["171_estimador_v6.py<br/>paquete v6"]
    V6 --> V7["252_estimador_v7.py<br/>un solo codificador"]
    V7 --> OX["246 + 249: ONNX int8<br/>y vocabulario podado"]
    OX --> P{{"prueba de extremo a extremo<br/>estima de verdad, con y sin cerradas"}}
    P -->|verde| OK(["artefacto desplegable"])
    P -->|roja| STOP(["no se despliega"])
    ORQ["177_rehacer_artefacto.py<br/>encadena y se niega si faltan requisitos"] -.-> V5
    style P fill:#f2e9d5,stroke:#7d5a17
    style STOP fill:#f4e2dc,stroke:#a4442a
```

La prueba final no es opcional: **es lo único que demuestra que el artefacto sirve**. La primera
vez que se construyó el v6 parecía correcto y no podía codificar texto nuevo, porque los pesos
del codificador nunca se habían guardado. Lo destapó justamente esa prueba.

Con el v7 el codificador ya no se reajusta, así que la parte que más se rompía —un
codificador nuevo con una cabeza vieja— desaparece. Si se cambia el codificador base, el
ONNX y la tabla del vocabulario se regeneran con `246` y `249`, y se mide con `253`.

### B.3.8 · Vista dinámica · el ciclo de vida de un proyecto

```mermaid
sequenceDiagram
    autonumber
    actor U as Desarrollador
    participant W as PBE (web)
    participant L as Generador (LLM)
    participant A as EstimadorV6
    participant D as Historial

    U->>W: Crea el proyecto
    W->>L: Descompón en tareas
    L-->>W: Lista de títulos
    W->>A: estimar_tareas(titulos)
    A-->>W: horas, valor_probable y banda por tarea
    W->>A: estimar_proyecto(titulos)
    A-->>W: total en frío + intervalo al 80 % (se estrecha al cerrar tareas)
    W-->>U: Plan PROVISIONAL, marcado como tal

    loop Según avanza el proyecto
        U->>W: Cierra una tarea con sus horas reales
        W->>D: Guarda las horas
        W->>A: estimar_proyecto(titulos, cerradas)
        Note over A: k tareas conocidas:<br/>off = log(suma real / suma predicha) x k/(k+12)<br/>total = suma real + suma(predicho abiertas x e^off)
        A-->>W: total anclado
        W-->>U: Plan actualizado
    end

    Note over U,W: Desde k = 3 la mejora es significativa<br/>en el 100 % de los sorteos

    D->>A: Reentrenamiento por lotes (solo las cabezas)
```

---

## B.4 · Los datos

### El dataset completo

| | |
|---|---|
| fichero | `data/processed/pbe_effort_v4.parquet` · 90.975 filas · 52 columnas · 533 proyectos |
| particiones | train 63.926 · val 8.445 · test 4.225 · externo 13.255 (150 proyectos enteros) |
| tareas de software | 89.851 |
| fuentes | TAWOS · The Public Jira Dataset · SiP (Jones) |

### Lo que el modelo usa de verdad

| | |
|---|---|
| subconjunto | tareas de software **sin Apache** |
| tamaño | **24.848 tareas · 204 proyectos** (114 con ≥ 8 tareas, que son la unidad de evaluación) |
| fuentes | TAWOS 9.559 · SiP 7.791 · JiraPublico 7.498 |
| horas | mediana 2,0 h · media 6,11 h · p01 0,09 h · p99 66 h |
| particiones | train 16.648 · val 3.183 · test fija 4.225 · externo 792 |

**Por qué se excluye Apache:** su mediana es 0,67 h contra 2,0 h del trabajo comercial, y
mezclarlo desplaza toda la escala hacia abajo. Se midió (E38) y empeora. Apache sí se usa, pero
**solo como corpus de réplica independiente** para validar conclusiones —186 proyectos,
64.587 tareas— nunca para entrenar.

**Rasgos que el diseño prohíbe usar**, aunque estén en el corpus, porque en producción no
existen cuando se crea la tarea: `source` (PBE sería un nivel no visto), recuentos de enlaces y
de componentes (se acumulan mientras la tarea se trabaja, y valen cero al crearla), y story
points.

---

## B.5 · Arquitectura interna

### B.5.1 · Codificación del título

Tres vistas del mismo texto, porque cada una acierta en cosas distintas:

| vista | qué es | dimensiones |
|---|---|---|
| **semántica** | `intfloat/multilingual-e5-large`, **congelado**, prefijo `query: `, en ONNX int8 | 1024 |
| **léxica densa** | TF-IDF de caracteres `char_wb` 3–5 (vocab 35.450) + palabras 1–2 (vocab 20.256), `sublinear_tf`, reducidos con SVD | 256 + 256 |
| **léxica dispersa** | el mismo TF-IDF **sin reducir**, para los modelos lineales | ~55.700 |

**Un solo codificador.** El v6 llevaba además dos codificadores ajustados sobre el mismo
`e5-large` —A, con pérdida cuadrática, y C, con clasificación sobre la rejilla más orden por
pares— y pesaban en el apilado +0,251 y +0,233. Pero el apilado se ajusta en
`split_t3 == "val"`, que tiene tareas de los **mismos proyectos** que vieron esos codificadores
al ajustarse: premiaba justo lo que no generaliza a un proyecto nuevo (la fuga de E54).

Medido fuera de pliegue, con `GroupKFold` por proyecto entero y los codificadores ajustados
**por pliegue** (E120, `248`–`251`). El ±25 % va sin ajustar a la rejilla, por eso queda por
debajo del 17 % de B.9:

| | ±25 % | SA | Spearman |
|---|---|---|---|
| e5-large base | 14,9 % | 0,0268 | 0,320 |
| base + ajustado C | 14,7 % | 0,0232 | 0,305 |
| solo el ajustado C | 14,4 % | 0,0054 | 0,193 |

Añadir el canal ajustado **empeora**. El v7 los quita y se queda con el base.

**En ONNX int8 y con el vocabulario podado.** De los 250.002 tokens de XLM-R se conservan
90.559: los que aparecen en el corpus más todo token de alfabeto latino. Solo con los del corpus
el 24 % de las tareas del generador tenía algún token podado; con los latinos, ninguna. Delante del modelo va una tabla
id_original → id_podado y un token ausente cae en `<unk>`. El codificador pasa de 2,2 GB a
**397 MB** y corre en CPU sin torch, con las métricas del bloque de prueba intactas (B.9).

**Y no el e5-small.** Con la misma receta fuera de pliegue el small empata en ±25 %
(−0,16 puntos, IC95 [−0,62, +0,24]) pero ordena peor: Spearman −0,037, IC95 [−0,053, −0,021],
y SA −0,0061. Con el large ya en 397 MB no compensa.

### B.5.2 · Cabeza A · horas por tarea (la mediana)

Tres canales apilados con `RidgeCV` (α = 7,85) ajustado **en validación**, no en
entrenamiento, para que los pesos no premien al canal que más sobreajusta:

| canal | peso |
|---|---|
| bolsa de 7 HGB sobre e5 + SVD (85 % de muestras, 80 % de rasgos cada uno) | **+0,759** |
| Ridge sobre TF-IDF disperso | +0,386 |
| LinearSVR sobre TF-IDF disperso | −0,032 |

El peso negativo del SVR no es un error: el apilado lo usa como **corrección** del Ridge, que ve
los mismos rasgos. Los modelos de canal son los del v6; solo se reajustó el apilado. La banda
por tarea al 80 % es ×4,45: el cuantil **conforme** del residuo de validación, con el apilado
ajustado por pliegues dentro de validación para no medirla sobre lo que ajustó.

La salida se exponencia y se multiplica por un **factor de estilo de 0,831**. El texto del
generador es más escueto que un título de Jira e infla las predicciones (el sesgo pasa de 1,02 a
1,20 con el mismo contenido reescrito al estilo del generador); el factor lo compensa, ajustado
en un bloque y verificado en el otro.

> **Abierto en el v7.** El factor es el heredado del v6. Sobre las 199 tareas de la prueba de
> extremo a extremo, la cabeza A del v7 da por tarea **×0,81** lo que daba la del v6; el
> total no cambia (lo calcula la cabeza B). Re-medirlo con el corpus de 800 títulos reescritos
> no sirve: lo reescribió un LLM local y no el generador, y el procedimiento no reproduce ni el
> valor del v6 (0,584 frente a 0,831). Cerrarlo pide tareas del generador
> actual con horas reales.

### B.5.3 · Cabeza B · horas para sumar (la media)

Cuatro canales, **todos con pérdida de Poisson sobre horas** (no sobre log), apilados con un
HGB también de Poisson ajustado en validación:

- bolsa de 5 HGB sobre e5 + SVD
- HGB sobre solo el SVD
- HGB sobre la predicción por analogía
- **`PoissonRegressor` lineal sobre el TF-IDF disperso crudo** (α = 1e−3)

El canal lineal disperso es el que menos se espera y sí aporta: recoge señal léxica literal
(`typo`, `bump`, `readme`) que los árboles sobre vectores densos diluyen.

### B.5.4 · Analogía y rejilla

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

El umbral de acuerdo entre vecinos (**1,656**, que cubre el 94,7 % de las tareas) sigue
decidiendo si el valor base es la analogía o el apilado, pero ya no decide si se ajusta.

**Lo que gana ajustar**, sobre el bloque de prueba de 4.225 tareas:

| | MdMRE | ±25 % | factor 2 | MAE | SA | Spearman |
|---|---|---|---|---|---|---|
| sin ajustar | 0,726 | 15,5 % | 0,403 | 5,11 | 0,0267 | 0,298 |
| **ajustado** | 0,730 | **17,0 %** | **0,424** | 5,11 | 0,0271 | 0,297 |

Sube el ±25 % y el factor 2 **sin mover el MAE, el SA ni el orden entre tareas**. No es un
modelo mejor: es la regla de decisión correcta para la métrica que se reporta.

---

## B.6 · Los mecanismos que usan información del proyecto

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

Con λ=12 encima de la estimación se **pierden 4–5 puntos**; con λ=96 se ganan **+1,5 puntos de
media**, y hasta **+3,1** con 20 cerradas.

> **Error fácil de cometer, y caro:** si el desplazamiento se estima comparando el real de las
> cerradas contra la predicción **en frío**, y luego se aplica sobre la predicción **corregida**
> por la estimación, el resultado se hunde a 49–58 % frente al 68 % de no anclar. **El ancla hay
> que estimarla contra la misma base sobre la que se va a aplicar.**

**Efecto medido sobre el total** (114 proyectos, sin estimación del usuario):

| tareas cerradas | MdMRE | ±25 % | MAE (h) |
|---|---|---|---|
| ninguna | 0,355 | 36,8 % | 455 |
| 3 | 0,319 | 40,7 % | 390 |
| 5 | 0,292 | 43,2 % | 377 |
| 10 | 0,260 | 48,4 % | 339 |
| 20 | 0,228 | 53,3 % | 312 |

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
| 3 | 39,8 % | **66,0 %** |
| 5 | 40,7 % | **66,4 %** |
| 10 | 44,6 % | **70,5 %** |
| 20 | 49,8 % | **70,3 %** |

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
| 1 | ×5,40 |
| 2 | ×5,02 |
| 3 | ×4,22 |
| 5 | ×3,65 |
| 10 | ×2,56 |
| 20 | ×2,14 |

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

### B.6.6 · Enganche con el generador de backlog

El estimador no recibe texto de Jira en produccion: recibe **lo que escribe el generador
de historias y tareas** de PBE. Son dos repositorios distintos y los une
`src/34_estimacion.py` (en el repositorio del generador), que toma la salida del
generador y le pone horas.

Dos detalles del enganche que no son opcionales:

**1 · Se le pasa el INGLES.** El generador trabaja en ingles internamente y solo traduce
al final, asi que el adaptador usa `h["tareas"]` y nunca `h["tareas_es"]`. Pasarle el
espanol seria salirse del contrato sin que nada fallase: solo acertaria menos.

**2 · El factor de estilo es CONDICIONAL al origen del texto, y cada cabeza tiene el
suyo.** Se midió sobre el texto del generador, que es más escueto que un título de Jira.
Sobre texto escrito por una persona el sesgo medido era ×1,002, o sea ninguno: aplicarlo ahí
solo encoge la predicción.

```python
est.estimar_tareas(textos, estilo="persona")     # por omisión: NO aplica factor
est.estimar_tareas(textos, estilo="generador")   # aplica 0,8313
```

Durante un tiempo se aplicó siempre, y el artefacto mostraba un **31 % menos de horas** de las
reales (sesgo ×0,820 en lugar de 1,0). Ninguna métrica del proyecto lo veía, porque todas se
calculan sobre un modelo limpio y no sobre el paquete. Corregido:

| 2.000 tareas reales de prueba | ±25 % | factor 2 | MAE | SA | sesgo |
|---|---|---|---|---|---|
| aplicando siempre el factor | 16,9 % | 45,5 % | 5,19 | 0,0625 | ×0,820 |
| **condicional (lo correcto)** | **17,5 %** | **49,0 %** | **5,15** | **0,0700** | **×1,000** |

> Dejar el sesgo dentro daba **más** ±25 % (18,9 % contra 17,5 %), porque la métrica premia
> subestimar. Se eligió la salida insesgada: para una herramienta de planificación, mostrar un
> 17 % menos de horas haría que los equipos planificaran corto de forma sistemática.

**El de la cabeza B es otro (E74).** La cabeza B no aplica nada, y el factor de la cabeza A no vale: son dos
modelos con perdidas distintas. Medido sobre las 800 tareas reales ya reescritas al
estilo del generador, todas fuera del entrenamiento, con cociente de sumas:

| texto de entrada | suma predicha / suma real |
|---|---|
| titulo original de Jira | **1,0018** — calibrado |
| el mismo contenido en estilo del generador | **1,3775** |
| **factor que lo corrige** | **0,7273** |

Sin esa correccion **el total del proyecto sale un 37 % alto**. Con ella, la media por
tarea de un backlog generado queda en 6,3 h, que es la media del corpus (6,11 h).

El factor se aplica **antes** del anclaje con tareas cerradas, no despues: la contraccion
de James-Stein hace que no se cancele solo.

---

## B.7 · Contrato de la interfaz

```python
est = EstimadorV6(paquete="models/estimador_pbe_v7.joblib")   # la clase sirve v6 y v7

est.estimar_tareas(textos,               # -> DataFrame
                   estilo="persona",     #   o "generador" si viene de la tubería 1
                   personas=None,        #   quién hará cada tarea; None donde no se sepa
                   historial=None,       #   {persona: [(texto, horas_reales), ...]}
                   estimaciones=None)    #   lo que estimó el usuario; None donde no
#   texto · horas · valor_probable · banda_baja · banda_alta
#   ajuste_persona · con_estimacion

est.estimar_proyecto(textos,             # -> dict
                     cerradas=None,      #   horas REALES ya registradas; None donde no
                     estimaciones=None,  #   lo que estimó el usuario; None donde no
                     nivel=0.8)
#   total_horas · horas_cerradas · horas_abiertas
#   banda_baja · banda_alta · nivel · tareas · cerradas
#   con_estimacion · lambda_usada · tramo_banda · k_banda · contraccion_tramo

est.tira_probabilidad(textos,            # -> lista por tarea de (desde_h, hasta_h, prob)
                      nivel=None, max_trozos=None)
```

Reglas del contrato, que importan:

- **El texto se espera en inglés**, solo el **título**, sin descripción. El generador de PBE ya
  produce inglés internamente y traduce solo para mostrar. Pasarle el español no falla: solo
  acierta menos.
- `cerradas`, `estimaciones` y `personas` son listas **de la misma longitud que `textos`**, con
  `None` en las posiciones que no aplican.
- `horas` es la mediana; `valor_probable` es un valor de la rejilla. **Ninguna de las dos es
  para sumar**: el total sale de `estimar_proyecto`, que usa la otra cabeza.
- `historial` solo debe traer tareas **del mismo proyecto** (ver B.6.4).
- Las rutas del ONNX, del tokenizador y de la tabla van **relativas al paquete**: la clase las
  resuelve desde la carpeta que contiene `models/`. No hace falta `HF_HOME` ni red.
- Los campos de diagnóstico (`lambda_usada`, `tramo_banda`, `k_banda`,
  `contraccion_tramo`, `ajuste_persona`, `con_estimacion`) existen para que la interfaz pueda
  explicar **de dónde sale** cada número, que es un criterio de aceptación de las historias de
  usuario.

### Lo que la interfaz debería mostrar

| situación | qué enseñar |
|---|---|
| backlog recién creado | total con su banda, marcado como provisional |
| el usuario estima tareas | indicar qué % lleva estimado y que el total mejora al completar |
| hay tareas cerradas | cuántas anclan el total, y que la banda se estrecha por eso |
| detalle de una tarea | horas, valor probable, banda, y opcionalmente la tira |
| tarea asignada | el factor de la persona y cuántas cerradas suyas lo sostienen |

---

## B.8 · Artefactos y rendimiento medido

```mermaid
flowchart TB
    subgraph SRV["Proceso de servicio"]
        API["EstimadorV6<br/>proceso Python"]
        PKG[("estimador_pbe_v7.joblib<br/>106 MB")]
        B[("models/e5_v7<br/>ONNX int8 · 397 MB<br/>+ tokenizador + tabla")]
        API --> PKG
        API --> B
    end
```

Un solo codificador para las dos rutas, dentro del paquete: no se descarga nada ni hace falta el
caché de Hugging Face. Medido en CPU, sin GPU y sin red (`255_medidas_servicio_v7.py`):

| | |
|---|---|
| carga del paquete | **3,0 s** |
| primera llamada a `estimar_tareas`, 42 tareas (abre el ONNX) | 12,2 s |
| `estimar_tareas`, lote de 42, ya cargado | **574 ms · 13,7 ms/tarea** |
| `estimar_proyecto`, 42 tareas, ya cargado | **529 ms** |
| memoria residente del estimador solo | 1,7 GB de RAM |
| generador + estimador, en CPU (paquete de producción) | 5,2 GB de RAM, pico |

Con el v6 eran tres codificadores de 2,2 GB y, en CPU, unos 10 GB de RAM para las dos tuberías.
La carga sigue siendo **perezosa**: en servicio conviene precalentar al arrancar.

### Por qué un modelo propio y no un LLM en línea

Banco medido sobre 601 tareas contra cuatro LLM locales (`gemma2:9b`, `llama3.1:8b`,
`mistral:7b`, `qwen2.5:7b`) en tres condiciones de prompt, **36 comparaciones pareadas**:

| veredicto | n |
|---|---|
| nuestro modelo mejor | 17 |
| empate | 18 |
| **el LLM mejor** | **1** |

Y el coste no se parece: el LLM tarda **0,31 a 2,45 s por tarea** frente a los **11 ms** medidos
aquí, entre **28× y 223× más lento**. La sonda de contaminación descarta que los LLM reconozcan
las incidencias: ninguno acertó la clave exacta en 450 intentos. (Los 11 ms son del v6 en GPU;
el v7 en CPU va a 13,7 ms por tarea.)

---

## B.9 · Calidad medida

### Total del proyecto — *la métrica que se presenta*

114 proyectos, **todos predichos como si fueran nuevos**:

| | MdMRE | ±25 % | factor 2 | MAE (h) |
|---|---|---|---|---|
| **en frío** | **0,355** | **36,8 %** | 72,8 % | 455 |
| con 3 tareas cerradas | 0,319 | 40,7 % | 80,8 % | 390 |
| con 5 tareas cerradas | 0,292 | 43,2 % | 84,4 % | 377 |
| con 10 tareas cerradas | 0,260 | 48,4 % | 87,4 % | 339 |
| con 20 tareas cerradas | 0,228 | 53,3 % | 87,2 % | 312 |

**Con estimación del usuario** (43 proyectos que la traen en ≥80 % de sus tareas):

| | ±25 % |
|---|---|
| en frío | 46,3 % |
| el usuario estima **todo** el backlog | 73,2 % |
| **+ 10 cerradas con λ=96** | **87,4 %** |

> **Cuidado con el 87,4 %.** Necesita **tres condiciones a la vez**: que el proyecto tenga
> estimación del usuario, que esté estimado *todo* el backlog, y que además haya 10 tareas
> cerradas. No es lo que da el sistema a un proyecto nuevo. La cifra de titular es **36,8 % en
> frío**, subiendo a **53,3 % con 20 cerradas**.

### Por tarea — bloque de prueba, 4.225 tareas

| | MdMRE | ±25 % | MAE (h) |
|---|---|---|---|
| línea base (mediana por proyecto) | 0,727 | 16,4 % | 5,26 |
| Deep-SE (LSTM) | 0,713 | 16,8 % | 5,19 |
| GPT2SP (GPT-2 ajustado) | 0,680 | 16,9 % | 5,05 |
| **nuestro, ajustado a la rejilla** | 0,730 | **17,0 %** | 5,11 |
| *una constante de 2,45 h* | *0,741* | *17,8 %* | *5,33* |

Dos lecturas, y las dos hay que darlas:

**Estamos por encima de lo publicado**, y con un protocolo **más estricto**: nuestras
predicciones son fuera de pliegue agrupando por **proyecto entero**, así que el modelo no vio ni
la tarea ni su proyecto; las reimplementaciones entrenan con tareas de los mismos proyectos.

**Pero una constante le gana a todo.** No es un resultado sobre el modelo sino sobre la métrica:
PRED(25) mide la dispersión y la curtosis de estimado/real, no la capacidad de ordenar
(Shepperd y Kitchenham). Por eso se reporta **siempre junto al SA y al Spearman**, donde esa
constante saca −0,015 y 0,000 y el modelo no.

### v7 contra v6, sobre el mismo bloque

Lo que sale del artefacto (no fuera de pliegue), sobre las mismas 4.225 tareas:

| | ±25 % horas | ±25 % valor probable | SA | Spearman | cobertura de la banda |
|---|---|---|---|---|---|
| v6 (3 codificadores) | 16,2 % | 18,7 % | 0,0645 | 0,421 | 77,9 % |
| v7 fp32 | 16,8 % | 18,7 % | 0,0546 | 0,398 | 77,9 % |
| **v7 int8 podado** (desplegado) | **16,5 %** | 17,7 % | 0,0528 | 0,399 | 78,0 % |

En este bloque el v6 **ordena algo mejor**, y es lo esperable: sus codificadores ajustados
vieron tareas de los mismos proyectos que el bloque. Para un proyecto nuevo la medida es la de
B.5.1, fuera de pliegue, donde quitarlos **mejora** el SA y el Spearman. El total del proyecto
no cambia entre v6 y v7: lo calcula la cabeza B, que nunca usó los codificadores ajustados.

### Lo que gana la suma

Las **mismas** predicciones de la cabeza B, medidas en las dos unidades sobre los mismos 114
proyectos:

| unidad | n | MdMRE | ±25 % |
|---|---|---|---|
| por tarea | 24.622 | 1,370 | 12,1 % |
| **por proyecto** | 114 | **0,355** | **36,8 %** |

**Sumar divide el MdMRE por 3,86.** Que la cabeza de la media salga mal por tarea no es un
defecto: con un objetivo asimétrico la media está por encima de la mediana, así que sobreestima
la tarea típica a propósito. Es el precio de poder sumarla, y es lo que justifica **dos cabezas
y no una**.

---

## B.10 · Límite conocido: la compresión de la escala

El diagnóstico, en una regresión:

```
log(real) = −0,831 + 1,001·log(nº de tareas) + 1,375·log(tamaño típico de tarea)
```

El **número de tareas se conoce exacto**. El **tamaño típico de tarea** se predice con solo el
**34 %** de su dispersión real (0,271 frente a 0,791). El modelo no puede saber por el texto si
un equipo trabaja en tickets de media hora o en tareas de ocho: **esa información no está en el
título**. De ahí el sesgo monótono por tamaño de tarea, de ×5,36 en proyectos de menos de
0,5 h/tarea a ×0,61 en los de más de 4 h.

**Se intentó corregir por siete vías y ninguna se sostiene:** mapeo de cuantiles, calibración
lineal en log, cambio de etiqueta (BBSE), un modelo de escala de proyecto propio, contraer el
anclaje hacia ese modelo, entrenar la escala con Apache, y una réplica independiente. El
contraste conjunto sobre **300 proyectos de dos corpus reales** (Fisher: Wilcoxon p = 0,24,
McNemar p = 0,67) **no muestra efecto**. Y la compresión **se replica y es peor** en Apache
(cubre el 19 % del recorrido, Spearman 0,108), así que es un fenómeno general y no un artefacto
de este corpus.

**Consecuencia de diseño, y por eso el anclaje es el mecanismo central:** la única información
sobre la escala que el texto no da la aporta **la primera tarea cerrada**.

---

## B.11 · Lo que se midió y se descartó

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
| codificadores ajustados por pliegue (A y C) | fuera de pliegue empeoran: SA 0,0268 → 0,0232 |
| e5-small en lugar de e5-large | empata en ±25 %, ordena peor: Spearman 0,283 contra 0,320 |

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

## B.13 · Operación

**Cómo presentarlo en producto.** Mostrar la estimación en frío como **provisional** y
recalcularla en cuanto el equipo cierre las primeras **tres** tareas. No hace falta esperar a
veinte: desde k = 3 la mejora es significativa en el 100 % de los sorteos.

**Qué vigilar.**

| señal | umbral | qué significa |
|---|---|---|
| tareas cerradas por debajo de la mediana del equipo | sistemático | el anclaje se sesgaría a la baja; es el único escenario que lo rompe |
| tamaño típico de tarea del proyecto | < 1 h | zona de sesgo ×1,7 a ×5,4; avisar de que la estimación es poco fiable |
| idioma del título | ≠ inglés | fuera de contrato |

**Reentrenamiento.** Por lotes, con las horas reales que va acumulando el time tracker. El
**codificador no se toca**; se reajustan las cabezas en el orden de
[B.3.7](#b37--orden-de-reconstrucción) y se repite la prueba de extremo a extremo.
La evaluación debe mantener siempre `GroupKFold` **por proyecto entero**.

---

## B.14 · Resumen de decisiones de diseño

| decisión | alternativa descartada | por qué |
|---|---|---|
| **dos cabezas**, mediana y media | una sola cabeza | sumar medianas deja el total en el 44 % |
| Poisson sobre horas para el total | log + smearing de Duan | mejora MdMRE, MAE, RMSE, ±50 % y factor 2 a la vez |
| apilado ajustado en **validación** | ajustado en entrenamiento | si no, los pesos premian al canal que más sobreajusta |
| **un solo** codificador, el base | tres (base + dos ajustados, v6) | fuera de pliegue los ajustados empeoran el SA y el orden; 6,5 GB → 397 MB |
| ONNX int8, vocabulario podado | fp32 completo | mismas métricas, 2,2 GB → 397 MB, sin torch ni caché de HF |
| codificador base **congelado** | ajustarlo de extremo a extremo | permite cachear vectores y reentrenar solo las cabezas |
| solo el **título**, en inglés | título + descripción | es lo que el generador produce |
| **sin Apache** en entrenamiento | corpus completo | su mediana de 0,67 h desplaza toda la escala |
| cociente de **sumas** para anclar | mediana de cocientes | con la mediana, más tareas cerradas daba **peor** resultado |
| corrector de **Poisson** para el usuario | corrector cuadrático | el cuadrático empeora la estimación cruda del usuario |
| **intervalo solo sobre lo abierto**, calibrado por k | multiplicar el total entero por el residuo en frío | lo cerrado se sabe; antes la banda era ×10,6 con 0 y con 20 cerradas |
| **λ condicional**: 12 sin estimación, 96 con ella | λ=12 siempre | con estimación, λ=12 pierde 4–5 puntos sobre lo abierto |
| **ajuste a la rejilla siempre**, 30 valores | solo cuando los vecinos concuerdan, 41 valores | +1,5 puntos de ±25 % sin mover MAE, SA ni orden |
| **factor de estilo condicional** al origen del texto | aplicarlo siempre | aplicado a texto de persona dejaba el sesgo en ×0,82 |
| ajuste por persona **solo en la vista por tarea** | aplicarlo también al total | al sumar se cancelan y empeoran lo abierto 1,1 puntos |
| historial **solo del proyecto en curso** | usar proyectos pasados del equipo | medido: de −1,8 a −3,7 puntos |
| modelo propio | LLM local en línea | gana 17 de 36 comparaciones, empata 18, y es 28–223× más rápido |
