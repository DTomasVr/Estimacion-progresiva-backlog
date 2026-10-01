# Diseño de la tubería de estimación de esfuerzo · Agix

**Modelo documentado:** `estimador_agix_v6` · corpus v4 sin Apache · construido el 2026-09-29
**Alcance:** únicamente el diseño vigente. Las alternativas medidas y descartadas están en
`INFORME_PROYECTO.md`; aquí solo aparecen cuando explican por qué el diseño es así.

Este documento reúne **todo el diseño actual en un solo sitio**: contexto, diagramas de flujo,
arquitectura interna, contrato, datos, rendimiento medido, calidad, límites y operación.
Sustituye por completo a `DIAGRAMAS_C4.md`, que describía una versión anterior (codificador
*mpnet* de 768 dimensiones, `frio_v4.joblib`, un solo regresor).

---

## Índice

1. [Qué hace, y en una tabla](#1--qué-hace-y-en-una-tabla)
2. [La decisión que estructura todo: dos cabezas](#2--la-decisión-que-estructura-todo-dos-cabezas)
3. [Diagramas de flujo](#3--diagramas-de-flujo)
4. [Los datos](#4--los-datos)
5. [Arquitectura interna](#5--arquitectura-interna)
6. [Los tres mecanismos que usan información del proyecto](#6--los-tres-mecanismos-que-usan-información-del-proyecto)
7. [Contrato de la interfaz](#7--contrato-de-la-interfaz)
8. [Artefactos y rendimiento medido](#8--artefactos-y-rendimiento-medido)
9. [Calidad medida](#9--calidad-medida)
10. [Límite conocido: la compresión de la escala](#10--límite-conocido-la-compresión-de-la-escala)
11. [Operación](#11--operación)
12. [Resumen de decisiones de diseño](#12--resumen-de-decisiones-de-diseño)

---

## 1 · Qué hace, y en una tabla

Agix descompone un proyecto en tareas con un LLM. Este componente recibe **el texto de esas
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

## 2 · La decisión que estructura todo: dos cabezas

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
**el servidor v6 no lo usa**.

---

## 3 · Diagramas de flujo

### 3.1 · Flujo de extremo a extremo

```mermaid
flowchart LR
    U(["Usuario"]) -->|describe el proyecto| A["Agix · web"]
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

### 3.2 · Camino de una petición, con las dos rutas

El diagrama que más importa: **las dos preguntas no recorren el mismo camino**.

```mermaid
flowchart TB
    IN(["lista de titulos"]) --> NORM["normalizar<br/>titulo, ingles, sin descripcion"]
    NORM --> VIS["codificar: 3 vistas del texto"]
    VIS --> RA{"que se pregunta?"}

    RA -->|cuanto tarda ESTA tarea| CA["Cabeza A · MEDIANA<br/>apilado de 5 canales"]
    CA --> EST["x factor de estilo 0,831"]
    EST --> BND["banda x4,32"]
    VIS --> ANA["analogia: 25 vecinos<br/>enganchada a la rejilla"]
    ANA --> VP{"vecinos<br/>de acuerdo?"}
    VP -->|si, el 85,3 %| VPS["valor_probable = rejilla"]
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
    TOT --> INT["intervalo al 80 %<br/>x0,36 a x2,01"]
    INT --> OUT2(["total_horas · banda_baja · banda_alta"])

    style CA fill:#f2e9d5,stroke:#7d5a17
    style CB fill:#dcebe9,stroke:#12645d
    style ANC fill:#f4e2dc,stroke:#a4442a
```

### 3.3 · Flujo interno del modelo

```mermaid
flowchart TB
    subgraph COD["1 · Codificacion del titulo"]
        direction LR
        T["titulo"] --> E5["e5-large congelado<br/>1024 dim"]
        T --> TF["TF-IDF caracteres 3-5<br/>+ palabras 1-2"]
        TF --> SVD["SVD 256 + 256"]
        T --> AJ["e5 ajustado A<br/>escalar cuadratico"]
        T --> AC["e5 ajustado C<br/>clasificacion + orden por pares"]
    end

    subgraph CA["2a · Cabeza A · horas por TAREA (mediana)"]
        direction TB
        A1["bolsa de 7 HGB<br/>e5 + SVD"] --> AP{{"apilado RidgeCV<br/>ajustado en validacion"}}
        A2["Ridge disperso"] --> AP
        A3["SVR disperso"] --> AP
        A4["HGB sobre e5 ajustado A"] --> AP
        A5["HGB sobre e5 ajustado C"] --> AP
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

### 3.4 · Nivel 1 · Contexto

```mermaid
C4Context
title Nivel 1 · Contexto del componente de estimación

Person(dev, "Desarrollador / líder de proyecto", "Crea proyectos, revisa las tareas generadas y registra sus horas")
Person(tesista, "Responsable del modelo", "Construye el corpus, entrena y evalúa")

System(estimacion, "Componente de estimación de esfuerzo", "Horas por tarea y total del proyecto a partir del texto; ancla con las tareas ya cerradas")

System_Ext(agix, "Agix · aplicación web", "Gestión de proyectos: backlog, tablero y time tracker")
System_Ext(llm, "Generador de tareas (LLM)", "Descompone el proyecto en tareas")
System_Ext(corpus, "Corpus públicos", "TAWOS, The Public Jira Dataset, SiP (Jones)")
System_Ext(hf, "Hugging Face Hub", "Pesos de intfloat/multilingual-e5-large")

Rel(dev, agix, "Crea proyectos y registra horas", "HTTPS")
Rel(agix, llm, "Pide la descomposición en tareas", "HTTP/JSON")
Rel(agix, estimacion, "Pide horas por tarea y total del proyecto", "HTTP/JSON")
Rel(estimacion, agix, "Devuelve horas, total e intervalo", "HTTP/JSON")
Rel(agix, estimacion, "Horas reales de las tareas cerradas", "en cada recálculo")
Rel(tesista, estimacion, "Entrena, selecciona y evalúa", "scripts")
Rel(corpus, estimacion, "Alimentan el entrenamiento", "volcados SQL / mongodump")
Rel(hf, estimacion, "Provee el codificador base congelado", "descarga única")

UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="2")
```

### 3.5 · Nivel 2 · Contenedores en ejecución

```mermaid
C4Container
title Nivel 2 · Contenedores en tiempo de ejecución (v6)

Person(dev, "Desarrollador", "Usa Agix")
System_Ext(agix, "Agix · aplicación web", "Backend + frontend")

Container_Boundary(est, "Componente de estimación de esfuerzo") {
  Container(api, "EstimadorV6", "Python", "Dos entradas: estimar_tareas y estimar_proyecto; carga perezosa de codificadores")
  Container(encbase, "Codificador base", "PyTorch · multilingual-e5-large CONGELADO", "Texto a vector de 1024. Lo usan las DOS rutas")
  Container(encaj, "Codificadores ajustados A y C", "PyTorch · e5-large reajustado", "Solo para la ruta por tarea. 2,2 GB cada uno")
  Container(cabA, "Cabeza A · mediana", "scikit-learn · apilado RidgeCV de 5 canales", "Horas de UNA tarea, con banda x4,32")
  Container(cabB, "Cabeza B · media", "scikit-learn · apilado HGB de Poisson de 4 canales", "Horas que se pueden SUMAR")
  Container(analogia, "Índice de analogía", "numpy · 16.648 vectores float16", "25 vecinos más próximos, mediana enganchada a la rejilla")
  Container(anclaje, "Anclaje por proyecto", "Python", "Cociente de sumas contraído con James-Stein, lambda = 12")
  Container(corrector, "Corrector de la estimación del usuario", "scikit-learn · HGB de Poisson", "Vector + log(estimación) + log(predicción en frío)")
  Container(intervalo, "Intervalo del total", "Python · cuantiles empíricos", "Residuos en log de los 114 proyectos")
  ContainerDb(artefactos, "Almacén de artefactos", "joblib + carpetas de pesos", "estimador_agix_v6.joblib de 221 MB, mas models/e5_ajustado y _C")
  ContainerDb(historial, "Historial de tareas", "base de datos de Agix", "Tareas cerradas con horas reales y estimaciones del usuario")
  Container(reentreno, "Reentrenamiento por lotes", "Python", "Reajusta las cabezas; el codificador base NO se toca")
}

Rel(dev, agix, "Crea el proyecto y registra horas", "HTTPS")
Rel(agix, api, "estimar_tareas / estimar_proyecto", "JSON")
Rel(api, encbase, "Texto de las tareas", "en proceso")
Rel(api, encaj, "Solo en la ruta por tarea", "en proceso")
Rel(encbase, cabB, "Vectores", "numpy")
Rel(encaj, cabA, "Vectores ajustados", "numpy")
Rel(encbase, analogia, "Vector normalizado", "numpy")
Rel(analogia, cabA, "Valor de la rejilla", "numpy")
Rel(analogia, cabB, "Canal de analogía", "numpy")
Rel(cabB, corrector, "Predicción en frío como rasgo", "numpy")
Rel(cabB, anclaje, "Horas por tarea para sumar", "numpy")
Rel(historial, anclaje, "Horas reales de las cerradas", "JSON")
Rel(anclaje, intervalo, "Total puntual", "numpy")
Rel(intervalo, api, "Total y banda", "numpy")
Rel(api, artefactos, "Carga al arrancar", "joblib")
Rel(agix, historial, "Horas reales al cerrar", "SQL")
Rel(reentreno, historial, "Lee tareas cerradas", "SQL")
Rel(reentreno, artefactos, "Escribe el artefacto nuevo", "joblib")

UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

> **Decisión que se ve aquí:** el codificador base está **congelado** y los codificadores
> ajustados **solo hacen falta para la ruta por tarea**. Un despliegue que solo necesite el
> total del proyecto —que es la métrica que se presenta— ahorra **4,4 GB**.

### 3.6 · Nivel 2b · Tubería de datos y entrenamiento (fuera de línea)

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
  ContainerDb(procesado, "Dataset", "Parquet · agix_effort_v4.parquet", "90.975 filas: train 63.926, val 8.445, test 4.225, externo 13.255")
  Container(codificacion, "Codificación con caché", "Python · 53_encode_v4, 98_codificadores", "Reutiliza vectores calculados; solo codifica el texto nuevo")
  ContainerDb(vectores, "Vectores", "npy · embeddings/codificadores/", "e5-large.npy y los dos ajustados, alineados por fila")
  Container(finetune, "Ajuste de codificadores", "PyTorch · 147_finetune_rapido", "Brazos A y C sobre e5-large, con la tabla de embeddings congelada")
  Container(entrenamiento, "Entrenamiento de las cabezas", "Python · 163_modelo_v5, 171_estimador_v6", "Ajusta los canales y el apilado; elige SIEMPRE en validación")
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

### 3.7 · Orden de reconstrucción

**El orden importa.** Si cambian los pesos de un codificador ajustado, cambian sus vectores, y
la cabeza que los consume queda ajustada sobre los antiguos. Servir con un codificador que no
case con su cabeza **no falla: solo acierta menos**.

```mermaid
flowchart LR
    F["147_finetune_rapido.py<br/>reentrena el codificador C"] --> V5["163_modelo_v5.py<br/>apilado por tarea"]
    V5 --> V6["171_estimador_v6.py<br/>paquete de servicio"]
    V6 --> P{{"prueba de extremo a extremo<br/>estima de verdad, con y sin cerradas"}}
    P -->|verde| OK(["artefacto desplegable"])
    P -->|roja| STOP(["no se despliega"])
    ORQ["177_rehacer_artefacto.py<br/>encadena y se niega si faltan requisitos"] -.-> V5
    style P fill:#f2e9d5,stroke:#7d5a17
    style STOP fill:#f4e2dc,stroke:#a4442a
```

La prueba final no es opcional: **es lo único que demuestra que el artefacto sirve**. La primera
vez que se construyó el v6 parecía correcto y no podía codificar texto nuevo, porque los pesos
del codificador nunca se habían guardado. Lo destapó justamente esa prueba.

### 3.8 · Vista dinámica · el ciclo de vida de un proyecto

```mermaid
sequenceDiagram
    autonumber
    actor U as Desarrollador
    participant W as Agix (web)
    participant L as Generador (LLM)
    participant A as EstimadorV6
    participant D as Historial

    U->>W: Crea el proyecto
    W->>L: Descompón en tareas
    L-->>W: Lista de títulos
    W->>A: estimar_tareas(titulos)
    A-->>W: horas, valor_probable y banda por tarea
    W->>A: estimar_proyecto(titulos)
    A-->>W: total en frío + intervalo al 80 % (x0,36 a x2,01)
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

## 4 · Los datos

### El dataset completo

| | |
|---|---|
| fichero | `data/processed/agix_effort_v4.parquet` · 90.975 filas · 52 columnas · 533 proyectos |
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
existen cuando se crea la tarea: `source` (Agix sería un nivel no visto), recuentos de enlaces y
de componentes (se acumulan mientras la tarea se trabaja, y valen cero al crearla), y story
points.

---

## 5 · Arquitectura interna

### 5.1 · Codificación del título

Tres vistas del mismo texto, porque cada una acierta en cosas distintas:

| vista | qué es | dimensiones |
|---|---|---|
| **semántica** | `intfloat/multilingual-e5-large`, **congelado**, prefijo `query: ` | 1024 |
| **léxica densa** | TF-IDF de caracteres `char_wb` 3–5 (vocab 35.450) + palabras 1–2 (vocab 20.256), `sublinear_tf`, reducidos con SVD | 256 + 256 |
| **léxica dispersa** | el mismo TF-IDF **sin reducir**, para los modelos lineales | ~55.700 |

Además, **dos codificadores ajustados** sobre el mismo `e5-large`, que aportan cosas distintas
y por eso van los dos:

- **A** — ajustado con un escalar de pérdida cuadrática: estima mejor la **escala**.
- **C** — ajustado con cabeza de clasificación sobre la rejilla más una pérdida de **ordenación
  por pares**: **ordena** mejor.

En el ajuste se **congela la tabla de embeddings** (es la mitad de los parámetros y no aporta en
títulos de ~48 piezas), se recorta cada lote a su longitud real y se entrena en bf16.

### 5.2 · Cabeza A · horas por tarea (la mediana)

Cinco canales apilados con `RidgeCV` (α = 48,3) ajustado **en validación**, no en
entrenamiento, para que los pesos no premien al canal que más sobreajusta:

| canal | peso |
|---|---|
| bolsa de 7 HGB sobre e5 + SVD (85 % de muestras, 80 % de rasgos cada uno) | **+0,280** |
| HGB sobre e5 **ajustado A** | **+0,251** |
| HGB sobre e5 **ajustado C** | **+0,233** |
| Ridge sobre TF-IDF disperso | +0,188 |
| LinearSVR sobre TF-IDF disperso | −0,015 |

El peso negativo del SVR no es un error: el apilado lo usa como **corrección** del Ridge, que ve
los mismos rasgos.

La salida se exponencia y se multiplica por un **factor de estilo de 0,831**. El texto del
generador es más escueto que un título de Jira e infla las predicciones (el sesgo pasa de 1,02 a
1,20 con el mismo contenido reescrito al estilo del generador); el factor lo compensa, ajustado
en un bloque y verificado en el otro.

### 5.3 · Cabeza B · horas para sumar (la media)

Cuatro canales, **todos con pérdida de Poisson sobre horas** (no sobre log), apilados con un
HGB también de Poisson ajustado en validación:

- bolsa de 5 HGB sobre e5 + SVD
- HGB sobre solo el SVD
- HGB sobre la predicción por analogía
- **`PoissonRegressor` lineal sobre el TF-IDF disperso crudo** (α = 1e−3)

El canal lineal disperso es el que menos se espera y sí aporta: recoge señal léxica literal
(`typo`, `bump`, `readme`) que los árboles sobre vectores densos diluyen.

### 5.4 · Analogía y rejilla

Para cada tarea se buscan los **25 vecinos más próximos** por coseno entre los 16.648 vectores
de entrenamiento (guardados en `float16`) y se toma la **mediana** de sus horas reales. Ese
número se **engancha a la rejilla** de los 41 valores que aparecen al menos 60 veces en el
entrenamiento:

```
0,083 0,1 0,167 0,25 0,333 0,417 0,5 0,583 0,667 0,7 0,75 0,833 1 1,167 1,25 1,333
1,5 1,75 2 2,25 2,5 2,75 3 3,5 4 4,5 5 5,5 6 6,5 7 8 9 10 11 12 14 16 24 32 40
```

El objetivo está fuertemente cuantizado: la gente registra 1 h, 2 h, medio día. Un modelo de
regresión devuelve 1,83 h y no acierta ninguno de los valores que la gente escribe de verdad.
Cuando los vecinos **están de acuerdo** —desviación de `log(horas)` entre ellos ≤ **1,524**, que
cubre el **85,3 %** de las tareas (medido sin contar la propia tarea como vecina, que es lo que
pasa en servicio)— se muestra el valor de la rejilla como `valor_probable`; si no, la predicción
del apilado. El umbral se eligió maximizando el ±25 % en validación.

---

## 6 · Los tres mecanismos que usan información del proyecto

### 6.1 · Anclaje con tareas cerradas — *el mecanismo más valioso del diseño*

Cuando el equipo ya cerró tareas, sus horas son **un dato**, no una estimación:

```python
k     = número de tareas cerradas
off   = log( Σ horas_reales(cerradas) / Σ predicción(cerradas) ) × k / (k + 12)
total = Σ horas_reales(cerradas)  +  Σ predicción(abiertas) × exp(off)
```

Tres decisiones aquí, y **las tres costaron encontrarlas**:

1. **Cociente de sumas**, no mediana de cocientes. Si se estima con la mediana de
   `log(real/predicho)` y el estimador predice la media, esa mediana es sistemáticamente
   negativa. Anclando así, cuantas más tareas cerradas **peor**: con 20 el sesgo llegaba a
   −45,7 %.
2. **Las cerradas se saben, no se estiman.** Sumar la predicción de lo ya cerrado es tirar
   información.
3. **Contracción de James-Stein con λ = 12.** Con `k` pequeño el cociente es ruidoso. Se barrió
   λ de 0 a 24: sin contracción el MdMRE con una tarea cerrada es **0,726** en vez de **0,347**.

**Efecto medido:**

| tareas cerradas | orden de la escala (Spearman) | MdMRE | ±25 % | Wilcoxon vs frío |
|---|---|---|---|---|
| ninguna | 0,444 | 0,355 | 36,8 % | — |
| 3 | **0,700** | 0,319 | 40,7 % | **p = 0,0003 (100 % de sorteos)** |
| 5 | **0,769** | 0,292 | 43,2 % | **p < 0,0001 (100 %)** |

**Y no depende de que el equipo cierre al azar.** Usando el **orden real** de cierre de los 82
proyectos que tienen fecha, el sesgo es −1,6 % (frente a −0,3 % del sorteo) y la exactitud
*mejora*. El único escenario que rompería el anclaje —despachar sistemáticamente lo más corto
primero, −80,5 % con tres cerradas— no ocurre en este corpus y **es observable en producción**.

### 6.2 · Corrección de la estimación del usuario

Si el formulario pide una estimación, se corrige con un HGB de **Poisson** sobre
`[vector e5 | log(estimación) | log(predicción en frío)]`.

**Tiene que ser el de Poisson.** El corrector de error cuadrático, que es el que sirve por
tarea, **empeora** la estimación cruda del usuario a nivel de proyecto (sesgo −35,5 %). Sobre
los 43 proyectos con estimación en al menos el 80 % de sus tareas, el ±25 % pasa de **44,2 %**
en frío a **65,1 %**.

### 6.3 · Intervalo del total

Cuantiles empíricos de `log(real / estimado)` sobre los 114 proyectos de evaluación. Al 80 %:
**×0,36 a ×2,01**.

> **Limitación conocida y deliberada:** el intervalo se calcula siempre con los residuos **en
> frío**, aunque haya tareas cerradas. Es decir, **es conservador cuando hay anclaje**: la
> estimación puntual mejora pero la banda no se estrecha. Recalibrar por `k` requeriría más
> proyectos de los que hay.

---

### 6,4 · Enganche con el generador de backlog

El estimador no recibe texto de Jira en produccion: recibe **lo que escribe el generador
de historias y tareas** de Agix. Son dos repositorios distintos y los une
`src/34_estimacion.py` (en el repositorio del generador), que toma la salida del
generador y le pone horas.

Dos detalles del enganche que no son opcionales:

**1 · Se le pasa el INGLES.** El generador trabaja en ingles internamente y solo traduce
al final, asi que el adaptador usa `h["tareas"]` y nunca `h["tareas_es"]`. Pasarle el
espanol seria salirse del contrato sin que nada fallase: solo acertaria menos.

**2 · El factor de estilo del total NO es el de la mediana (E74).** La cabeza A
multiplica por 0,8313 para compensar que el texto del generador es mas escueto que un
titulo de Jira. La cabeza B no aplica nada, y el factor de la cabeza A no vale: son dos
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

## 7 · Contrato de la interfaz

```python
est = EstimadorV6()                      # carga models/estimador_agix_v6.joblib

est.estimar_tareas(textos)               # -> DataFrame
#   texto · horas · valor_probable · banda_baja · banda_alta

est.estimar_proyecto(textos,             # -> dict
                     cerradas=None,      #   horas REALES ya registradas; None donde no
                     estimaciones=None,  #   lo que estimó el usuario; None donde no
                     nivel=0.8)
#   total_horas · banda_baja · banda_alta · nivel · tareas · cerradas
```

Reglas del contrato, que importan:

- **El texto se espera en inglés** (`idioma_esperado`), solo el **título** (`vista`), sin
  descripción. El generador de Agix ya produce inglés internamente y traduce solo para mostrar.
- `cerradas` y `estimaciones` son listas **de la misma longitud que `textos`**, con `None` en
  las posiciones que no aplican.
- `horas` es la mediana; `valor_probable` es un valor de la rejilla cuando los vecinos están de
  acuerdo. **Ninguna de las dos es para sumar**: el total sale de `estimar_proyecto`, que usa la
  otra cabeza.

---

## 8 · Artefactos y rendimiento medido

```mermaid
flowchart TB
    subgraph SRV["Proceso de servicio"]
        API["EstimadorV6<br/>proceso Python"]
        PKG[("estimador_agix_v6.joblib<br/>221 MB")]
        B["e5-large congelado<br/>descarga de HF"]
        FA[("models/e5_ajustado<br/>2,2 GB")]
        FC[("models/e5_ajustado_C<br/>2,2 GB")]
        API --> PKG
        API --> B
        API -.->|solo para estimar_tareas| FA
        API -.->|solo para estimar_tareas| FC
    end
    style FA stroke-dasharray: 5 5
    style FC stroke-dasharray: 5 5
```

**Los dos codificadores ajustados solo hacen falta para `estimar_tareas`.** La ruta del total
—`estimar_proyecto`, que es la métrica que se presenta— usa **únicamente el codificador base**.
Verificado en proceso limpio: tras estimar un proyecto de 42 tareas, A y C siguen sin cargar.

Medido en este equipo, con GPU CUDA disponible:

| | |
|---|---|
| carga del paquete joblib | **4,5 s** |
| primera llamada a `estimar_tareas` (carga los 3 codificadores) | 44 s |
| primera llamada a `estimar_proyecto` (carga solo el base) | 31 s |
| `estimar_tareas`, lote de 42 | **462 ms · 11 ms/tarea** |
| `estimar_proyecto`, 42 tareas, ya cargado | **200–275 ms** |
| memoria residente del proceso | ~1,9 GB de RAM |

Los pesos de los codificadores viven en VRAM con GPU y no cuentan en esa RAM; en CPU hay que
sumar ~2,2 GB por codificador cargado. La carga es **perezosa**: en servicio conviene
precalentar al arrancar, no en la primera petición del usuario.

> **Optimización pendiente, no aplicada:** el paquete arrastra de la versión 3 una matriz
> `centros` de **68 MB** y los objetos `corrector`, `smearing` y `suma_residual`, que el
> servidor v6 **no usa**. Quitarlos dejaría el artefacto en ~150 MB.

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
las incidencias: ninguno acertó la clave exacta en 450 intentos.

---

## 9 · Calidad medida

### Total del proyecto — *la métrica que se presenta*

114 proyectos, **todos predichos como si fueran nuevos**:

| | MAE (h) | RMSE | MdMRE | R²log | ±25 % | factor 2 | sesgo |
|---|---|---|---|---|---|---|---|
| constante (no lee el texto) | 886 | 2046 | 0,694 | 0,364 | 8,8 % | 24,6 % | −203 % |
| **en frío** | **455** | 1090 | **0,355** | **0,817** | **36,8 %** | **72,8 %** | −0,3 % |
| **con 5 tareas cerradas** | **379** | 991 | **0,291** | **0,911** | **43,7 %** | **85,5 %** | +0,2 % |
| con 20 tareas cerradas | 291 | 795 | 0,219 | 0,908 | 54,3 % | 87,5 % | +1,5 % |

### Por tarea — prueba fija, 4.225 tareas

| MAE (h) | SA | Δ de Glass | ζ | R²log | ±25 % | factor 2 | Spearman |
|---|---|---|---|---|---|---|---|
| 4,91 | 0,432 | 0,204 | 119 % | 0,202 | 16,1 % | 44,4 % | 0,421 |

### Lo que gana la suma (E73)

Las **mismas** predicciones de la cabeza B, medidas en las dos unidades sobre los mismos 114
proyectos:

| unidad | n | MdMRE | ±25 % | factor 2 |
|---|---|---|---|---|
| por tarea | 24.622 | 1,370 | 12,1 % | 33,1 % |
| **por proyecto** | 114 | **0,355** | **36,8 %** | **72,8 %** |

**Sumar divide el MdMRE por 3,86.** Que la cabeza de la media salga mal por tarea no es un
defecto: con un objetivo asimétrico la media está por encima de la mediana, así que sobreestima
la tarea típica a propósito. Es el precio de poder sumarla, y es exactamente lo que justifica
**dos cabezas y no una** — por eso la columna «horas por tarea» que ve el usuario sale de la
cabeza A, no de esta.

**Lectura honesta:** tarea a tarea el tamaño del efecto frente al azar es **pequeño** (Δ de
Glass 0,204). El valor del modelo está en **el total del proyecto** y en corregir la estimación
del usuario, no en clavar una tarea suelta.

---

## 10 · Límite conocido: la compresión de la escala

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

## 11 · Operación

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
**codificador base no se toca**; se reajustan las cabezas y, si se rehacen los codificadores
ajustados, hay que repetir la cadena completa en el orden de [3.7](#37--orden-de-reconstrucción).
La evaluación debe mantener siempre `GroupKFold` **por proyecto entero**.

---

## 12 · Resumen de decisiones de diseño

| decisión | alternativa descartada | por qué |
|---|---|---|
| **dos cabezas**, mediana y media | una sola cabeza | sumar medianas deja el total en el 44 % |
| Poisson sobre horas para el total | log + smearing de Duan | mejora MdMRE, MAE, RMSE, ±50 % y factor 2 a la vez |
| apilado ajustado en **validación** | ajustado en entrenamiento | si no, los pesos premian al canal que más sobreajusta |
| **dos** codificadores ajustados | solo uno | uno ordena mejor y el otro estima mejor la escala |
| codificador base **congelado** | ajustarlo de extremo a extremo | permite cachear vectores y reentrenar solo las cabezas |
| solo el **título**, en inglés | título + descripción | es lo que el generador produce |
| **sin Apache** en entrenamiento | corpus completo | su mediana de 0,67 h desplaza toda la escala |
| cociente de **sumas** para anclar | mediana de cocientes | con la mediana, más tareas cerradas daba **peor** resultado |
| corrector de **Poisson** para el usuario | corrector cuadrático | el cuadrático empeora la estimación cruda del usuario |
| intervalo en frío siempre | recalibrar por k | no hay proyectos suficientes; se prefiere conservador |
| modelo propio | LLM local en línea | gana 17 de 36 comparaciones, empata 18, y es 28–223× más rápido |
