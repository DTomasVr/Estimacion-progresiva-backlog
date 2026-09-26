# Diagramas C4 · Componente de estimación de esfuerzo de Agix

Cuatro vistas: contexto, contenedores en ejecución, contenedores de la tubería de datos y
componentes internos, más una vista dinámica de una petición. Sintaxis Mermaid C4 (se
renderiza en GitHub, VS Code y la mayoría de visores Markdown).

---

## Nivel 1 · Contexto

```mermaid
C4Context
title Nivel 1 · Contexto del componente de estimación

Person(dev, "Desarrollador / líder de proyecto", "Crea proyectos, revisa las tareas generadas y registra sus horas")
Person(tesista, "Responsable del modelo", "Construye el corpus, entrena y evalúa")

System(estimacion, "Componente de estimación de esfuerzo", "Predice las horas de una tarea a partir de su texto; opcionalmente corrige la estimación que da el usuario")

System_Ext(agix, "Agix · aplicación web", "Gestión de proyectos: backlog, tablero y time tracker")
System_Ext(llm, "Generador de tareas (LLM)", "Ollama local / Groq. Descompone el proyecto en tareas")
System_Ext(corpus, "Corpus públicos", "TAWOS, The Public Jira Dataset, SiP (Jones)")
System_Ext(hf, "Hugging Face Hub", "Pesos del codificador multilingüe")

Rel(dev, agix, "Crea proyectos y registra horas", "HTTPS")
Rel(agix, llm, "Pide la descomposición en tareas", "HTTP/JSON")
Rel(agix, estimacion, "Pide horas y banda por tarea", "HTTP/JSON")
Rel(estimacion, agix, "Devuelve horas, banda y confianza", "HTTP/JSON")
Rel(agix, estimacion, "Envía las horas reales al cerrar la tarea", "lote nocturno")
Rel(tesista, estimacion, "Entrena, selecciona y evalúa", "scripts")
Rel(corpus, estimacion, "Alimentan el entrenamiento", "volcados SQL / mongodump")
Rel(hf, estimacion, "Provee el codificador congelado", "descarga única")

UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="2")
```

---

## Nivel 2 · Contenedores en ejecución (lo que se despliega)

```mermaid
C4Container
title Nivel 2 · Contenedores en tiempo de ejecución

Person(dev, "Desarrollador", "Usa Agix")
System_Ext(agix, "Agix · aplicación web", "Backend FastAPI + frontend")

Container_Boundary(est, "Componente de estimación de esfuerzo") {
  Container(api, "API de estimación", "Python · FastAPI", "Un endpoint por tarea y uno por lote; valida entrada y compone la respuesta")
  Container(encoder, "Codificador SBERT", "PyTorch · paraphrase-multilingual-mpnet-base-v2 congelado", "Texto → vector de 768 dimensiones. 9 ms por tarea en GPU, 1,13 GB de pesos")
  Container(regresor, "Regresor de horas", "scikit-learn · HistGradientBoosting", "Vector + categoría + numéricas → log-horas. 1,3 MB, 2,4 ms por predicción")
  Container(corrector, "Corrector de la estimación del usuario", "scikit-learn · HGB (brazo C1)", "Estimación del usuario + predicción en frío + texto → factor de corrección")
  Container(bandas, "Calibrador conforme", "Python · cuantiles sobre validación", "Banda multiplicativa al 80 % y escalado en línea por la mediana del proyecto")
  ContainerDb(artefactos, "Almacén de artefactos", "Ficheros · joblib + npy", "models/frio_v4.joblib, vectores cacheados, cuantiles de calibración")
  ContainerDb(historial, "Historial de tareas", "PostgreSQL", "Tareas cerradas con horas reales y estimaciones del usuario")
  Container(reentreno, "Trabajo de reentrenamiento", "Python · por lotes", "Reajusta el regresor cuando entran horas nuevas; el codificador no se toca")
}

Rel(dev, agix, "Crea la tarea y, si quiere, escribe su estimación", "HTTPS")
Rel(agix, api, "POST /estimar", "JSON")
Rel(api, encoder, "Texto de la tarea", "llamada en proceso")
Rel(encoder, regresor, "Vector de 768", "numpy")
Rel(api, corrector, "Si el usuario dio su estimación", "llamada en proceso")
Rel(regresor, corrector, "Predicción en frío como rasgo", "numpy")
Rel(api, bandas, "Predicción puntual", "llamada en proceso")
Rel(bandas, api, "Rango a mostrar", "numpy")
Rel(api, historial, "Guarda la predicción emitida", "SQL")
Rel(agix, historial, "Horas reales al cerrar la tarea", "SQL")
Rel(reentreno, historial, "Lee tareas cerradas", "SQL")
Rel(reentreno, artefactos, "Escribe el regresor nuevo", "joblib")
Rel(regresor, artefactos, "Carga el modelo al arrancar", "joblib")
Rel(bandas, artefactos, "Carga los cuantiles", "json")

UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

**Decisión de diseño que se ve aquí:** el codificador está congelado y solo el regresor (1,3 MB,
20 s de ajuste) se reentrena. Ajustar el codificador de extremo a extremo se probó y no mejoró
nada (MdMRE 0,686 frente a 0,685).

---

## Nivel 2b · Contenedores de la tubería de datos (fuera de línea)

```mermaid
C4Container
title Nivel 2b · Tubería de datos y entrenamiento

System_Ext(tawos, "TAWOS", "Volcado SQL, 458.232 incidencias, 39 proyectos")
System_Ext(jira, "The Public Jira Dataset", "mongodump de 5,87 GB, 2.686.282 incidencias")
System_Ext(sip, "SiP (Jones)", "CSV de una empresa real, horas reales y estimadas")

Container_Boundary(tuberia, "Tubería de datos") {
  Container(extraccion, "Extractores", "Python · jira_archive.py, tawos_sql.py", "Leen los volcados en streaming sin bases de datos intermedias")
  ContainerDb(interim, "Capa intermedia", "Parquet · data/interim/", "Una tabla por colección, sin limpiar")
  Container(construccion, "Constructor del dataset", "Python · 39_build_v3, 52_build_v4", "Armoniza taxonomías, unifica unidades, descarta filas y arma las particiones")
  ContainerDb(procesado, "Dataset de entrenamiento", "Parquet · agix_effort_v4.parquet", "90.975 filas · train 63.926 / val 8.445 / test 4.225 / externo 13.255")
  Container(codificacion, "Codificación con caché", "Python · 53_encode_v4", "Reutiliza vectores ya calculados y solo codifica el texto nuevo")
  ContainerDb(vectores, "Vectores", "npy · embeddings/", "v4_mpnet_congelado.npy + .ids.npy, alineados por task_id")
  Container(entrenamiento, "Entrenamiento y selección", "Python · 54, 71, 77", "Ajusta los brazos y elige en validación, nunca en prueba")
  Container(evaluacion, "Evaluación", "Python · exp_common, 59, 65", "MdMRE, PRED(25), MAE, SA y veredictos pareados con IC95")
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
Rel(vectores, entrenamiento, "Rasgos", "numpy")
Rel(procesado, entrenamiento, "Objetivo y metadatos", "parquet")
Rel(entrenamiento, evaluacion, "Predicciones por bloque", "numpy")
Rel(evaluacion, resultados, "Escribe", "csv/json")

UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

**Garantías de la partición:** `test` es cronológica dentro de cada proyecto, `externo` son 150
proyectos enteros nunca vistos, y los duplicados se agrupan por hash de texto para que no crucen
particiones.

---

## Nivel 3 · Componentes de la API de estimación

```mermaid
C4Component
title Nivel 3 · Dentro de la API de estimación

Container_Boundary(api, "API de estimación") {
  Component(entrada, "Normalizador de entrada", "Python", "Une título y descripción, limpia el texto y deriva task_kind y numéricas")
  Component(enc, "Adaptador del codificador", "sentence-transformers", "Texto → vector 768; lote cuando llegan varias tareas")
  Component(rasgos, "Ensamblador de rasgos", "scikit-learn", "Concatena vector + one-hot(task_kind, source) + log1p(n_links, n_componentes, …)")
  Component(frio, "Predictor en frío", "HistGradientBoosting", "Rasgos → log-horas. Es la ruta por defecto y el respaldo cuando falta la estimación")
  Component(corr, "Corrector C1", "HistGradientBoosting", "Rasgos + log(estimación del usuario) + predicción en frío → log(real/estimación)")
  Component(escala, "Escalado por proyecto", "Python", "Multiplica por la mediana de las últimas tareas cerradas del proyecto")
  Component(banda, "Banda conforme", "Python", "Aplica el cuantil 80 % de los residuos de validación: banda multiplicativa")
  Component(salida, "Compositor de respuesta", "Python", "Horas, rango, ruta usada y número de tareas de historial empleadas")
}

ContainerDb(artefactos, "Almacén de artefactos", "joblib/npy/json", "")
ContainerDb(historial, "Historial de tareas", "PostgreSQL", "")

Rel(entrada, enc, "Texto limpio")
Rel(enc, rasgos, "Vector 768")
Rel(rasgos, frio, "Matriz de rasgos")
Rel(rasgos, corr, "Matriz de rasgos")
Rel(frio, corr, "Predicción en frío como rasgo")
Rel(frio, escala, "Horas en frío")
Rel(corr, escala, "Horas corregidas")
Rel(escala, banda, "Predicción puntual")
Rel(banda, salida, "Predicción y rango")
Rel(escala, historial, "Lee las últimas tareas cerradas", "SQL")
Rel(frio, artefactos, "Carga el modelo", "joblib")
Rel(banda, artefactos, "Carga los cuantiles", "json")

UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

## Vista dinámica · Una petición de estimación

```mermaid
sequenceDiagram
    autonumber
    actor U as Desarrollador
    participant W as Agix (web)
    participant A as API de estimación
    participant E as Codificador SBERT
    participant M as Regresor / corrector
    participant B as Calibrador
    participant D as PostgreSQL

    U->>W: Crea el proyecto
    W->>W: El LLM genera las tareas
    W->>A: POST /estimar (texto, categoría, proyecto, estimación del usuario si la hay)
    A->>E: Codifica el texto
    E-->>A: Vector de 768 (9 ms)
    A->>D: Últimas 10 tareas cerradas del proyecto
    D-->>A: Horas reales
    alt El usuario escribió su estimación
        A->>M: Rasgos + estimación + predicción en frío
        M-->>A: Horas corregidas (ruta C1)
    else No la escribió
        A->>M: Rasgos
        M-->>A: Horas en frío (respaldo)
    end
    A->>B: Predicción puntual
    B-->>A: Rango al 80 % (×2,43 corregido / ×5,07 en frío)
    A-->>W: Horas, rango y ruta usada
    W-->>U: Muestra la estimación con su rango
    U->>W: Cierra la tarea con las horas reales
    W->>D: Guarda horas reales
    D->>M: Reentrenamiento por lotes (solo el regresor)
```

---

## Cifras que respaldan los diagramas

| | |
|---|---|
| Codificador | 9 ms/tarea en GPU · 1,13 GB · determinista |
| Regresor | 2,4 ms/predicción · 1,3 MB · 20 s de reajuste |
| En frío (prueba fija) | MdMRE 0,685 · MAE 5,00 h · SA 0,42 |
| Con estimación corregida | MdMRE 0,494 · MAE 4,30 h · SA 0,50 |
| Banda al 80 % | ×5,07 en frío · ×2,43 corregida |
| Frente a LLMs locales | el LLM gana 1 de 24 comparaciones y tarda 28–218 veces más |
