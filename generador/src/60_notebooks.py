"""Genera los notebooks de la carpeta de asesoria."""
import json
import os

DEST = r"F:\PBE-asesoria\notebooks"
os.makedirs(DEST, exist_ok=True)
BASE = r"F:\\Software Task generation Model"
PY = r"C:\\Users\\User\\Documents\\Programacao\\PBE-Dataset\\.venv\\Scripts\\python.exe"

CABECERA = f'''import os, subprocess, sys, json
import pandas as pd
BASE = r"{BASE}"
PY   = r"{PY}"
os.chdir(BASE)

def correr(script, *args, mostrar=25):
    """Ejecuta un script del proyecto y muestra el final de su salida."""
    r = subprocess.run([PY, "-u", f"src/{{script}}", *map(str, args)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    lineas = [l for l in (r.stdout or "").split("\\n") if l.strip()
              and "warn" not in l.lower() and "Loading weights" not in l]
    print("\\n".join(lineas[-mostrar:]))
    if r.returncode: print("ERROR:", (r.stderr or "")[-800:])
    return r.returncode == 0

def tabla(ruta, **kw):
    d = pd.read_csv(f"{{BASE}}/{{ruta}}", **kw)
    display(d)
    return d
print("listo:", BASE)'''


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.split("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.split("\n")}


def guardar(nombre, celdas):
    # jupyter espera que cada linea de "source" acabe en salto de linea menos la
    # ultima; sin esto las celdas se concatenan en una sola linea y no compilan
    salto = chr(10)
    for c in celdas:
        src = c["source"]
        c["source"] = [l if l.endswith(salto) else l + salto
                       for l in src[:-1]] + src[-1:]
    nb = {"cells": celdas, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5}
    with open(f"{DEST}/{nombre}", "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    print("escrito", nombre)


# ============================================================ 1. datos
guardar("1_datos.ipynb", [
    md("""# 1 · Los datos

Del volcado de Jira al conjunto de entrenamiento, y la destilación de dominios de producto.

**Nada aquí necesita ejecutarse para leerse**: cada paso trae el resultado que dio.
Ejecutarlo entero son unas 2 horas, casi todas de generación con el LLM profesor."""),
    code(CABECERA),
    md("""## 1.1 Reconstruir la jerarquía

Dos volcados públicos: **TAWOS** (Apache 2.0) y un volcado de **Jira público** (CC BY 4.0).

El trabajo invisible fue reconstruir la relación padre → hijo:

* el volcado de Jira extraído previamente estaba filtrado a incidencias con horas (2,9 %); se
  releyó completo en streaming → **2 686 282 incidencias en 8,1 min**
* TAWOS no tiene columna de padre: se reconstruyó del historial de cambios
  (92 706 cambios de «Epic Link», 73 054 de «Epic Child») → **2,4 min**
* 26 de los 39 proyectos de TAWOS se solapaban con Jira y se resolvieron a favor de Jira

**Resultado:** 2 845 066 incidencias · 244 677 relaciones · 1 289 proyectos (solo 39 con descripción)."""),
    code('''# correr("01_extract_jira_full.py")      # 8,1 min
# correr("02_extract_tawos_changelog.py") # 2,4 min
# correr("03_build_hierarchy.py")         # 50 s
tabla("results/03_hierarchy_profile.csv").head(8)'''),
    md("""## 1.2 Partición por proyecto entero

Partir por incidencia dejaría tareas del mismo proyecto a los dos lados. Una sola partición la
comparten el SBERT, los dos generadores y toda la evaluación."""),
    code('correr("04_split_projects.py")\ntabla("results/04_split_resumen.csv")'),
    md("""## 1.3 De incidencias a ejemplos

Filtros, en orden: fuera el trabajo reactivo (bugs, soporte); solo relaciones padre→subtarea y
epic→hijos; padre e hijo del mismo proyecto; al menos 2 hijos; sin tareas repetidas dentro del
mismo padre; sin padres casi idénticos dentro de un proyecto (coseno ≥ 0,97).

**Quedan 11 202 ejemplos de historia → tareas y 6 068 de epic → historias.**"""),
    code('correr("08_preprocess.py")\ntabla("results/08_preprocess_resumen.csv")'),
    md("""## 1.4 La destilación de dominio · **lo nuevo de esta versión**

El corpus es software técnico, pero PBE recibe descripciones de producto. Para «una tienda de
audiolibros» el modelo respondía con vocabulario de Hyperledger.

Se comprobó primero que el dato no existiera publicado:

* la revisión sistemática de Motger et al. (2025) encuentra que las historias de usuario son el
  **4,8 %** de los conjuntos de requisitos catalogados, y ninguno trae tareas
* se rastrearon **45 repositorios** de producto en GitHub: de 1 402 incidencias con listas de
  tareas, sobreviven al filtro **19 (1,4 %)** y **ninguna** es descomposición de producto
* los **sub-issues** nativos los usa **1 de 478** incidencias abiertas (0,2 %)

Conclusión: la destilación desde un LLM no es un atajo, es la única vía disponible."""),
    code('''# 30 sectores de consumo -> 528 descripciones
correr("41_dominios_sinteticos.py")'''),
    code('''# el profesor escribe un backlog por descripcion (qwen2.5:7b, Apache 2.0). ~76 min
# correr("42_generar_sintetico.py", "qwen2.5:7b")
import json
B = [json.loads(l) for l in open(f"{BASE}/data/synth/backlogs_qwen2.5_7b.jsonl", encoding="utf-8")]
print(f"{len(B)} backlogs, {sum(len(b['historias']) for b in B)} historias")
print("\\nejemplo:", B[0]["descripcion"])
for h in B[0]["historias"][:3]:
    print(" *", h["historia"][:100])
    for t in h["tareas"][:2]: print("     -", t[:90])'''),
    code('''# mezcla real + sintetico, con deduplicacion semantica y particiones disjuntas
correr("43_datos_mixtos.py")'''),
    md("""### Por qué el scraping no sirvió

El filtro está en `51_filtrar_github.py`. Merece leerse porque es un resultado publicable."""),
    code('correr("51_filtrar_github.py")\ntabla("results/51_github_filtrado.csv")'),
])

# ====================================================== 2. entrenamiento
guardar("2_entrenamiento.ipynb", [
    md("""# 2 · El entrenamiento

Dos modelos mT5-small con el vocabulario podado, más la etapa de destilación.

Ejecutarlo entero: unas **3 horas** de GPU."""),
    code(CABECERA),
    md("""## 2.1 Poda de vocabulario · la decisión que más rindió

mT5 trae 250 100 piezas para 101 idiomas. Se conservan las **75 260** que aparecen en nuestro
corpus (inglés y español).

| | antes | después |
|---|---|---|
| mT5-small | 300M | **121M** |
| mT5-base | 582M | 314M |
| ejemplos/s entrenando | 19,5 | **67,6** |

**La poda es exacta:** la diferencia de logits con el modelo original es 0,0, y en español
reservado hay 0 % de palabras desconocidas. Sin ella no habría despliegue en una laptop."""),
    code('correr("10_prune_vocab.py")'),
    md("""## 2.2 Las dos mitades

**Abajo** (historia → tareas): 22 871 ejemplos, lote 16, tasa 1,4·10⁻³, parada temprana.
**Arriba** (descripción → historia): One2One, 71 817 pares, más una segunda etapa corta que fija
el formato «Como…, quiero…, para…» (100 % de las historias salen en formato).

Adafactor, pesos fp32 con cálculo en bf16, lotes agrupados por longitud, y dos puertas antes de
lanzar: `--probe` mide la VRAM real y `--mini` exige que memorice 200 ejemplos."""),
    code('''# DATOS=hist  -> abajo   |  DATOS=o2o + o2ofmt -> arriba
# os.environ["DATOS"]="hist"; correr("09_train_mt5.py", "models/mt5-small-pruned", 16, 192, "1.4e-3", 12)
print("modelos entrenados disponibles:")
for d in sorted(os.listdir(f"{BASE}/models")):
    if d.startswith("mt5"): print("  ", d)'''),
    md("""## 2.3 La destilación · **lo nuevo**

Se continúa desde los modelos de producción sobre la mezcla real + sintético. **Una sola época
basta**: en las dos mitades la mejor fue la primera y la segunda ya empeora.

| mitad | val inicial | mejor |
|---|---|---|
| abajo | 3,3623 | 3,3120 |
| arriba | 3,5892 | **2,4127** |

Arriba es donde vive el efecto: es la mitad que tiene que saber *qué funcionalidades lleva un
producto*."""),
    code('''# os.environ["DATOS"]="mix";  correr("09_train_mt5.py", "models/mt5-small-pruned_hist_pbe_v1", 16, 192, "5e-4", 4)
# os.environ["DATOS"]="mixa"; correr("09_train_mt5.py", "models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1", 32, 48, "5e-4", 4)
print("los modelos en produccion son:")
print("  arriba: mt5-small-pruned_o2ofmt_30sec")
print("  abajo : mt5-small-pruned_hist_30sec (+ ..._bs8_lr9.2e-4 como segundo del conjunto)")'''),
    md("""## 2.4 Un resultado negativo que conviene contar

Se probó una segunda versión con **211 sectores** en vez de 30 (18 564 historias sintéticas).
Sobre las mismas historias humanas dio **0,6449 contra 0,6504**: empate. Y al leer sus salidas
aparecieron pérdida de fluidez y degradación del dominio técnico.

**Cubrir un dominio ayuda a ese dominio; añadir dominios ajenos de baja calidad perjudica.**"""),
])

# ======================================================== 3. evaluacion
guardar("3_evaluacion.ipynb", [
    md("""# 3 · La evaluación

Tres poblaciones distintas, porque cada una responde una pregunta distinta."""),
    code(CABECERA),
    md("""## 3.1 La métrica

No existe una única descomposición correcta: dos equipos parten la misma historia distinto y los
dos aciertan. Por eso la métrica es de **similitud semántica** (F1 suave con el SBERT propio) y no
de coincidencia exacta, y **todo va con intervalo de confianza** por bootstrap pareado al 95 %.

Si un intervalo cruza el cero, no se afirma nada."""),
    md("""## 3.2 Contra los LLM · el banco de 64 proyectos

Mismos 64 proyectos reservados, misma máquina, cuatro sistemas. Las salidas de los LLM están
guardadas, así que recalcular nuestro brazo cuesta minutos."""),
    code('''# correr("20_banco_llms.py", 60)     # regenera nuestro brazo
correr("22_resumen_banco.py", mostrar=30)'''),
    code('tabla("results/22_banco_pareado.csv")'),
    md("""**Los seis empatan.** Antes de la destilación, dos de ellos ganaban con intervalos que
excluían el cero. Y seguimos siendo de **5 a 10 veces más rápidos**."""),
    md("""## 3.3 La validación que no se puede discutir

Entrenar con backlogs de un LLM y evaluar contra referencias del mismo LLM mediría mimetismo. Por
eso la medición decisiva usa **328 historias escritas por personas**, publicadas por otro grupo
(Bragilovski et al., CC BY 4.0).

De sus 9 proyectos se comprobó el solapamiento con nuestro Dalpiaz y **se descartaron 3** por
repetirse. Quedan 6 proyectos de consumo."""),
    code('correr("48_test_real.py")'),
    code('''# PBE_DEVICE=cpu python src/49_eval_real.py ambos   -> compara produccion contra destilado
tabla("results/49_real.csv")'''),
    md("""| | F1 suave | precisión | cobertura |
|---|---|---|---|
| antes | 0,5522 | 0,6052 | 0,5089 |
| **después** | **0,6504** | **0,7223** | **0,5922** |

**+0,098 con IC95 [+0,053; +0,139], ganando en 6 de 6 proyectos.**"""),
    md("""## 3.4 Leer las salidas, no solo las métricas

Esto es lo que más ha rendido en todo el proyecto. La métrica decía que todo iba bien y la lectura
reveló que el modelo caía al sector vecino, que colapsaba en un solo tema, y que la versión de 211
sectores perdía fluidez.

El F1 suave con SBERT es **ciego a la agramaticalidad**: «so that I can plan my jewelry» puntúa
casi igual que una frase correcta."""),
    code('correr("45_leer_salidas.py", "nuevo", mostrar=40)'),
    md("""## 3.5 Otras comprobaciones

* **Dominios nuevos:** se dejó fuera un dominio entero y se reentrenó, tres veces. Sin caída.
* **Contaminación:** se preguntó a cada LLM si reconocía el proyecto: del 55 % al 81 % dijeron que
  sí. El nuestro, 0 %, por construcción.
* **Traducción de la salida:** fidelidad 0,90 contra un piso de 0,20, con retención de términos
  técnicos del 100 % gracias al enmascarado."""),
    code('tabla("results/29_lodo_v2.csv")\ntabla("results/27_traduccion_salida.csv")'),
])

# ========================================================== 4. pipeline
guardar("4_pipeline.ipynb", [
    md("""# 4 · El pipeline, en vivo

Este sí conviene ejecutarlo: carga los modelos y genera un backlog delante de quien mire.

Cuatro etapas. El 95 % del tiempo está en los dos generadores."""),
    code(CABECERA),
    code('''import importlib.util
os.environ["PBE_DEVICE"] = "cuda"          # "cpu" para medir el despliegue sin grafica
spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
P = pl.Pipeline(traducir=True)
print(f"cargado en {P.carga_s}s sobre {pl.DISPOSITIVO}")
print("arriba:", pl.M_ARRIBA.split("/")[-1])
print("abajo :", pl.M_ABAJO.split("/")[-1], "+", pl.M_ABAJO2.split("/")[-1])'''),
    code('''desc = ("Una plataforma de comercio electrónico para vender audiolibros y ebooks, "
        "con muestras de los primeros capítulos, biblioteca personal y pagos en la app.")
r = P(desc, k_historias=20, n_muestras=60)
print(f"{len(r['historias'])} historias, {r['n_tareas']} tareas, {r['tiempos']['total']:.1f}s\\n")
for h in r["historias"][:6]:
    print(f"[{h['votos']} votos · relevancia {h['relevancia']:.2f}] {h.get('historia_es') or h['historia']}")
    for t in (h.get("tareas_es") or h["tareas"]):
        print("    -", t)
    print()'''),
    md("""### Qué mirar en la salida

* **votos**: de 30 muestreos, cuántos propusieron esta misma historia. Si *todas* tienen 1 voto,
  el modelo está improvisando: es su señal de «no sé».
* **relevancia**: coseno con la descripción. Por encima de 0,6 suele ser contenido del dominio.
* **origen**: de qué proyectos reales se apoyó la recuperación. Es procedencia, algo que un LLM
  no da."""),
    code('''import time
for disp in ["cuda", "cpu"]:
    pass  # cambiar PBE_DEVICE y reinstanciar para comparar; medido: 4,18 s en GPU, 11,4 s en CPU
print("GPU  4,18 s por backlog   (52 tareas)")
print("CPU  11,4 s con 4 hilos y 3,2 GB de RAM")
print("llama3.1:8b en la MISMA CPU: 181,9 s  ->  16 veces mas lento")'''),
    md("""## La página

La demo web es el mismo pipeline detrás de un servidor de la biblioteca estándar, sin
dependencias nuevas:

```
python src/33_servidor.py --gpu
```

y abrir **http://127.0.0.1:8123**. El selector «historias» controla cuántas se piden (12 a 40).
Los ejemplos precalculados aparecen al instante porque no generan nada."""),
])
print("\\ntodos los notebooks escritos en", DEST)
