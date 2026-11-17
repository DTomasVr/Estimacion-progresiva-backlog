"""Pasa DISENO_PIPELINE.md y entrega/ESTADO_Y_METRICAS.md del v6 al v7.

Todas las cifras se leen de results/ (estimador_v7, large_contra_small,
poda_vocabulario, codificadores_int8): ninguna se escribe a mano. Cada
reemplazo comprueba que el texto de origen existe exactamente una vez, para
que un documento que ya cambio no se edite a ciegas.
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
DOC = Path("DISENO_PIPELINE.md")
EST = Path("entrega/ESTADO_Y_METRICAS.md")
R = Path("results")


def j(p):
    return json.load(open(R / p, encoding="utf-8"))


def n(x, d=2):
    return f"{x:,.{d}f}".replace(",", "@").replace(".", ",").replace("@", ".")


def pc(x, d=1):
    return n(100 * x, d) + " %"


def cambia(s, viejo, nuevo):
    assert s.count(viejo) == 1, f"no esta exactamente una vez: {viejo[:80]!r}"
    return s.replace(viejo, nuevo)


def cifras():
    c = j("estimador_v7/construccion.json")
    sv = j("estimador_v7/servicio.json")
    rs = j("estimador_v7/resumen.json")
    fe = j("estimador_v7/factor_estilo.json")
    i8 = j("codificadores_int8/resumen.json")
    po = j("poda_vocabulario/resumen.json")
    bj = j("large_contra_small/base_sola.json")
    pr = pd.read_csv(R / "estimador_v7/prueba.csv")
    bs = pd.read_csv(R / "large_contra_small/base_sola.csv").set_index("codificador")
    mz = pd.read_csv(R / "large_contra_small/metricas.csv")
    m = lambda k, cn: mz[(mz.codificador == k) & (mz.canal == cn)].iloc[0]
    fila = lambda mod, sal: pr[(pr.modelo == mod) & (pr.salida == sal)].iloc[0]
    return dict(c=c, sv=sv, rs=rs, fe=fe, i8=i8, po=po, bj=bj, pr=pr, bs=bs, m=m, fila=fila)


def diseno(s, K):
    c, sv, rs, fe, i8, po, bj, bs, m, fila = (K[k] for k in (
        "c", "sv", "rs", "fe", "i8", "po", "bj", "bs", "m", "fila"))
    mb = n(rs["mb_codificador"], 0)
    k80 = n(c["factor_banda80"][1])
    acu = pc(sv["acuerdo_vecinos"])
    w = c["pesos"]
    gl, gs = m("large", "base"), m("small", "base")
    ml, ms_ = m("large", "mezcla"), m("small", "mezcla")
    cl, cs = m("large", "C"), m("small", "C")
    v6h, v7h, v8h = (fila(x, "horas") for x in (
        "v6 (3 codificadores)", "v7 fp32 (1 codificador)", "v7 int8 podado"))
    v6p, v7p, v8p = (fila(x, "valor_probable") for x in (
        "v6 (3 codificadores)", "v7 fp32 (1 codificador)", "v7 int8 podado"))
    ratio = rs["generador_mediana_v7int8_sobre_v6"]

    s = cambia(s, "| e5-large congelado + 2 ajustados + dos apilados | `estimador_pbe_v6`, 29/09/2026 |",
               f"| e5-large congelado (ONNX int8, {mb} MB) + dos apilados | `estimador_pbe_v7`, 21/10/2026 |")
    s = cambia(s, "[B.11 operación](#b11--operación) · [B.12 decisiones](#b12--resumen-de-decisiones-de-diseño)",
               "[B.11 descartes](#b11--lo-que-se-midió-y-se-descartó) · "
               "[B.12 trampas](#b12--dos-trampas-de-medición-que-conviene-no-repetir) · "
               "[B.13 operación](#b13--operación) · [B.14 decisiones](#b14--resumen-de-decisiones-de-diseño)")
    s = cambia(s, "> `estimador_pbe_v6`, corpus v4 sin Apache, construido el 29/09/2026. Los experimentos",
               f"> `estimador_pbe_v7`, corpus v4 sin Apache, construido el 21/10/2026 sobre el v6 del\n"
               f"> 29/09/2026: el mismo diseño con **un solo codificador** (ver B.5.1). Los experimentos")
    s = cambia(s, "**el servidor v6 no lo usa**.", "**el servidor no lo usa**.")

    # B.3.2
    s = cambia(s, 'CA["Cabeza A · MEDIANA<br/>apilado de 5 canales"]',
               'CA["Cabeza A · MEDIANA<br/>apilado de 3 canales"]')
    s = cambia(s, 'BND["banda x4,32"]', f'BND["banda x{k80}"]')
    s = cambia(s, 'VP -->|si, el 85,3 %| VPS', f'VP -->|si, el {acu}| VPS')

    # B.3.3
    s = cambia(s, '        T["titulo"] --> E5["e5-large congelado<br/>1024 dim"]',
               '        T["titulo"] --> E5["e5-large congelado<br/>ONNX int8 · 1024 dim"]')
    s = cambia(s, '''        T --> AJ["e5 ajustado A<br/>escalar cuadratico"]
        T --> AC["e5 ajustado C<br/>clasificacion + orden por pares"]
''', "")
    s = cambia(s, '''        A4["HGB sobre e5 ajustado A"] --> AP
        A5["HGB sobre e5 ajustado C"] --> AP
''', "")

    # C4 nivel 1
    s = cambia(s, 'System_Ext(hf, "Hugging Face Hub", "Pesos de intfloat/multilingual-e5-large")',
               'System_Ext(hf, "Hugging Face Hub", "Origen de intfloat/multilingual-e5-large; en servicio va dentro del paquete")')
    s = cambia(s, 'Rel(hf, estimacion, "Provee el codificador base congelado", "descarga única")',
               'Rel(hf, estimacion, "Provee el codificador base congelado", "descarga única, al construir")')

    # C4 nivel 2
    s = cambia(s, "title Nivel 2 · Contenedores en tiempo de ejecución (v6)",
               "title Nivel 2 · Contenedores en tiempo de ejecución (v7)")
    s = cambia(s, '"Dos entradas: estimar_tareas y estimar_proyecto; carga perezosa de codificadores")',
               '"Dos entradas: estimar_tareas y estimar_proyecto; carga perezosa del codificador")')
    s = cambia(s, 'Container(encbase, "Codificador base", "PyTorch · multilingual-e5-large CONGELADO", "Texto a vector de 1024. Lo usan las DOS rutas")',
               f'Container(encbase, "Codificador", "ONNX Runtime · multilingual-e5-large CONGELADO, int8, vocabulario podado", "Texto a vector de 1024. {mb} MB. Lo usan las DOS rutas")')
    s = cambia(s, 'Container(encaj, "Codificadores ajustados A y C", "PyTorch · e5-large reajustado", "Solo para la ruta por tarea. 2,2 GB cada uno")\n', "")
    s = cambia(s, '"scikit-learn · apilado RidgeCV de 5 canales", "Horas de UNA tarea, con banda x4,32")',
               f'"scikit-learn · apilado RidgeCV de 3 canales", "Horas de UNA tarea, con banda x{k80}")')
    s = cambia(s, '"estimador_pbe_v6.joblib de 221 MB, mas models/e5_ajustado y _C")',
               f'"estimador_pbe_v7.joblib de {n(c["mb"], 0)} MB, mas models/e5_v7 (ONNX, tokenizador, tabla del vocabulario)")')
    s = cambia(s, 'Rel(api, encaj, "Solo en la ruta por tarea", "en proceso")\n', "")
    s = cambia(s, 'Rel(encbase, cabB, "Vectores", "numpy")',
               'Rel(encbase, cabA, "Vectores", "numpy")\nRel(encbase, cabB, "Vectores", "numpy")')
    s = cambia(s, 'Rel(encaj, cabA, "Vectores ajustados", "numpy")\n', "")
    s = cambia(s, '''> **Decisión que se ve aquí:** el codificador base está **congelado** y los codificadores
> ajustados **solo hacen falta para la ruta por tarea**. Un despliegue que solo necesite el
> total del proyecto —que es la métrica que se presenta— ahorra **4,4 GB**.''',
               f'''> **Decisión que se ve aquí:** hay **un solo codificador**, congelado, y lo comparten las dos
> rutas. El v6 llevaba además dos codificadores ajustados (2,2 GB cada uno) para la ruta por
> tarea; medidos fuera de pliegue no aportaban y el v7 los quitó (B.5.1). El codificador va en
> ONNX int8 con el vocabulario podado: **{mb} MB** frente a los 6,5 GB de los tres del v6.''')

    # C4 nivel 2b
    s = cambia(s, '"e5-large.npy y los dos ajustados, alineados por fila")',
               '"e5-large.npy, alineado por fila (y los ajustados de los experimentos)")')
    s = cambia(s, '"Brazos A y C sobre e5-large, con la tabla de embeddings congelada")',
               '"Brazos A y C sobre e5-large. Solo experimentos: el v7 no los usa")')
    s = cambia(s, '"Python · 163_modelo_v5, 171_estimador_v6", "Ajusta los canales y el apilado; elige SIEMPRE en validación")',
               '"Python · 163_modelo_v5, 171_estimador_v6, 252_estimador_v7", "Ajusta los canales y el apilado; elige SIEMPRE en validación")')

    # B.3.7
    s = cambia(s, '''    V5 --> V6["171_estimador_v6.py<br/>paquete de servicio"]
    V6 --> P{{"prueba de extremo a extremo<br/>estima de verdad, con y sin cerradas"}}''',
               '''    V5 --> V6["171_estimador_v6.py<br/>paquete v6"]
    V6 --> V7["252_estimador_v7.py<br/>un solo codificador"]
    V7 --> OX["246 + 249: ONNX int8<br/>y vocabulario podado"]
    OX --> P{{"prueba de extremo a extremo<br/>estima de verdad, con y sin cerradas"}}''')
    s = cambia(s, "del codificador nunca se habían guardado. Lo destapó justamente esa prueba.",
               "del codificador nunca se habían guardado. Lo destapó justamente esa prueba.\n\n"
               "Con el v7 el codificador ya no se reajusta, así que la parte que más se rompía —un\n"
               "codificador nuevo con una cabeza vieja— desaparece. Si se cambia el codificador base, el\n"
               "ONNX y la tabla del vocabulario se regeneran con `246` y `249`, y se mide con `253`.")

    # B.5.1
    s = cambia(s, "| **semántica** | `intfloat/multilingual-e5-large`, **congelado**, prefijo `query: ` | 1024 |",
               "| **semántica** | `intfloat/multilingual-e5-large`, **congelado**, prefijo `query: `, en ONNX int8 | 1024 |")
    a = s.index("Además, **dos codificadores ajustados** sobre el mismo `e5-large`")
    b = s.index("### B.5.2 · Cabeza A")
    s = s[:a] + f'''**Un solo codificador.** El v6 llevaba además dos codificadores ajustados sobre el mismo
`e5-large` —A, con pérdida cuadrática, y C, con clasificación sobre la rejilla más orden por
pares— y pesaban en el apilado +0,251 y +0,233. Pero el apilado se ajusta en
`split_t3 == "val"`, que tiene tareas de los **mismos proyectos** que vieron esos codificadores
al ajustarse: premiaba justo lo que no generaliza a un proyecto nuevo (la fuga de E54).

Medido fuera de pliegue, con `GroupKFold` por proyecto entero y los codificadores ajustados
**por pliegue** (E120, `248`–`251`):

| | ±25 % | SA | Spearman |
|---|---|---|---|
| e5-large base | {pc(gl.PRED25)} | {n(gl.SA, 4)} | {n(gl.spearman, 3)} |
| base + ajustado C | {pc(ml.PRED25)} | {n(ml.SA, 4)} | {n(ml.spearman, 3)} |
| solo el ajustado C | {pc(cl.PRED25)} | {n(cl.SA, 4)} | {n(cl.spearman, 3)} |

Añadir el canal ajustado **empeora**. El v7 los quita y se queda con el base.

**En ONNX int8 y con el vocabulario podado.** De los 250.002 tokens de XLM-R se conservan
{n(rs["vocab_produccion"], 0)}: los que aparecen en el corpus más todo token de alfabeto latino.
Solo con los del corpus el {pc(po["titulos_con_oov_generador"], 0)} de las tareas del generador
tenía algún token podado; con los latinos, ninguna. Delante del modelo va una tabla
id_original → id_podado y un token ausente cae en `<unk>`. El codificador pasa de 2,2 GB a
**{mb} MB** y corre en CPU sin torch, con las métricas del bloque de prueba intactas (B.9).

**Y no el e5-small.** Con la misma receta fuera de pliegue el small empata en ±25 %
({n(100 * bj["PRED25"]["small_menos_large"], 2)} puntos, IC95 [{n(100 * bj["PRED25"]["IC95"][0], 2)},
{n(100 * bj["PRED25"]["IC95"][1], 2)}]) pero ordena peor: Spearman {n(bj["spearman"]["small_menos_large"], 3)},
IC95 [{n(bj["spearman"]["IC95"][0], 3)}, {n(bj["spearman"]["IC95"][1], 3)}], y SA
{n(bj["SA"]["small_menos_large"], 4)}. Con el large ya en {mb} MB no compensa.

''' + s[b:]

    # B.5.2
    a = s.index("Cinco canales apilados con `RidgeCV` (α = 48,3)")
    b = s.index("La salida se exponencia y se multiplica por un **factor de estilo de 0,831**.")
    s = s[:a] + f'''Tres canales apilados con `RidgeCV` (α = {n(c["alfa"])}) ajustado **en validación**, no en
entrenamiento, para que los pesos no premien al canal que más sobreajusta:

| canal | peso |
|---|---|
| bolsa de 7 HGB sobre e5 + SVD (85 % de muestras, 80 % de rasgos cada uno) | **+{n(w["bolsa"], 3)}** |
| Ridge sobre TF-IDF disperso | +{n(w["ridge"], 3)} |
| LinearSVR sobre TF-IDF disperso | {n(w["svr"], 3)} |

El peso negativo del SVR no es un error: el apilado lo usa como **corrección** del Ridge, que ve
los mismos rasgos. Los modelos de canal son los del v6; solo se reajustó el apilado. La banda
por tarea al 80 % es ×{k80}: el cuantil **conforme** del residuo de validación, con el apilado
ajustado por pliegues dentro de validación para no medirla sobre lo que ajustó.

''' + s[b:]
    s = cambia(s, "en un bloque y verificado en el otro.\n\n### B.5.3",
               f'''en un bloque y verificado en el otro.

> **Abierto en el v7.** El factor es el heredado del v6. Sobre las 199 tareas de la prueba de
> extremo a extremo, la cabeza A del v7 da por tarea **×{n(ratio)}** lo que daba la del v6; el
> total no cambia (lo calcula la cabeza B). Re-medirlo con el corpus de 800 títulos reescritos
> no sirve: lo reescribió un LLM local y no el generador, y el procedimiento no reproduce ni el
> valor del v6 ({n(fe["v6"]["factor"][0], 3)} frente a 0,831). Cerrarlo pide tareas del generador
> actual con horas reales.

### B.5.3''')

    # B.5.4
    s = cambia(s, "El umbral de acuerdo entre vecinos (**1,524**, que cubre el 85,3 % de las tareas) sigue",
               f"El umbral de acuerdo entre vecinos (**{n(sv['umbral_dispersion'], 3)}**, que cubre el {acu} de las tareas) sigue")

    # B.7
    s = cambia(s, "est = EstimadorV6()                      # carga models/estimador_pbe_v6.joblib",
               "est = EstimadorV6(paquete=\"models/estimador_pbe_v7.joblib\")   # la clase sirve v6 y v7")
    s = cambia(s, "- `historial` solo debe traer tareas **del mismo proyecto** (ver B.6.4).",
               "- `historial` solo debe traer tareas **del mismo proyecto** (ver B.6.4).\n"
               "- Las rutas del ONNX, del tokenizador y de la tabla van **relativas al paquete**: la clase las\n"
               "  resuelve desde la carpeta que contiene `models/`. No hace falta `HF_HOME` ni red.")

    # B.8
    a = s.index("```mermaid\nflowchart TB\n    subgraph SRV[\"Proceso de servicio\"]")
    b = s.index("### Por qué un modelo propio y no un LLM en línea")
    s = s[:a] + f'''```mermaid
flowchart TB
    subgraph SRV["Proceso de servicio"]
        API["EstimadorV6<br/>proceso Python"]
        PKG[("estimador_pbe_v7.joblib<br/>{n(c["mb"], 0)} MB")]
        B[("models/e5_v7<br/>ONNX int8 · {mb} MB<br/>+ tokenizador + tabla")]
        API --> PKG
        API --> B
    end
```

Un solo codificador para las dos rutas, dentro del paquete: no se descarga nada ni hace falta el
caché de Hugging Face. Medido en CPU, sin GPU y sin caché (`255_medidas_servicio_v7.py`):

| | |
|---|---|
| carga del paquete | **{n(sv["carga_s"], 1)} s** |
| primera llamada a `estimar_tareas`, 42 tareas (abre el ONNX) | {n(sv["primera_llamada_42_s"], 1)} s |
| `estimar_tareas`, lote de 42, ya cargado | **{n(sv["caliente_42_ms"], 0)} ms · {n(sv["ms_por_tarea_caliente"], 1)} ms/tarea** |
| `estimar_proyecto`, 42 tareas, ya cargado | **{n(sv["estimar_proyecto_42_ms"], 0)} ms** |
| memoria residente del estimador solo | {n(sv["ram_pico_gb"], 1)} GB de RAM |
| generador + estimador, en CPU (paquete de producción) | {n(sv["ram_pico_pipeline_completa_cpu_gb"], 1)} GB de RAM, pico |

Con el v6 eran tres codificadores de 2,2 GB y, en CPU, unos 10 GB de RAM para las dos tuberías.
La carga sigue siendo **perezosa**: en servicio conviene precalentar al arrancar.

''' + s[b:]
    s = cambia(s, "las incidencias: ninguno acertó la clave exacta en 450 intentos.",
               "las incidencias: ninguno acertó la clave exacta en 450 intentos. (Los 11 ms son del v6 en GPU;\n"
               f"el v7 en CPU va a {n(sv['ms_por_tarea_caliente'], 1)} ms por tarea.)")

    # B.9
    s = cambia(s, "constante saca −0,015 y 0,000 y el modelo no.\n\n### Lo que gana la suma",
               f'''constante saca −0,015 y 0,000 y el modelo no.

### v7 contra v6, sobre el mismo bloque

Lo que sale del artefacto (no fuera de pliegue), sobre las mismas 4.225 tareas:

| | ±25 % horas | ±25 % valor probable | SA | Spearman | cobertura de la banda |
|---|---|---|---|---|---|
| v6 (3 codificadores) | {pc(v6h.PRED25)} | {pc(v6p.PRED25)} | {n(v6h.SA, 4)} | {n(v6h.spearman, 3)} | {pc(v6p.cobertura_banda)} |
| v7 fp32 | {pc(v7h.PRED25)} | {pc(v7p.PRED25)} | {n(v7h.SA, 4)} | {n(v7h.spearman, 3)} | {pc(v7p.cobertura_banda)} |
| **v7 int8 podado** (desplegado) | **{pc(v8h.PRED25)}** | {pc(v8p.PRED25)} | {n(v8h.SA, 4)} | {n(v8h.spearman, 3)} | {pc(v8p.cobertura_banda)} |

En este bloque el v6 **ordena algo mejor**, y es lo esperable: sus codificadores ajustados
vieron tareas de los mismos proyectos que el bloque. Para un proyecto nuevo la medida es la de
B.5.1, fuera de pliegue, donde quitarlos **mejora** el SA y el Spearman. El total del proyecto
no cambia entre v6 y v7: lo calcula la cabeza B, que nunca usó los codificadores ajustados.

### Lo que gana la suma''')

    # B.11: una fila mas en descartes
    s = cambia(s, "| descomponer la tarea en pasos (WBS) | ρ(nº de pasos, horas) = +0,012 |",
               "| descomponer la tarea en pasos (WBS) | ρ(nº de pasos, horas) = +0,012 |\n"
               f"| codificadores ajustados por pliegue (A y C) | fuera de pliegue empeoran: SA {n(gl.SA, 4)} → {n(ml.SA, 4)} |\n"
               f"| e5-small en lugar de e5-large | empata en ±25 %, ordena peor: Spearman {n(gs.spearman, 3)} contra {n(gl.spearman, 3)} |")

    # B.13 y B.14
    s = cambia(s, '''**codificador base no se toca**; se reajustan las cabezas y, si se rehacen los codificadores
ajustados, hay que repetir la cadena completa en el orden de [B.3.7](#b37--orden-de-reconstrucción).''',
               '''**codificador no se toca**; se reajustan las cabezas en el orden de
[B.3.7](#b37--orden-de-reconstrucción) y se repite la prueba de extremo a extremo.''')
    s = cambia(s, "| **dos** codificadores ajustados | solo uno | uno ordena mejor y el otro estima mejor la escala |",
               f"| **un solo** codificador, el base | tres (base + dos ajustados, v6) | fuera de pliegue los ajustados empeoran el SA y el orden; 6,5 GB → {mb} MB |\n"
               f"| ONNX int8, vocabulario podado | fp32 completo | mismas métricas, 2,2 GB → {mb} MB, sin torch ni caché de HF |")
    return s


def estado(s, K):
    c, sv, rs, bj, m, fila = (K[k] for k in ("c", "sv", "rs", "bj", "m", "fila"))
    mb = n(rs["mb_codificador"], 0)
    gl, ml = m("large", "base"), m("large", "mezcla")
    v6h, v8h = fila("v6 (3 codificadores)", "horas"), fila("v7 int8 podado", "horas")
    v8p = fila("v7 int8 podado", "valor_probable")
    s = cambia(s, "Fecha: 15 de octubre de 2026.", "Fecha: 21 de octubre de 2026 (estimador v7).")
    s = cambia(s, "`models/estimador_pbe_v6.joblib`, con respaldos por cada cambio.",
               f"`models/estimador_pbe_v7.joblib` + `models/e5_v7/` (el codificador, {mb} MB). El v6\n"
               "sigue en `models/` y la clase sirve los dos.\n\n"
               f"- **Un solo codificador**, el e5-large base, en ONNX int8 con el vocabulario podado: {mb} MB\n"
               "  en lugar de los tres de 2,2 GB del v6. Detalle en la sección 6b.")
    s = cambia(s, "\n---\n\n## 7. Recomendación", f'''
---

## 6b. El v7: de tres codificadores a uno

**Por qué.** Medido fuera de pliegue, con proyectos que el modelo no vio, los dos codificadores
ajustados del v6 no aportaban: añadirlos al base baja el SA de {n(gl.SA, 4)} a {n(ml.SA, 4)} y el
Spearman de {n(gl.spearman, 3)} a {n(ml.spearman, 3)}. Pesaban en el apilado porque este se
ajustaba con tareas de los mismos proyectos que habían visto.

**Qué cuesta en el bloque de prueba** (4.225 tareas, lo que sale del artefacto):

| | ±25 % horas | ±25 % valor probable | SA | Spearman |
|---|---|---|---|---|
| v6 | {pc(v6h.PRED25)} | {pc(fila("v6 (3 codificadores)", "valor_probable").PRED25)} | {n(v6h.SA, 4)} | {n(v6h.spearman, 3)} |
| **v7 desplegado** | **{pc(v8h.PRED25)}** | {pc(v8p.PRED25)} | {n(v8h.SA, 4)} | {n(v8h.spearman, 3)} |

Ahí el v6 ordena algo mejor porque juega con ventaja (vio esos proyectos); para un proyecto
nuevo el v7 es mejor. **El total del proyecto no cambia**: lo calcula la otra cabeza.

**Qué gana:** el paquete de producción pasa de 6,0 a 3,1 GB; las dos tuberías en CPU, de unos
10 GB de RAM a {n(sv["ram_pico_pipeline_completa_cpu_gb"], 1)} GB; el estimador va a
{n(sv["ms_por_tarea_caliente"], 1)} ms por tarea en CPU y no necesita red ni caché de HF.

**Abierto:** sobre texto del generador las horas **por tarea** salen ×{n(rs["generador_mediana_v7int8_sobre_v6"])}
las del v6. No se puede saber cuál acierta sin tareas generadas con horas reales.

**Y el e5-small:** empata en ±25 % pero ordena peor (Spearman
{n(bj["spearman"]["small_menos_large"], 3)}, IC95 [{n(bj["spearman"]["IC95"][0], 3)},
{n(bj["spearman"]["IC95"][1], 3)}]). Con el large ya en {mb} MB no compensa.

---

## 7. Recomendación''')
    return s


def main():
    K = cifras()
    for f, fn in ((DOC, diseno), (EST, estado)):
        s = io.open(f, encoding="utf-8").read()
        s2 = fn(s, K)
        io.open(f, "w", encoding="utf-8").write(s2)
        print(f"{f}: {len(s):,} -> {len(s2):,} caracteres")


if __name__ == "__main__":
    main()
