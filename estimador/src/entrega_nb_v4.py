"""Cells for the v4 notebook: the Public Jira corpus and the merge into v3.

The merge rules are the part of this work most likely to be challenged in a
defence, so each one is verified in the notebook against the delivered file
rather than described: the v3 test block is compared identifier by identifier,
the leakage guards are recomputed as set intersections, and the external block
is checked to share no project with training.
"""
from entrega_comun import ESTILO, code, construir, md


def celdas():
    c = []
    c.append(md("""
# Dataset v4 — el corpus público de Jira y su mezcla con v3

**PBE · componente de estimación de esfuerzo · anexo de datos**

v4 añade a v3 el conjunto **The Public Jira Dataset** (Montgomery, Lüders y
Maalej, MSR 2022): 2 686 282 incidencias de dieciséis Jiras públicos, de las que
78 634 registran tiempo real de trabajo.

Este cuaderno documenta la adquisición, el perfilado, los **seis criterios de
descarte (D1–D6)**, la armonización de esquema y —lo más importante para poder
defender los resultados— los **tres criterios de mezcla**, cada uno verificado
contra el fichero entregado.

## Índice

| § | Contenido |
|---|---|
| 1 | Por qué hacía falta un segundo corpus |
| 2 | Adquisición: leer un volcado de MongoDB sin instalar MongoDB |
| 3 | Cobertura: cuánto tiempo real registra cada Jira |
| 4 | Perfilado del bloque crudo |
| 5 | Criterios de descarte D1–D6 |
| 6 | Armonización de esquema |
| 7 | **Criterios de mezcla** y su verificación |
| 8 | Perfil final de v4 y diccionario |

## Linaje

```
The Public Jira Dataset (mongodump 5,87 GB)
   └─ 2 686 282 incidencias  ─►  78 634 con tiempo  ─►  72 501 tras D1–D6
                                                             │
v3 (18 474 filas) ───────────────────────────────────────────┴─►  v4 · 90 975 filas
```
"""))

    c.append(code(ESTILO + '''
print("entorno listo")
print("carpeta de datos:", DATOS.resolve())
'''))

    # ---------------------------------------------------------------- 1
    c.append(md("""
---
## 1. Por qué hacía falta un segundo corpus

v3 son 17 350 tareas de **dos** organizaciones. Todos los experimentos se
estancaban en un error mediano de 0,68 y quedaban dos explicaciones abiertas que
v3 no podía distinguir:

1. **Falta de datos.** 10 597 filas de entrenamiento pueden ser pocas para que
   el texto revele su señal.
2. **Falta de información.** El texto de una tarea sencillamente no contiene lo
   que hace falta para fijar sus horas.

Distinguirlas exige un corpus mucho mayor y de otra procedencia. Además, este
conjunto conserva `timeoriginalestimate` junto a `timespent`, cosa que TAWOS no
tiene: permite comprobar si la calidad de la estimación humana medida en SiP
—una sola empresa, 18 desarrolladores— se sostiene en otras organizaciones.
"""))

    # ---------------------------------------------------------------- 2
    c.append(md("""
---
## 2. Adquisición: un volcado de MongoDB sin instalar MongoDB

El conjunto se distribuye como `mongodump-JiraReposAnon.archive`, 5,87 GB, que
según sus autores ocupa unos **60 GB** una vez restaurado. No se restauró.

Dos observaciones lo hacen innecesario:

**Primera.** `mongodump --gzip --archive` comprime el archivo entero como **un
único flujo gzip** (se confirma leyendo los primeros bytes: `1f 8b`). Debajo,
el formato es:

```
magic 0x8199e26d
prólogo:  cabecera + un documento de metadatos por colección + terminador
bloques:  cabecera de espacio de nombres + documentos BSON + terminador  (repetido)
```
donde un terminador son los cuatro bytes `ff ff ff ff` colocados donde iría un
prefijo de longitud BSON.

**Segunda.** Cada incidencia trae unos 300 campos personalizados casi todos
nulos, más el `changelog` (32 M de cambios en total) y los comentarios (9 M).
Decodificar todo eso costaría muchísimo más que los pocos campos que se usan.
El lector recorre el BSON **a nivel de byte** y salta lo que no se pidió usando
el prefijo de longitud de cada elemento.

Resultado: 2 686 282 incidencias procesadas en **383 s**, sin instalar nada y
sin escribir 60 GB en disco. El lector está en
`scripts/00_lectores/jira_archive.py` y el extractor en
`scripts/01_extraccion/50_extract_jira.py`.
"""))

    c.append(code('''
campos = pd.DataFrame([
    dict(campo_jira="key", columna="task_id", uso="identificador; base de la desduplicación D5"),
    dict(campo_jira="fields.summary", columna="title", uso="texto"),
    dict(campo_jira="fields.description", columna="description", uso="texto"),
    dict(campo_jira="fields.timespent", columna="effort_hours", uso="OBJETIVO (segundos → horas)"),
    dict(campo_jira="fields.timeoriginalestimate", columna="HoursEstimate", uso="estimación humana"),
    dict(campo_jira="fields.aggregatetimespent", columna="—", uso="detección de agregación (D3)"),
    dict(campo_jira="fields.issuetype.name", columna="raw_kind / task_kind", uso="categoría"),
    dict(campo_jira="fields.project.key", columna="project", uso="proyecto"),
    dict(campo_jira="fields.assignee", columna="desarrollador", uso="identificador anonimizado"),
    dict(campo_jira="fields.priority.name", columna="priority", uso="prioridad"),
    dict(campo_jira="fields.created / resolutiondate", columna="created / completed", uso="orden temporal"),
    dict(campo_jira="fields.issuelinks", columna="n_links", uso="enlaces"),
    dict(campo_jira="fields.subtasks", columna="n_links_estructurales", uso="subtareas"),
    dict(campo_jira="fields.components", columna="n_componentes", uso="componentes"),
])
tabla(campos, "20_campos_jira",
      "Tabla 20 · Campos leídos del volcado y a qué columna de v3 corresponden")
'''))

    # ---------------------------------------------------------------- 3
    c.append(md("""
---
## 3. Cobertura: cuánto tiempo real registra cada Jira

El registro de trabajo es una función **opcional** de Jira. La mayoría de los
proyectos abiertos no la usa, así que el denominador enorme es lo que hace
utilizable el 2,93 % que sí la usa.
"""))

    c.append(code('''
cob = pd.read_csv(RAIZ / "tablas_origen" / "jira_publico_cobertura.csv")
cob = cob.sort_values("con_tiempo", ascending=False)
tabla(cob, "21_cobertura", "Tabla 21 · Cobertura de registro horario por organización")

print(f"\\nTOTAL incidencias        : {cob.issues.sum():,}")
print(f"      con tiempo real    : {cob.con_tiempo.sum():,} "
      f"({100*cob.con_tiempo.sum()/cob.issues.sum():.2f}%)")
print(f"      con estimación+tiempo: {cob.con_ambos.sum():,}")
print(f"      organizaciones sin ni una hora: "
      f"{int((cob.con_tiempo == 0).sum())} de {len(cob)}")

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4))
d = cob[cob.con_tiempo > 0]
y = np.arange(len(d))
ax[0].barh(y, d.issues, color=GRIS, label="incidencias totales")
ax[0].barh(y, d.con_tiempo, color=TINTA, label="con tiempo registrado")
ax[0].set_yticks(y, d.fuente)
ax[0].invert_yaxis()
ax[0].set_xscale("log")
ax[0].set_xlabel("incidencias (escala log)")
ax[0].legend()
ax[0].set_title("Volumen total frente a volumen aprovechable")

ax[1].barh(y, d.pct_tiempo, color=AMBAR)
ax[1].set_yticks(y, d.fuente)
ax[1].invert_yaxis()
ax[1].set_xlabel("% de incidencias con tiempo registrado")
ax[1].set_title("Tasa de registro · muy desigual entre organizaciones")
fig.suptitle("Figura 20 · Solo 10 de los 16 Jiras registran horas", x=.01,
             ha="left", fontsize=11, weight="600")
guardar(fig, "fig20_cobertura")
plt.show()
'''))

    # ---------------------------------------------------------------- 4
    c.append(md("""
---
## 4. Perfilado del bloque crudo

Antes de decidir qué descartar hay que entender qué llegó. Cuatro preguntas:
¿en qué unidad está `timespent`?, ¿está cuantizado como los puntos de historia?,
¿cuántas filas son en realidad contenedores?, y ¿cuánto de esto ya lo teníamos?
"""))

    c.append(code('''
import glob
fr = []
for p in sorted(glob.glob(str(INTER / "jira_publico" / "*.parquet"))):
    d = pd.read_parquet(p)
    d["fuente_jira"] = Path(p).stem
    fr.append(d)
crudo = pd.concat(fr, ignore_index=True)
crudo["horas"] = crudo.timespent / 3600.0
print(f"bloque crudo: {len(crudo):,} filas × {crudo.shape[1]} columnas")
print(f"proyectos: {crudo.project_key.nunique():,}")

print("\\nvalores de timespent más frecuentes (segundos):")
vc = crudo.timespent.value_counts().head(10)
print(pd.DataFrame({"segundos": vc.index, "minutos": vc.index / 60,
                    "filas": vc.values}).to_string(index=False))
'''))

    c.append(md("""
El valor más frecuente es **1 200 s = 20 minutos** y le siguen 1 800 (30 min),
2 400 (40 min) y 600 (10 min). Es decir: la unidad es **segundos**, y la gente
registra en incrementos de diez minutos. Nada que ver con la cuantización en
horas enteras que sí aparece en SiP.
"""))

    c.append(code('''
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
vc = crudo.timespent.value_counts().head(14).sort_index()
ax[0].bar(range(len(vc)), vc.values, color=TINTA, alpha=.85)
ax[0].set_xticks(range(len(vc)), [f"{v/60:g}" for v in vc.index], rotation=45)
ax[0].set_xlabel("minutos registrados")
ax[0].set_ylabel("filas")
ax[0].set_title("Granularidad de 10 minutos")
ax[0].yaxis.set_major_formatter(MILES)

granos = pd.Series({
    "múltiplo de 10 min": float((crudo.timespent % 600 == 0).mean()),
    "múltiplo de 1 h": float((crudo.timespent % 3600 == 0).mean()),
    "múltiplo de 8 h": float((crudo.timespent % 28800 == 0).mean()),
})
ax[1].barh(granos.index, 100 * granos.values, color=AMBAR)
ax[1].set_xlabel("% de las filas")
ax[1].set_title("Cuantización · no es como los puntos de historia")

h = crudo.horas[crudo.horas > 0]
ax[2].hist(np.log10(h), bins=70, color=TINTA, alpha=.85)
ax[2].axvline(np.log10(h.median()), color="#111820", ls="--", lw=1.2)
ax[2].text(np.log10(h.median()), ax[2].get_ylim()[1]*.9,
           f"  mediana {h.median():.2f} h", fontsize=8.5)
ax[2].set_xlabel("log10(horas)")
ax[2].set_title("Distribución del esfuerzo crudo")
fig.suptitle("Figura 21 · Qué aspecto tiene el tiempo registrado en Jira público",
             x=.01, ha="left", fontsize=11, weight="600")
guardar(fig, "fig21_perfil_crudo")
plt.show()
'''))

    c.append(code('''
v3 = pd.read_parquet(DATOS / "pbe_effort_v3.parquet")
claves_tawos = set(v3[v3.source == "TAWOS"].task_id
                   .str.replace("TAWOS-", "", regex=False))
solape = crudo.key.isin(claves_tawos)

diag = pd.DataFrame([
    dict(hallazgo="esfuerzo fuera de [0,05 h , 200 h]",
         filas=int(((crudo.horas < 0.05) | (crudo.horas > 200)).sum()),
         comentario=f"mínimo {crudo.horas.min()*3600:.0f} s, máximo "
                    f"{crudo.horas.max():,.0f} h"),
    dict(hallazgo="tipo contenedor (Epic, New Project…)",
         filas=int(crudo.issuetype.isin(
             ["Epic", "New Project", "Initiative", "Program", "Umbrella"]).sum()),
         comentario="agrupan trabajo de otras incidencias"),
    dict(hallazgo="agregación: agg_timespent > timespent y con subtareas",
         filas=int(((crudo.agg_timespent.fillna(0) > crudo.timespent)
                    & (crudo.n_subtareas.fillna(0) > 0)).sum()),
         comentario="el tiempo pertenece a los hijos"),
    dict(hallazgo="sin descripción",
         filas=int(crudo.description.isna().sum()),
         comentario="se conservan: es el caso real de PBE"),
    dict(hallazgo="ya presentes vía TAWOS",
         filas=int(solape.sum()),
         comentario="ambos corpus minan Apache, Atlassian, Sonatype, Hyperledger, Spring"),
])
tabla(diag, "22_diagnostico", "Tabla 22 · Diagnóstico del bloque crudo")

print("\\nsolape con TAWOS, por organización:")
print(crudo[solape].fuente_jira.value_counts().to_string())
'''))

    c.append(md("""
El solape con TAWOS **no es hipotético**: 5 100 incidencias aparecen
literalmente dos veces porque los dos corpus minan los mismos Jiras. Sin
desduplicar, una parte del «corpus nuevo» sería el corpus viejo contado otra
vez, y cualquier mejora medida estaría inflada.
"""))

    # ---------------------------------------------------------------- 5
    c.append(md("""
---
## 5. Criterios de descarte D1–D6

Se aplican **en este orden** sobre las 78 634 filas con tiempo registrado.
"""))

    c.append(code('''
reglas = pd.DataFrame([
    dict(id="D1", regla="esfuerzo fuera de [0,05 h , 200 h]",
         motivo="6 segundos y 208 571 horas no son tareas; el máximo crudo es un error de registro"),
    dict(id="D2", regla="tipo de incidencia contenedor",
         motivo="Epic y New Project agrupan trabajo de otras incidencias"),
    dict(id="D3", regla="agg_timespent > timespent y n_subtareas > 0",
         motivo="el tiempo registrado pertenece a los hijos, no a esta incidencia"),
    dict(id="D4", regla="sin texto utilizable (menos de 3 caracteres normalizados)",
         motivo="nada que vectorizar"),
    dict(id="D5", regla="clave de incidencia ya presente vía TAWOS",
         motivo="evita contar dos veces el mismo trabajo y inflar la mejora medida"),
    dict(id="D6", regla="texto ya presente en validación o prueba de v3",
         motivo="guarda de fuga: una fila de entrenamiento que repita un texto de prueba inflaría el propio A/B"),
])
tabla(reglas, "23_criterios_dropeo", "Tabla 23 · Criterios de descarte del bloque público")
'''))

    c.append(code('''
import hashlib, re
CONTENEDOR = {"Epic", "New Project", "Initiative", "Program", "Umbrella"}
MIN_H, MAX_H = 0.05, 200.0

def norma(s):
    return (s.fillna("").astype(str).str.lower()
            .str.replace(r"\\s+", " ", regex=True).str.strip())

d = crudo.copy()
pasos = []

def paso(mask, etiqueta):
    global d
    antes = len(d)
    d = d[mask].copy()
    pasos.append(dict(criterio=etiqueta, descartadas=antes - len(d), quedan=len(d)))

pasos.append(dict(criterio="entrada · incidencias con timespent > 0",
                  descartadas=0, quedan=len(d)))
paso((d.horas >= MIN_H) & (d.horas <= MAX_H), "D1 · esfuerzo fuera de rango")
paso(~d.issuetype.isin(CONTENEDOR), "D2 · tipo contenedor")
paso(~((d.agg_timespent.fillna(0) > d.timespent) & (d.n_subtareas.fillna(0) > 0)),
     "D3 · agregación de subtareas")

d["title"] = d.summary.fillna("").astype(str).str.strip()
d["descripcion"] = d.description.fillna("").astype(str).str.strip()
d["text"] = (d.title + ". " + d.descripcion).str.strip()
d["text"] = d.text.str.replace(r"^\\. ", "", regex=True).str.strip(". ")
d["text_norm"] = norma(d.text)
paso(d.text_norm.str.len() >= 3, "D4 · sin texto utilizable")
paso(~d.key.isin(claves_tawos), "D5 · ya presente vía TAWOS")

d["text_hash"] = [hashlib.md5(s.encode()).hexdigest()[:16] for s in d.text_norm]
reservado = set(v3[(v3.is_software_task == 1)
                   & v3.split_t3.isin(["val", "test"])].text_hash)
paso(~d.text_hash.isin(reservado), "D6 · texto en val/test de v3")

emb = pd.DataFrame(pasos)
emb["pct_restante"] = (100 * emb.quedan / len(crudo)).round(1)
tabla(emb, "24_embudo", "Tabla 24 · Embudo de descarte, paso a paso")

v4 = pd.read_parquet(DATOS / "pbe_effort_v4.parquet")
entregadas = int((v4.source == "JiraPublico").sum())
print(f"\\nfilas reconstruidas aquí : {len(d):,}")
print(f"filas en v4 entregado    : {entregadas:,}")
print(f"coinciden                : {len(d) == entregadas}")
'''))

    c.append(code('''
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
e = emb.iloc[1:]
y = np.arange(len(e))
ax[0].barh(y, e.descartadas, color=AVISO)
ax[0].set_yticks(y, [s.split(" · ")[0] for s in e.criterio])
ax[0].invert_yaxis()
ax[0].set_xlabel("filas descartadas")
ax[0].set_title("Cuánto quita cada criterio")
for i, (v, t) in enumerate(zip(e.descartadas, e.criterio)):
    ax[0].text(v + max(e.descartadas) * .02, i, f"{v:,}", va="center", fontsize=8.5)

ax[1].plot(range(len(emb)), emb.quedan, marker="o", color=TINTA, lw=2)
ax[1].fill_between(range(len(emb)), emb.quedan, color=SUAVE)
ax[1].set_xticks(range(len(emb)),
                 ["entrada"] + [s.split(" · ")[0] for s in emb.criterio[1:]],
                 rotation=45)
ax[1].set_ylabel("filas restantes")
ax[1].yaxis.set_major_formatter(MILES)
ax[1].set_title(f"De {len(crudo):,} a {len(d):,} ({100*len(d)/len(crudo):.1f}%)")
fig.suptitle("Figura 22 · Embudo de descarte del bloque público", x=.01,
             ha="left", fontsize=11, weight="600")
guardar(fig, "fig22_embudo")
plt.show()
'''))

    c.append(md("""
D4 y D6 descartan **cero** filas, y eso es exactamente lo que se quiere de una
guarda: se declara, se comprueba, y no hace falta. D6 en cero significa que
ningún texto del bloque nuevo coincidía con validación o prueba de v3, así que
el A/B posterior no está contaminado por construcción.
"""))

    # ---------------------------------------------------------------- 6
    c.append(md("""
---
## 6. Armonización de esquema

Las filas nuevas se vuelcan al esquema exacto de v3. Donde v3 tiene una columna
que este corpus no puede aportar (todo lo específico de SiP y las banderas de
cambio tras la estimación de TAWOS) se escribe nulo, nunca un valor inventado.
"""))

    c.append(code('''
jp = v4[v4.source == "JiraPublico"]
n3 = v4[v4.source != "JiraPublico"]

filas = []
for col in v4.columns:
    filas.append(dict(columna=col,
                      cobertura_v3=round(100 * n3[col].notna().mean(), 1),
                      cobertura_jira=round(100 * jp[col].notna().mean(), 1)))
cobcols = pd.DataFrame(filas)
solo3 = cobcols[(cobcols.cobertura_jira == 0) & (cobcols.cobertura_v3 > 0)]
solo4 = cobcols[(cobcols.cobertura_v3 == 0) & (cobcols.cobertura_jira > 0)]
print("columnas que solo existen en el bloque v3 (nulas en el nuevo):")
print("  " + ", ".join(solo3.columna))
print("\\ncolumnas que solo existen en el bloque nuevo:")
print("  " + ", ".join(solo4.columna) if len(solo4) else "  (ninguna)")
tabla(cobcols, "25_cobertura_columnas",
      "Tabla 25 · Cobertura de cada columna en los dos bloques")
'''))

    c.append(code('''
mapa = (jp.groupby(["raw_kind", "task_kind"]).size()
        .reset_index(name="filas").sort_values("filas", ascending=False).head(18))
tabla(mapa, "26_mapa_categorias",
      "Tabla 26 · Cómo se armoniza issuetype de Jira a task_kind")

fig, ax = plt.subplots(figsize=(9, 3.4))
k = jp.task_kind.value_counts()
med = jp.groupby("task_kind").effort_hours.median().reindex(k.index)
y = np.arange(len(k))
ax.barh(y, k.values, color=TINTA, alpha=.85)
ax.set_yticks(y, [f"{i}  (mediana {m:.2f} h)" for i, m in zip(k.index, med)])
ax.invert_yaxis()
ax.set_xlabel("tareas")
ax.xaxis.set_major_formatter(MILES)
ax.set_title("Figura 23 · Categorías del bloque público tras armonizar")
guardar(fig, "fig23_categorias")
plt.show()
'''))

    # ---------------------------------------------------------------- 7
    c.append(md("""
---
## 7. Criterios de mezcla

Esta es la parte que decide si los resultados posteriores significan algo. La
regla que gobierna todo:

> **El bloque de prueba de v3 queda intacto, fila por fila.** El material nuevo
> solo puede entrar en entrenamiento, en validación, o en un bloque *externo*
> que nunca se usa para entrenar.

El motivo es directo: si al añadir datos se cambiara también el examen, la
comparación «¿ayudan más datos?» mediría la dificultad de dos exámenes
distintos, no la calidad de dos modelos. Con la prueba fija, es un A/B limpio.

| # | criterio | qué garantiza |
|---|---|---|
| M1 | la prueba de v3 no se toca | el resultado preinscrito sigue siendo comparable |
| M2 | 150 proyectos del corpus nuevo se retienen **enteros** como bloque externo | mide arranque en frío real: organización jamás vista |
| M3 | el resto va a entrenamiento, con una franja de validación propia sorteada **por texto** | ningún texto repetido cruza la frontera |

M2 retiene proyectos completos y no filas sueltas: si un proyecto aparece en
entrenamiento y en el bloque externo, el modelo ya conoce la escala de esa
organización y el «arranque en frío» sería falso.
"""))

    c.append(code('''
sw = v4[v4.is_software_task == 1]
comp = (sw.groupby(["source", "split_t3"]).size().unstack(fill_value=0))
comp["total"] = comp.sum(axis=1)
tabla(comp.reset_index(), "27_composicion_v4",
      "Tabla 27 · Composición de v4 por fuente y bloque")

# --------- verificación de los tres criterios ---------
sw3 = v3[v3.is_software_task == 1]
te_v3 = set(sw3[sw3.split_t3 == "test"].task_id)
te_v4 = set(sw[sw.split_t3 == "test"].task_id)
h_te = set(sw[sw.split_t3 == "test"].text_hash)
h_tr = set(sw[sw.split_t3 == "train"].text_hash)
h_va = set(sw[sw.split_t3 == "val"].text_hash)
p_ext = set(sw[sw.split_t3 == "externo"].project)
p_tr = set(sw[sw.split_t3 == "train"].project)

ver = pd.DataFrame([
    dict(criterio="M1 · el bloque de prueba de v3 es idéntico",
         esperado="True", obtenido=str(te_v3 == te_v4),
         detalle=f"{len(te_v4):,} filas, mismos identificadores"),
    dict(criterio="M1 · ninguna fila nueva entró en prueba",
         esperado="0", obtenido=str(int((sw.split_t3 == "test").sum() - len(te_v3))),
         detalle="test solo contiene TAWOS y SiP"),
    dict(criterio="M3 · textos compartidos entrenamiento/prueba",
         esperado="0", obtenido=str(len(h_te & h_tr)), detalle="guarda de fuga D6"),
    dict(criterio="M3 · textos compartidos validación/prueba",
         esperado="0", obtenido=str(len(h_te & h_va)), detalle="guarda de fuga D6"),
    dict(criterio="M2 · proyectos del bloque externo vistos en entrenamiento",
         esperado="0", obtenido=str(len(p_ext & p_tr)),
         detalle=f"{len(p_ext)} proyectos retenidos enteros"),
    dict(criterio="integridad · identificadores duplicados",
         esperado="0", obtenido=str(int(v4.task_id.duplicated().sum())),
         detalle="tras D5, ninguna incidencia se repite"),
])
ver["pasa"] = np.where(ver.esperado == ver.obtenido, "SÍ", "NO")
tabla(ver, "28_verificacion_mezcla",
      "Tabla 28 · Verificación de los criterios de mezcla sobre el fichero entregado")
assert (ver.pasa == "SÍ").all(), "una comprobación de mezcla ha fallado"
print("\\nlas seis comprobaciones pasan sobre pbe_effort_v4.parquet")
'''))

    c.append(code('''
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
orden = ["train", "val", "test", "externo"]
p = comp.drop(columns="total").T.reindex(orden)
abajo = np.zeros(len(p))
for i, s in enumerate(p.columns):
    ax[0].bar(p.index, p[s], bottom=abajo, label=s, color=PALETA[i])
    abajo = abajo + p[s].to_numpy()
ax[0].set_ylabel("tareas")
ax[0].yaxis.set_major_formatter(MILES)
ax[0].legend()
ax[0].set_title("Composición de cada bloque · la prueba no recibe nada nuevo")

tam = sw[sw.split_t3 == "externo"].project.value_counts()
ax[1].hist(np.log10(tam), bins=26, color=AMBAR, alpha=.9)
ax[1].set_xlabel("log10(tareas por proyecto)")
ax[1].set_ylabel("proyectos")
ax[1].set_title(f"Bloque externo · {len(tam)} proyectos, {tam.sum():,} tareas")
fig.suptitle("Figura 24 · Cómo quedó repartido v4", x=.01, ha="left",
             fontsize=11, weight="600")
guardar(fig, "fig24_reparto")
plt.show()
'''))

    # ---------------------------------------------------------------- 8
    c.append(md("""
---
## 8. Perfil final de v4

El bloque nuevo **no procede de la misma distribución** que v3, y eso hay que
declararlo: la mediana del Jira público es de 0,83 h frente a 2,00 h de TAWOS y
SiP. No es un defecto del corpus, es una diferencia real de cultura de registro
—en proyectos abiertos se apunta el rato concreto que se dedicó— y condiciona
cómo se pueden usar estas filas.
"""))

    c.append(code('''
perfil = (sw.groupby("source").effort_hours
          .describe(percentiles=[.05, .25, .5, .75, .95]).round(2))
tabla(perfil.reset_index(), "29_perfil_v4",
      "Tabla 29 · Distribución del esfuerzo por fuente")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
for i, s in enumerate(["TAWOS", "SiP", "JiraPublico"]):
    e = sw[sw.source == s].effort_hours
    ax[0].hist(np.log10(e), bins=60, alpha=.55, density=True,
               label=f"{s}  n={len(e):,}  mediana {e.median():.2f} h",
               color=PALETA[i])
ax[0].set_xlabel("log10(horas)")
ax[0].set_ylabel("densidad")
ax[0].legend(fontsize=8)
ax[0].set_title("El desplazamiento de distribución, declarado")

datos = [np.log10(sw[sw.source == s].effort_hours) for s in
         ["TAWOS", "SiP", "JiraPublico"]]
bp = ax[1].boxplot(datos, tick_labels=["TAWOS", "SiP", "Jira púb."],
                   patch_artist=True, showfliers=False)
for caja, col in zip(bp["boxes"], PALETA):
    caja.set_facecolor(col)
    caja.set_alpha(.65)
for m in bp["medians"]:
    m.set_color("#111820")
ax[1].set_ylabel("log10(horas)")
ax[1].set_title("Mismo dato, vista de cajas")

tam = sw.groupby("source").project.nunique()
ax[2].bar(tam.index, tam.values, color=PALETA[:3])
for i, v in enumerate(tam.values):
    ax[2].text(i, v, f"  {v}", ha="center", va="bottom", fontsize=9)
ax[2].set_ylabel("proyectos distintos")
ax[2].set_title("Diversidad organizativa")
fig.suptitle("Figura 25 · v4 · qué aporta y en qué se diferencia el bloque nuevo",
             x=.01, ha="left", fontsize=11, weight="600")
guardar(fig, "fig25_perfil_v4")
plt.show()
'''))

    c.append(code('''
est = jp[jp.HoursEstimate.notna() & (jp.HoursEstimate > 0)]
print(f"filas del bloque nuevo con estimación humana: {len(est):,}")
print(f"  de ellas, tiempo registrado EXACTAMENTE igual a la estimación: "
      f"{int(est.estimacion_igual_a_real.sum()):,} "
      f"({100*est.estimacion_igual_a_real.mean():.1f}%)")

sip = v4[(v4.source == "SiP") & (v4.is_software_task == 1)
         & v4.HoursEstimate.notna() & (v4.HoursEstimate > 0)]
igual_sip = np.isclose(sip.effort_hours, sip.HoursEstimate)
print(f"\\ncontrol · la misma medida en SiP, que usa reloj automático: "
      f"{int(igual_sip.sum()):,} de {len(sip):,} ({100*igual_sip.mean():.1f}%)")
print("\\nLectura: la igualdad exacta es MÁS frecuente en SiP que en el corpus")
print("público, así que no es un artefacto del relleno automático de Jira. Es")
print("cómo se registra el trabajo en general, y por eso esas filas se conservan")
print("en ambos corpus en lugar de descartarse en uno solo.")

comp_est = pd.DataFrame([
    dict(corpus="SiP (reloj automático)", n=len(sip),
         pct_igual=round(100 * float(igual_sip.mean()), 1)),
    dict(corpus="Jira público (registro manual)", n=len(est),
         pct_igual=round(100 * float(est.estimacion_igual_a_real.mean()), 1)),
])
tabla(comp_est, "30_control_igualdad",
      "Tabla 30 · Control: ¿es la igualdad estimación = real un defecto del corpus nuevo?")
'''))

    c.append(code('''
NOTAS_V4 = {
    "estimacion_igual_a_real": "1 si timespent coincidía exactamente con la estimación original",
    "n_cambios": "número de entradas del changelog de la incidencia",
    "fuente_jira": "organización de origen dentro del corpus público",
    "split_t3": "train / val / test / externo · 'externo' son 150 proyectos retenidos enteros",
}
import importlib.util
dic3 = pd.read_csv(DATOS / "diccionario_columnas_v3.csv") \\
    if (DATOS / "diccionario_columnas_v3.csv").exists() else None
notas = {}
if dic3 is not None:
    notas = dict(zip(dic3.columna, dic3.descripcion.fillna("")))
notas.update(NOTAS_V4)

filas = []
for col in v4.columns:
    s = v4[col]
    filas.append(dict(columna=col, tipo=str(s.dtype),
                      no_nulos=int(s.notna().sum()),
                      cobertura_pct=round(100 * s.notna().mean(), 1),
                      distintos=int(s.nunique(dropna=True)),
                      ejemplo=str(s.dropna().iloc[0])[:44] if s.notna().any() else "",
                      descripcion=notas.get(col, "")))
dic = pd.DataFrame(filas)
dic.to_csv(DATOS / "diccionario_columnas_v4.csv", index=False, encoding="utf-8-sig")
print(f"escrito {DATOS / 'diccionario_columnas_v4.csv'}")

print(f"\\nv4: {len(v4):,} filas × {v4.shape[1]} columnas "
      f"({int((v4.is_software_task==1).sum()):,} tareas de software)")
print(f"SHA-256 (16): {sha(DATOS / 'pbe_effort_v4.parquet')}")
dic
'''))

    c.append(md("""
---
## Resumen

**Descarte.** De 78 634 filas con tiempo registrado sobreviven **72 501**
(92,2 %). El criterio que más quita es D5, la desduplicación contra TAWOS, con
5 045 filas: no es una precaución teórica, es solape real entre dos corpus que
minan los mismos Jiras.

**Mezcla.** El entrenamiento pasa de 10 597 a 63 926 filas —seis veces más— sin
tocar una sola fila del bloque de prueba, y con 150 proyectos retenidos enteros
para poder medir arranque en frío sobre organizaciones nunca vistas.

**Advertencia de uso.** El bloque nuevo tiene una mediana de esfuerzo 2,4 veces
menor que v3. Entrenar mezclando sin más desplaza las predicciones hacia abajo;
en los experimentos esto se aborda con reponderación, recalibrado y con el
objetivo expresado como razón contra la mediana del proyecto. Quien reutilice
este conjunto debe decidir explícitamente qué hace con esa diferencia en lugar
de ignorarla.

El cuaderno del conjunto v3 es `01_dataset_v3.ipynb`.
"""))
    return c


def notebook():
    return construir(celdas(), "Dataset v4")
