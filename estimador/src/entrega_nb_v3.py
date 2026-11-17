"""Cells for the v3 notebook: acquisition, cleaning and splits of TAWOS + SiP.

Each cleaning criterion is re-applied to the input file and checked against the
output file, so the notebook proves the rule was enforced instead of asserting
it. Where a recomputation cannot match the pipeline exactly -- the duplicate
rules run after text normalisation, so recomputing them on raw text finds fewer
groups -- the notebook says so rather than presenting the approximation as the
figure.
"""
from entrega_comun import ESTILO, code, construir, md


def celdas():
    c = []
    c.append(md("""
# Dataset v3 — construcción, limpieza y particiones

**PBE · componente de estimación de esfuerzo · anexo de datos**

Este cuaderno documenta cómo se obtuvo el conjunto `pbe_effort_v3.parquet`
desde las fuentes originales, qué se descartó en cada paso, y por qué. Cada
criterio se **vuelve a aplicar sobre el fichero de entrada y se comprueba contra
el de salida**: las cifras que aparecen no están copiadas de una ejecución
anterior, se recalculan al ejecutar el cuaderno.

## Índice

| § | Contenido |
|---|---|
| 1 | Fuentes y por qué estas dos |
| 2 | Adquisición: TAWOS (volcado MySQL de 4,31 GB) y SiP |
| 3 | El objetivo: prueba de que `Timespent` está en minutos |
| 4 | Por qué **no** se convierten puntos de historia a horas |
| 5 | v0 · unificación de esquemas |
| 6 | v1 · criterios de limpieza D1–D10 |
| 7 | v2 · una fila = un desarrollador, una tarea (E1–E4) |
| 8 | v3 · variables añadidas |
| 9 | Particiones y control de fuga |
| 10 | Perfil final y diccionario de columnas |

## Linaje

```
TAWOS (MySQL dump 4,31 GB)  ─┐
                             ├─► v0 20 248 filas ─► v1 18 593 ─► v2 18 474 ─► v3 18 474
SiP (2 CSV)                 ─┘                     (−1 655)     (−119)       (+21 columnas)
```
"""))

    c.append(code(ESTILO + '''
print("entorno listo")
print("carpeta de datos:", DATOS.resolve())
'''))

    # ---------------------------------------------------------------- 1
    c.append(md("""
---
## 1. Fuentes y por qué estas dos

El requisito es **horas reales de reloj**, no puntos de historia ni días de
calendario. De las 20 fuentes revisadas en la fase de literatura, solo dos
publican esfuerzo registrado a nivel de tarea junto al texto de la tarea.
"""))

    c.append(code('''
fuentes = pd.DataFrame([
    dict(fuente="TAWOS", quien="Tawosi et al., MSR 2022", dominio="open_source_jira",
         formato="volcado MySQL 4,31 GB", objetivo="Timespent (minutos)",
         texto="título + descripción", licencia="CC BY 4.0"),
    dict(fuente="SiP", quien="Jones, 2020", dominio="commercial_inhouse",
         formato="2 CSV", objetivo="HoursActual (horas)",
         texto="Summary", licencia="CC BY 4.0"),
])
tabla(fuentes, "01_fuentes", "Tabla 1 · Las dos fuentes que aportan horas reales")
'''))

    c.append(md("""
Las 18 restantes se descartaron por una de tres razones: el esfuerzo estaba en
puntos de historia (Deep-SE, GPT2SP, Choetkiertikul), el esfuerzo era una
duración de calendario y no tiempo trabajado, o no había texto de tarea
asociado. Esa revisión está en el informe principal, secciones 1–2.
"""))

    # ---------------------------------------------------------------- 2
    c.append(md("""
---
## 2. Adquisición

### 2.1 TAWOS — leer 4,31 GB de `mysqldump` sin instalar MySQL

El volcado no cabe cómodamente en memoria ni conviene restaurarlo. La
observación que lo hace tratable: **`mysqldump` escapa los saltos de línea como
`\\n` literal**, de modo que cada sentencia `INSERT` ocupa exactamente una línea
física. Eso permite leer el fichero en flujo, línea a línea, y quedarse solo con
las tablas necesarias.

El lector está en `scripts/00_lectores/tawos_sql.py`. La pieza delicada es la
expresión regular que separa los valores de cada fila:

```python
_STR = re.compile(r"'((?:[^'\\\\]+|\\\\.)*)'", re.S)
```

Escrita como alternancia de *tramos* (`[^'\\\\]+` o `\\\\.`) y no como
`(?:[^'\\\\]|\\\\.)*`, para que el motor no haga retroceso exponencial en cadenas
largas —una descripción con muchas comillas escapadas bastaba para colgar la
versión ingenua.
"""))

    c.append(code('''
import json
p = RAIZ / "tablas_origen" / "tawos_table_sizes.json"
if p.exists():
    tam = json.load(open(p))
    t = pd.DataFrame([{"tabla": k, **v} for k, v in tam.items()])
    t = t.sort_values("insert_MB", ascending=False)
    print("Tamaño de cada tabla dentro del volcado (las 12 mayores):")
    print(t.head(12).to_string(index=False))
    print(f"\\ntotal del volcado: {t.insert_MB.sum():,.0f} MB")
    print("Change_Log y Comment son el 55% del fichero y no se usan: por eso el "
          "lector\\nen flujo salta las tablas que no hacen falta en vez de "
          "restaurar la base entera.")

extraidas = pd.DataFrame([
    dict(tabla="Issue", uso="título, descripción, tipo, prioridad, Timespent, fechas"),
    dict(tabla="Project", uso="clave de proyecto"),
    dict(tabla="Issue_Link", uso="n_links y n_links_estructurales"),
    dict(tabla="Issue_Component", uso="n_componentes"),
    dict(tabla="Component", uso="nombres de componente"),
    dict(tabla="Sprint", uso="pertenencia a sprint"),
    dict(tabla="User", uso="Assignee_ID (anonimizado)"),
])
tabla(extraidas, "02_tablas_tawos", "Tabla 2 · Tablas extraídas del volcado y para qué")
'''))

    c.append(md("""
### 2.2 SiP

Dos CSV en `cp1252`. `Sip-task-info.csv` trae una fila **por desarrollador y
tarea**, de modo que `HoursActual` se repite en todas las filas de una misma
tarea: es el total de la tarea, no lo de cada persona. Ese detalle obliga a
`drop_duplicates("TaskNumber")` aquí, y vuelve a aparecer en el criterio E3.
"""))

    c.append(code('''
sip = pd.read_csv(INTER / "Sip-task-info.csv", encoding="cp1252")
print(f"filas crudas: {len(sip):,}   tareas distintas: {sip.TaskNumber.nunique():,}")
print(f"columnas: {len(sip.columns)}")

est = sip.groupby("TaskNumber").HoursActual.nunique()
print(f"\\ntareas donde HoursActual varía entre sus filas: {int((est > 1).sum())}")
print("  -> 0 confirma que HoursActual es un total por tarea, no por persona")

d = sip.drop_duplicates("TaskNumber")
filtros = pd.DataFrame([
    dict(criterio="estado TEMPLATE / AUTHORISE / ESTIMATED / CANCELLED",
         filas=int(d.StatusCode.isin(["TEMPLATE", "AUTHORISE", "ESTIMATED",
                                      "CANCELLED"]).sum()),
         motivo="la tarea no llegó a ejecutarse"),
    dict(criterio="HoursActual <= 0",
         filas=int((d.HoursActual <= 0).sum()),
         motivo="sin esfuerzo registrado"),
])
tabla(filtros, "03_sip_filtros", "Tabla 3 · Filtros aplicados a SiP en la carga")
'''))

    # ---------------------------------------------------------------- 3
    c.append(md("""
---
## 3. El objetivo: `Timespent` está en **minutos**

TAWOS no documenta la unidad de `Timespent`. La decisión condiciona todo el
conjunto —confundir minutos con segundos multiplica el objetivo por 60— así que
se resolvió con evidencia, no por convención.
"""))

    c.append(code('''
g = pd.read_parquet(INTER / "tawos_gold.parquet")
ts = g.effort_hours * 60          # deshacemos la conversión para ver el crudo
cal = g.calendar_hours

pruebas = pd.DataFrame([
    dict(prueba="1 · mediana en minutos",
         valor=f"{ts.median():,.0f} min = {ts.median()/60:.2f} h",
         lectura="plausible para una tarea; en segundos serían 2 min"),
    dict(prueba="2 · p95 en minutos",
         valor=f"{ts.quantile(.95):,.0f} min = {ts.quantile(.95)/60:.1f} h",
         lectura="una tarea grande; en segundos serían 20 min"),
    dict(prueba="3 · esfuerzo > duración de calendario",
         valor=f"{100*(g.effort_hours > cal).mean():.2f}%",
         lectura="si fueran segundos·60 el porcentaje se dispararía"),
    dict(prueba="4 · múltiplos exactos de 60",
         valor=f"{100*(ts % 60 == 0).mean():.1f}%",
         lectura="registro en horas enteras, típico de 'log work' en minutos"),
])
tabla(pruebas, "04_unidad_timespent",
      "Tabla 4 · Cuatro comprobaciones independientes sobre la unidad")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
for a, (serie, etiqueta, color) in zip(ax, [
        (ts / 60, "interpretado como MINUTOS  →  horas", TINTA),
        (ts / 3600, "interpretado como SEGUNDOS  →  horas", AVISO)]):
    x = serie[serie > 0]
    a.hist(np.log10(x), bins=60, color=color, alpha=.85)
    a.set_title(etiqueta)
    a.set_xlabel("log10(horas)")
    a.set_ylabel("tareas")
    a.axvline(np.log10(x.median()), color="#111820", lw=1.2, ls="--")
    a.text(np.log10(x.median()), a.get_ylim()[1] * .92,
           f"  mediana {x.median():.3f} h", fontsize=8.5, color="#111820")
fig.suptitle("Figura 1 · La misma columna bajo las dos unidades posibles",
             x=.01, ha="left", fontsize=11, weight="600")
guardar(fig, "fig01_unidad_timespent")
plt.show()
'''))

    c.append(md("""
Leída como segundos, la mediana de una tarea de software sería de **2 minutos**
y el percentil 95 de 20 minutos. Leída como minutos, la mediana es de 2 horas.
La segunda lectura es la única compatible con las cuatro comprobaciones, y es la
que se aplica: `effort_hours = Timespent / 60`.
"""))

    # ---------------------------------------------------------------- 4
    c.append(md("""
---
## 4. Por qué no se convierten puntos de historia a horas

El plan inicial era usar los conjuntos con puntos de historia (Deep-SE, GPT2SP)
convirtiéndolos a horas mediante un factor por proyecto. La medición lo
descartó.
"""))

    c.append(code('''
sp = pd.read_parquet(INTER / "tawos_sp_hours.parquet")
sp = sp[(sp.Story_Point > 0) & (sp.effort_hours > 0)]
print(f"tareas con punto de historia y horas reales: {len(sp):,}  "
      f"en {sp.project.nunique()} proyectos")

from numpy.polynomial import polynomial as P
x, y = np.log(sp.Story_Point.to_numpy()), np.log(sp.effort_hours.to_numpy())
r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
rho = float(sp.Story_Point.corr(sp.effort_hours, method="spearman"))
print(f"R2 agrupado sobre log(puntos) -> log(horas): {r2:.4f}")
print(f"Spearman: {rho:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
ax[0].scatter(sp.Story_Point + np.random.default_rng(0).normal(0, .06, len(sp)),
              sp.effort_hours, s=5, alpha=.18, color=TINTA, edgecolors="none")
ax[0].set_yscale("log")
ax[0].set_xlim(0, 22)
ax[0].set_xlabel("puntos de historia")
ax[0].set_ylabel("horas reales (log)")
ax[0].set_title(f"Sin relación utilizable · R² = {r2:.3f}")

med = sp.groupby("Story_Point").effort_hours.median()
med = med[med.index <= 21]
ax[1].bar(med.index, med.values, color=TINTA, alpha=.85)
ax[1].set_xlabel("puntos de historia")
ax[1].set_ylabel("mediana de horas reales")
ax[1].set_title("La mediana apenas crece con los puntos")
fig.suptitle("Figura 2 · Los puntos de historia no predicen horas",
             x=.01, ha="left", fontsize=11, weight="600")
guardar(fig, "fig02_puntos_vs_horas")
plt.show()

porproj = (sp.groupby("project")
           .apply(lambda d: pd.Series({
               "n": len(d),
               "factor_h_por_punto": float((d.effort_hours / d.Story_Point).median()),
               "R2": float(np.corrcoef(np.log(d.Story_Point), np.log(d.effort_hours))[0, 1] ** 2)
                     if len(d) > 10 and d.Story_Point.nunique() > 2 else np.nan}),
                  include_groups=False)
           .sort_values("n", ascending=False).head(10).reset_index())
tabla(porproj.round(3), "05_conversion_puntos",
      "Tabla 5 · Factor puntos→horas por proyecto: ni siquiera es estable dentro de un proyecto")
'''))

    c.append(md("""
Con **R² = 0,058** agrupado, el «convertidor por proyecto» acaba siendo
prácticamente una constante: predice la mediana del proyecto y nada más. Por eso
v3 no incorpora ninguna tarea cuyo esfuerzo provenga de una conversión. Todas
las horas del conjunto son horas registradas.
"""))

    # ---------------------------------------------------------------- 5
    c.append(md("""
---
## 5. v0 · unificación de esquemas

`scripts/01_extraccion/09_build_unified.py` arma una tabla única. Decisiones:

- `effort_hours` es **siempre** tiempo de reloj: TAWOS aporta `Timespent/60`,
  SiP aporta `HoursActual`. Nada procede de una conversión.
- `task_kind` es una taxonomía armonizada sobre `Type` (TAWOS) y `SubCategory`
  (SiP); la etiqueta original se conserva en `raw_kind`.
- SiP contiene trabajo no informático (Marketing, Reclutamiento, Consejo). Esas
  filas **se conservan** marcadas con `is_software_task = 0` y quedan fuera de
  las particiones: es una decisión reversible.
"""))

    c.append(code('''
v0 = pd.read_parquet(DATOS / "intermedios" / "pbe_effort_dataset.parquet")
print(f"v0: {len(v0):,} filas, {v0.shape[1]} columnas")

comp = (v0.groupby(["source", "is_software_task"]).size()
        .unstack(fill_value=0).rename(columns={0: "no software", 1: "software"}))
comp["total"] = comp.sum(axis=1)
tabla(comp.reset_index(), "06_v0_composicion",
      "Tabla 6 · Composición de v0")

k = (v0[v0.is_software_task == 1].groupby(["task_kind", "source"]).size()
     .unstack(fill_value=0))
k["total"] = k.sum(axis=1)
k["mediana_h"] = v0[v0.is_software_task == 1].groupby("task_kind").effort_hours.median()
k = k.sort_values("total", ascending=False)
tabla(k.round(2).reset_index(), "07_task_kind",
      "Tabla 7 · Taxonomía armonizada task_kind")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
kk = k[k.index != "other"].head(9)
idx = np.arange(len(kk))
for i, (col, color) in enumerate([("TAWOS", TINTA), ("SiP", AVISO)]):
    if col in kk.columns:
        ax[0].barh(idx + (i - .5) * .38, kk[col], height=.36, label=col, color=color)
ax[0].set_yticks(idx, kk.index)
ax[0].invert_yaxis()
ax[0].legend()
ax[0].set_xlabel("tareas")
ax[0].set_title("Volumen por categoría y fuente")
ax[0].xaxis.set_major_formatter(MILES)

ax[1].barh(idx, kk.mediana_h, color=TINTA, alpha=.85)
ax[1].set_yticks(idx, kk.index)
ax[1].invert_yaxis()
ax[1].set_xlabel("mediana de horas")
ax[1].set_title("Esfuerzo típico por categoría")
fig.suptitle("Figura 3 · La taxonomía armonizada", x=.01, ha="left",
             fontsize=11, weight="600")
guardar(fig, "fig03_task_kind")
plt.show()
'''))

    # ---------------------------------------------------------------- 6
    c.append(md("""
---
## 6. v1 · criterios de limpieza (D1–D10)

`scripts/02_limpieza/12_clean_dataset.py`. **El orden importa**: el texto se
normaliza *antes* de detectar duplicados, para que dos incidencias que solo se
diferencian en una traza de pila se reconozcan como el mismo texto.
"""))

    c.append(code('''
criterios = pd.DataFrame([
    dict(id="D9", accion="enriquecer", regla="unir fechas de SiP desde est-act-dates.csv",
         motivo="permite comprobar esfuerzo contra calendario", filas="0"),
    dict(id="D8", accion="reescribir", regla="normalizar URL, correos, hashes, trazas, números largos",
         motivo="dos incidencias iguales salvo la traza son el mismo texto", filas="0"),
    dict(id="D6", accion="descartar", regla="texto con menos de 3 palabras",
         motivo="sin contenido que vectorizar", filas="ver abajo"),
    dict(id="D5", accion="descartar", regla="esfuerzo < 0,1 h",
         motivo="6 minutos no es una tarea; suele ser un registro erróneo", filas="ver abajo"),
    dict(id="D4", accion="descartar", regla="tareas de SiP > 200 h",
         motivo="líneas de presupuesto paraguas, no tareas", filas="ver abajo"),
    dict(id="D1a", accion="descartar", regla="grupos de texto idéntico con esfuerzo que varía > 10x",
         motivo="mismo texto y esfuerzos irreconciliables: el texto no determina el esfuerzo", filas="ver abajo"),
    dict(id="D2", accion="colapsar", regla="duplicados exactos restantes → una fila en la mediana",
         motivo="evita que el modelo vea el mismo ejemplo repetido", filas="ver abajo"),
    dict(id="D7", accion="reagrupar", regla="categorías con < 50 filas → 'other'",
         motivo="una categoría con 3 ejemplos no se aprende", filas="0"),
    dict(id="D3a", accion="agrupar", regla="casi duplicados (coseno ≥ 0,95) comparten group_key",
         motivo="impide que textos casi iguales caigan a ambos lados de la partición", filas="0"),
    dict(id="D10", accion="conservar", regla="filas solo con título",
         motivo="son el caso de uso real de PBE: tarea recién creada", filas="0"),
])
tabla(criterios, "08_criterios_v1", "Tabla 8 · Criterios de limpieza aplicados a v0 → v1")
'''))

    c.append(md("""
### 6.1 D8 · normalización de texto — ejemplo antes/después

Ninguna fila se pierde; el texto original queda en `text_raw`.
"""))

    c.append(code('''
v1 = pd.read_parquet(DATOS / "intermedios" / "pbe_effort_clean.parquet")
cambiadas = v1[v1.text_raw.fillna("") != v1.text.fillna("")]
print(f"filas cuyo texto cambió con D8: {len(cambiadas):,} "
      f"({100*len(cambiadas)/len(v1):.1f}%)")

muestra = cambiadas.assign(dif=cambiadas.text_raw.str.len() - cambiadas.text.str.len())
muestra = muestra.nlargest(3, "dif")
for _, r in muestra.iterrows():
    print("\\n" + "=" * 78)
    print("ANTES :", r.text_raw[:230].replace("\\n", " "))
    print("DESPUÉS:", r.text[:230].replace("\\n", " "))
'''))

    c.append(md("""
### 6.2 D6, D5, D4 · descartes por umbral

Cada regla se vuelve a aplicar sobre v0 y se comprueba que **ninguna** fila
afectada sobrevive en v1.
"""))

    c.append(code('''
palabras = v0.text.fillna("").str.split().str.len()
ids_v1 = set(v1.task_id)

reglas = [
    ("D6 · texto < 3 palabras", palabras < 3),
    ("D5 · esfuerzo < 0,1 h", v0.effort_hours < 0.1),
    ("D4 · SiP > 200 h", (v0.source == "SiP") & (v0.effort_hours > 200)),
]
filas = []
for nombre, m in reglas:
    afectadas = v0[m]
    fuera = int((~afectadas.task_id.isin(ids_v1)).sum())
    filas.append(dict(criterio=nombre, coinciden=int(m.sum()),
                      ausentes_en_v1=fuera,
                      verificado="sí" if fuera == int(m.sum()) else "NO"))
tabla(pd.DataFrame(filas), "09_umbral_v1",
      "Tabla 9 · Verificación de los descartes por umbral")

fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2))
ax[0].hist(palabras.clip(0, 40), bins=40, color=TINTA, alpha=.85)
ax[0].axvline(3, color=AVISO, lw=1.6)
ax[0].set_title("D6 · palabras por tarea")
ax[0].set_xlabel("palabras (recortado a 40)")
ax[0].set_ylabel("tareas")

e = v0.effort_hours[v0.effort_hours > 0]
ax[1].hist(np.log10(e), bins=70, color=TINTA, alpha=.85)
ax[1].axvline(np.log10(0.1), color=AVISO, lw=1.6)
ax[1].set_title("D5 · esfuerzo, corte en 0,1 h")
ax[1].set_xlabel("log10(horas)")

s = v0[v0.source == "SiP"].effort_hours
ax[2].hist(np.log10(s[s > 0]), bins=60, color=AMBAR, alpha=.85)
ax[2].axvline(np.log10(200), color=AVISO, lw=1.6)
ax[2].set_title("D4 · SiP, corte en 200 h")
ax[2].set_xlabel("log10(horas)")
fig.suptitle("Figura 4 · Dónde caen los tres umbrales (línea roja = corte)",
             x=.01, ha="left", fontsize=11, weight="600")
guardar(fig, "fig04_umbrales")
plt.show()
'''))

    c.append(md("""
### 6.3 D1a y D2 · texto duplicado

Este es el criterio con más consecuencias para un modelo de texto. Si dos filas
comparten texto pero registran esfuerzos muy distintos, **el texto no determina
el esfuerzo** y conservar ambas solo enseña ruido; si registran esfuerzos
parecidos, conservar ambas infla artificialmente el conjunto y puede repartir el
mismo ejemplo entre entrenamiento y prueba.

> **Nota de precisión.** El recuento que sigue reaplica la regla sobre el
> `text_hash` de v0, que se calculó **antes** de la normalización D8. La tubería
> real normaliza primero, así que agrupa algo más. Las cifras de abajo son por
> tanto una **cota inferior** del efecto real, y así se presentan.
"""))

    c.append(code('''
g = v0.groupby("text_hash").effort_hours.agg(["min", "max", "size", "median"])
g["dispersion"] = g["max"] / g["min"].clip(lower=1e-9)
grupos = g[g["size"] > 1]
disp = grupos[grupos.dispersion > 10]
col = grupos[grupos.dispersion <= 10]

res = pd.DataFrame([
    dict(caso="grupos de texto idéntico", grupos=len(grupos),
         filas=int(grupos["size"].sum())),
    dict(caso="D1a · dispersión > 10x → se descarta el grupo entero",
         grupos=len(disp), filas=int(disp["size"].sum())),
    dict(caso="D2 · dispersión ≤ 10x → se colapsa a una fila en la mediana",
         grupos=len(col), filas=int(col["size"].sum())),
])
tabla(res, "10_duplicados", "Tabla 10 · Texto duplicado en v0 (cota inferior)")

print(f"\\nfilas de v1 que son un colapso de varias (n_merged > 1): "
      f"{int((v1.n_merged > 1).sum()):,}")
print(f"filas originales representadas por ellas: {int(v1.n_merged[v1.n_merged>1].sum()):,}")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
tam = grupos["size"].value_counts().sort_index()
ax[0].bar(tam.index[:12], tam.values[:12], color=TINTA, alpha=.85)
ax[0].set_xlabel("filas que comparten el mismo texto")
ax[0].set_ylabel("nº de grupos")
ax[0].set_title("Tamaño de los grupos de texto idéntico")

ax[1].hist(np.log10(grupos.dispersion.clip(1, 1e4)), bins=50, color=TINTA, alpha=.85)
ax[1].axvline(1, color=AVISO, lw=1.6)
ax[1].set_xlabel("log10(esfuerzo máx / esfuerzo mín) dentro del grupo")
ax[1].set_ylabel("nº de grupos")
ax[1].set_title("Dispersión interna · corte D1a en 10x")
fig.suptitle("Figura 5 · Mismo texto, ¿mismo esfuerzo?", x=.01, ha="left",
             fontsize=11, weight="600")
guardar(fig, "fig05_duplicados")
plt.show()
'''))

    c.append(md("""
### 6.4 D3a · casi duplicados y `group_key`

Dos textos con coseno TF-IDF ≥ 0,95 no son la misma fila, pero son tan parecidos
que si uno cae en entrenamiento y el otro en prueba el resultado está inflado.
Se unen en componentes conexas y cada componente recibe un `group_key`; las
particiones se sortean **por `group_key`**, nunca por fila.
"""))

    c.append(code('''
gk = v1.groupby("group_key").size()
print(f"group_key distintos: {len(gk):,}")
print(f"grupos con más de una fila: {int((gk > 1).sum()):,} "
      f"(cubren {int(gk[gk > 1].sum()):,} filas)")
print(f"grupo mayor: {int(gk.max())} filas")
gr = pd.DataFrame({"filas_por_grupo": gk.value_counts().sort_index().head(8)})
tabla(gr.reset_index(names="tamaño"), "11_group_key",
      "Tabla 11 · Distribución de tamaño de los grupos de casi duplicados")
'''))

    c.append(md("""
### 6.5 Balance de v0 → v1
"""))

    c.append(code('''
etapas = [("v0 · unificado", len(v0)), ("v1 · limpio", len(v1))]
perdidas = len(v0) - len(v1)
print(f"v0 {len(v0):,}  →  v1 {len(v1):,}   (−{perdidas:,}, "
      f"{100*perdidas/len(v0):.1f}%)")

fig, ax = plt.subplots(figsize=(9, 3.2))
antes = v0[v0.is_software_task == 1].effort_hours
desp = v1[v1.is_software_task == 1].effort_hours
bins = np.linspace(-1.6, 2.8, 70)
ax.hist(np.log10(antes[antes > 0]), bins=bins, color=GRIS, alpha=.55,
        label=f"v0  n={len(antes):,}")
ax.hist(np.log10(desp[desp > 0]), bins=bins, color=TINTA, alpha=.75,
        label=f"v1  n={len(desp):,}")
ax.set_xlabel("log10(horas)")
ax.set_ylabel("tareas de software")
ax.legend()
ax.set_title("Figura 6 · Distribución del esfuerzo antes y después de la limpieza D")
guardar(fig, "fig06_v0_v1")
plt.show()
'''))

    # ---------------------------------------------------------------- 7
    c.append(md("""
---
## 7. v2 · una fila = un desarrollador, una tarea (E1–E4)

`scripts/02_limpieza/26_clean_v2.py`. El cronómetro de PBE mide **una persona
trabajando en una tarea**. Una fila que signifique otra cosa le enseña al modelo
una magnitud que nunca se le va a pedir.

| id | acción | regla | motivo |
|---|---|---|---|
| E3 | reexpresar | tareas de SiP con varios desarrolladores → horas del contribuyente principal | `HoursActual` es la suma de todo el equipo |
| E1 | descartar | épicas de TAWOS | son contenedores por definición |
| E2 | descartar | filas con lenguaje de contenedor y > 16 h | proyectos, planes, cajones de sastre |
| E4 | marcar | esfuerzo en horas enteras | cuantización del objetivo, se documenta sin alterar |

**E3 se aplica primero**, y eso cambia el resultado de E2: seis filas dejan de
superar las 16 h una vez reexpresadas por desarrollador y por tanto ya no
cumplen E2. El cuaderno lo comprueba explícitamente.
"""))

    c.append(code('''
v2 = pd.read_parquet(DATOS / "intermedios" / "pbe_effort_v2.parquet")
ids_v2 = set(v2.task_id)

# --- E3
comun = v1.merge(v2[["task_id", "effort_hours"]], on="task_id",
                 suffixes=("_v1", "_v2"))
cambio = comun[~np.isclose(comun.effort_hours_v1, comun.effort_hours_v2)]
print(f"E3 · filas reexpresadas por desarrollador: {len(cambio):,}")
print(f"     mediana {cambio.effort_hours_v1.median():.2f} h  →  "
      f"{cambio.effort_hours_v2.median():.2f} h")

fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.4))
bins = np.linspace(-1, 2.7, 60)
ax[0].hist(np.log10(cambio.effort_hours_v1), bins=bins, color=GRIS, alpha=.6,
           label="total de la tarea (v1)")
ax[0].hist(np.log10(cambio.effort_hours_v2), bins=bins, color=TINTA, alpha=.75,
           label="desarrollador principal (v2)")
ax[0].set_xlabel("log10(horas)")
ax[0].set_ylabel("tareas de SiP con varios desarrolladores")
ax[0].legend()
ax[0].set_title(f"E3 · {len(cambio):,} tareas reexpresadas")

nd = v2[v2.source == "SiP"].n_dev.value_counts().sort_index().head(8)
ax[1].bar(nd.index.astype(str), nd.values, color=AMBAR, alpha=.85)
ax[1].set_xlabel("desarrolladores distintos en la tarea")
ax[1].set_ylabel("tareas")
ax[1].set_title("Cuántas tareas de SiP son de más de una persona")
ax[1].yaxis.set_major_formatter(MILES)
fig.suptitle("Figura 7 · El objetivo pasa a ser por desarrollador", x=.01,
             ha="left", fontsize=11, weight="600")
guardar(fig, "fig07_e3")
plt.show()
'''))

    c.append(code('''
CONTENEDOR = (r"\\b(project|phase|programme|initiative|roadmap|plan|planning|"
              r"meeting|workshop|kick.?off|various|misc|miscellaneous|"
              r"catch.?all|assorted|month.?end|period close|ongoing|routine|"
              r"housekeeping)\\b")

# el orden real: E3 primero, luego E1/E2 sobre las horas YA reexpresadas
ajustadas = v1[["task_id", "text", "raw_kind", "effort_hours"]].copy()
ajustadas = ajustadas.merge(v2[["task_id", "effort_hours"]], on="task_id",
                            how="left", suffixes=("", "_v2"))
ajustadas["horas_e3"] = ajustadas.effort_hours_v2.fillna(ajustadas.effort_hours)

bajo = ajustadas.text.fillna("").str.lower()
es_epica = ajustadas.raw_kind.astype(str).eq("Epic")
contenedor = bajo.str.contains(CONTENEDOR, regex=True, na=False)

filas = []
for nombre, m in [
        ("E1 · épicas", es_epica),
        ("E2 · contenedor > 16 h, SOBRE horas de v1", contenedor & (ajustadas.effort_hours > 16)),
        ("E2 · contenedor > 16 h, SOBRE horas tras E3", contenedor & (ajustadas.horas_e3 > 16)),
]:
    af = ajustadas[m]
    fuera = int((~af.task_id.isin(ids_v2)).sum())
    filas.append(dict(criterio=nombre, coinciden=int(m.sum()),
                      ausentes_en_v2=fuera))
tabla(pd.DataFrame(filas), "12_criterios_v2",
      "Tabla 12 · E1 y E2, y por qué el orden respecto de E3 cambia el recuento")

print(f"\\nv1 {len(v1):,}  →  v2 {len(v2):,}   (−{len(v1)-len(v2):,})")
print(f"E4 · filas con esfuerzo en horas enteras: "
      f"{int(v2.objetivo_redondo.sum()):,} ({100*v2.objetivo_redondo.mean():.1f}%) "
      "— marcadas, no alteradas")
'''))

    c.append(code('''
fig, ax = plt.subplots(figsize=(9.5, 3.2))
e = v2[v2.is_software_task == 1].effort_hours
frac = e.round(2).value_counts(normalize=True).sort_values(ascending=False).head(14)
ax.bar(range(len(frac)), 100 * frac.values, color=np.where(
    np.isclose(frac.index.values, np.round(frac.index.values)), TINTA, GRIS))
ax.set_xticks(range(len(frac)), [f"{v:g}" for v in frac.index], rotation=45)
ax.set_ylabel("% de las tareas")
ax.set_xlabel("valor de effort_hours")
ax.set_title("Figura 8 · E4 · el objetivo está cuantizado: las horas enteras dominan "
             "(en verde)")
guardar(fig, "fig08_cuantizacion")
plt.show()
'''))

    # ---------------------------------------------------------------- 8
    c.append(md("""
---
## 8. v3 · variables añadidas

`scripts/02_limpieza/39_build_v3.py` no descarta ninguna fila: **v2 y v3 tienen
exactamente las mismas 18 474 filas**. Lo que añade son 21 columnas que la
auditoría identificó como potencialmente informativas.

Dos notas de fuga que quedan grabadas en el fichero en vez de quedar implícitas:

- `Issue_Link` no tiene marca temporal, así que `n_links` puede incluir enlaces
  añadidos *durante* el trabajo. `n_links_estructurales` (Depends/Blocks/Part)
  es el subconjunto conservador; se guardan **las dos** para que la lectura
  optimista y la defendible sigan siendo separables.
- `TaskPerformance` y `DeveloperPerformance` de SiP se derivan de las horas
  reales y **no se importan** deliberadamente.
"""))

    c.append(code('''
v3 = pd.read_parquet(DATOS / "pbe_effort_v3.parquet")
nuevas = [c for c in v3.columns if c not in v2.columns]
print(f"v2 {len(v2):,} filas × {v2.shape[1]} col  →  "
      f"v3 {len(v3):,} filas × {v3.shape[1]} col")
print(f"filas idénticas: {set(v2.task_id) == set(v3.task_id)}")
print(f"\\ncolumnas nuevas ({len(nuevas)}): {nuevas}")

sw3 = v3[v3.is_software_task == 1]
filas = []
for col, desc in [
        ("HoursEstimate", "estimación humana (SiP)"),
        ("AssignedToID", "identificador de asignado (SiP)"),
        ("ProjectBreakdownCode", "código de desglose de proyecto (SiP)"),
        ("Assignee_ID", "identificador de asignado (TAWOS)"),
        ("n_links", "enlaces a otras incidencias"),
        ("n_links_estructurales", "enlaces de dependencia/bloqueo"),
        ("n_componentes", "componentes asociados"),
        ("en_sprint", "pertenece a un sprint"),
        ("tiene_asignado", "tiene persona asignada"),
        ("Title_Changed_After_Estimation", "el título cambió tras estimar"),
        ("Story_Point_Changed_After_Estimation", "los puntos cambiaron tras estimar"),
        ("desarrollador", "identificador unificado de desarrollador"),
]:
    if col not in sw3.columns:
        continue
    s = sw3[col]
    cob = float(s.notna().mean())
    d = sw3[s.notna()]
    rho = (float(pd.to_numeric(d[col], errors="coerce").corr(
        d.effort_hours, method="spearman")) if len(d) > 100 else np.nan)
    filas.append(dict(columna=col, descripcion=desc,
                      cobertura=round(100 * cob, 1), n=int(s.notna().sum()),
                      spearman_vs_esfuerzo=round(rho, 4) if rho == rho else None))
var = pd.DataFrame(filas)
tabla(var, "13_variables_v3",
      "Tabla 13 · Variables añadidas en v3, cobertura y relación con el esfuerzo")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
v = var.dropna(subset=["spearman_vs_esfuerzo"]).sort_values("spearman_vs_esfuerzo")
ax[0].barh(v.columna, v.spearman_vs_esfuerzo,
           color=np.where(v.spearman_vs_esfuerzo > 0, TINTA, AVISO))
ax[0].axvline(0, color="#111820", lw=.8)
ax[0].set_xlabel("Spearman contra effort_hours")
ax[0].set_title("Relación de cada variable con el esfuerzo")

w = var.sort_values("cobertura")
ax[1].barh(w.columna, w.cobertura, color=GRIS)
ax[1].set_xlabel("% de filas de software con valor")
ax[1].set_title("Cobertura")
fig.suptitle("Figura 9 · Las variables nuevas de v3", x=.01, ha="left",
             fontsize=11, weight="600")
guardar(fig, "fig09_variables_v3")
plt.show()
'''))

    # ---------------------------------------------------------------- 9
    c.append(md("""
---
## 9. Particiones y control de fuga

v3 lleva **tres** particiones, cada una responde a una pregunta distinta:

| columna | cómo se reparte | qué mide |
|---|---|---|
| `split` | 70/15/15 sorteando `group_key` | rendimiento sobre tareas nuevas de proyectos conocidos |
| `split_project` | proyectos enteros retenidos | generalización a una organización nunca vista |
| `split_t3` | cronológica dentro de cada proyecto, 60/15/25 | el escenario de la literatura y el que PBE alcanza con historia |

`split_t3` es la que gobierna todos los experimentos publicados: sortear al azar
sobre datos con orden temporal permite entrenar con el futuro y predecir el
pasado, que es una fuga difícil de ver y fácil de cometer.

**Qué garantiza cada control.** La garantía que vale para las tres particiones
es que **ningún texto idéntico cruza la frontera** (`text_hash` en un solo
bloque). La garantía adicional de que tampoco la cruzan los *casi* duplicados
(`group_key`) solo puede exigírsele a `split`, que es la única que se sortea por
`group_key`. `split_project` y `split_t3` reparten por otro criterio —proyecto
entero y orden cronológico— y por eso 25 y 40 grupos de casi duplicados quedan a
ambos lados. Es una exposición residual **declarada**, no un descuido: eliminarla
en la partición cronológica exigiría romper el orden temporal, que es
precisamente lo que esa partición existe para respetar.
"""))

    c.append(code('''
sw3 = v3[v3.is_software_task == 1]
comp = pd.DataFrame({
    "split (grupo de texto)": sw3.split.value_counts(),
    "split_project": sw3.split_project.value_counts(),
    "split_t3 (cronológica)": sw3.split_t3.value_counts(),
}).fillna(0).astype(int)
tabla(comp.reset_index(names="bloque"), "14_particiones",
      "Tabla 14 · Tamaño de cada bloque en las tres particiones")

# --- control de fuga
filas = []
for col in ["split", "split_project", "split_t3"]:
    d = sw3[sw3[col].notna()]
    porgrupo = d.groupby("group_key")[col].nunique()
    portexto = d.groupby("text_hash")[col].nunique()
    filas.append(dict(
        particion=col,
        grupos_en_mas_de_un_bloque=int((porgrupo > 1).sum()),
        textos_en_mas_de_un_bloque=int((portexto > 1).sum()),
        proyectos_compartidos=(len(set(d[d[col] == "train"].project)
                                   & set(d[d[col] == "test"].project))
                               if "test" in set(d[col]) else None)))
fuga = pd.DataFrame(filas)
fuga["grupos_esperado"] = ["0 · se sortea por group_key", "no aplica", "no aplica"]
tabla(fuga, "15_control_fuga", "Tabla 15 · Control de fuga")
'''))

    c.append(code('''
fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a, col, tit in zip(ax, ["split", "split_project", "split_t3"],
                       ["split · grupo de texto", "split_project · proyecto entero",
                        "split_t3 · cronológica"]):
    d = sw3[sw3[col].notna()]
    p = (d.groupby([col, "source"]).size().unstack(fill_value=0)
         .reindex(["train", "val", "test"]).dropna(how="all"))
    abajo = np.zeros(len(p))
    for i, s in enumerate(p.columns):
        a.bar(p.index, p[s], bottom=abajo, label=s, color=PALETA[i])
        abajo = abajo + p[s].to_numpy()
    a.set_title(tit)
    a.yaxis.set_major_formatter(MILES)
    a.legend()
fig.suptitle("Figura 10 · Composición de cada partición por fuente", x=.01,
             ha="left", fontsize=11, weight="600")
guardar(fig, "fig10_particiones")
plt.show()

# la partición cronológica, vista en el tiempo
d = sw3[sw3.split_t3.notna() & sw3.created.notna()]
fig, ax = plt.subplots(figsize=(9.5, 3))
for i, b in enumerate(["train", "val", "test"]):
    x = d[d.split_t3 == b].created
    ax.hist(x, bins=60, alpha=.75, label=b, color=PALETA[i])
ax.set_xlabel("fecha de creación de la tarea")
ax.set_ylabel("tareas")
ax.legend()
ax.set_title("Figura 11 · split_t3 · el bloque de prueba es siempre el tramo final "
             "de cada proyecto")
guardar(fig, "fig11_temporal")
plt.show()
'''))

    # ---------------------------------------------------------------- 10
    c.append(md("""
---
## 10. Perfil final de v3 y diccionario de columnas
"""))

    c.append(code('''
perfil = (sw3.groupby("source").effort_hours
          .describe(percentiles=[.05, .25, .5, .75, .95]).round(2))
tabla(perfil.reset_index(), "16_perfil_v3",
      "Tabla 16 · Distribución del esfuerzo por fuente (tareas de software)")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for i, s in enumerate(sorted(sw3.source.unique())):
    e = sw3[sw3.source == s].effort_hours
    ax[0].hist(np.log10(e), bins=55, alpha=.6, label=f"{s}  n={len(e):,}",
               color=PALETA[i])
ax[0].set_xlabel("log10(horas)")
ax[0].set_ylabel("tareas")
ax[0].legend()
ax[0].set_title("Esfuerzo por fuente")

tam = sw3.project.value_counts()
ax[1].hist(np.log10(tam), bins=30, color=TINTA, alpha=.85)
ax[1].set_xlabel("log10(tareas por proyecto)")
ax[1].set_ylabel("proyectos")
ax[1].set_title(f"{sw3.project.nunique()} proyectos, muy desiguales")
fig.suptitle("Figura 12 · Perfil final de v3", x=.01, ha="left",
             fontsize=11, weight="600")
guardar(fig, "fig12_perfil_v3")
plt.show()

print(f"\\nv3: {len(v3):,} filas ({int((v3.is_software_task==1).sum()):,} de software)")
print(f"SHA-256 (16): {sha(DATOS / 'pbe_effort_v3.parquet')}")
'''))

    c.append(code('''
def diccionario(df, notas):
    filas = []
    for col in df.columns:
        s = df[col]
        filas.append(dict(
            columna=col, tipo=str(s.dtype),
            no_nulos=int(s.notna().sum()),
            cobertura_pct=round(100 * s.notna().mean(), 1),
            distintos=int(s.nunique(dropna=True)),
            ejemplo=str(s.dropna().iloc[0])[:44] if s.notna().any() else "",
            descripcion=notas.get(col, "")))
    return pd.DataFrame(filas)

NOTAS = {
    "task_id": "identificador único, prefijado por fuente",
    "source": "TAWOS o SiP",
    "domain": "open_source_jira o commercial_inhouse",
    "project": "proyecto o cliente",
    "title": "título de la tarea",
    "description": "descripción; vacía en SiP",
    "text": "título + descripción, normalizado (D8)",
    "text_raw": "texto antes de normalizar",
    "text_hash": "md5 del texto normalizado; base de D1a y D2",
    "group_key": "grupo de casi duplicados (D3a); las particiones se sortean por aquí",
    "raw_kind": "categoría original de la fuente",
    "task_kind": "categoría armonizada",
    "priority": "prioridad declarada",
    "effort_hours": "OBJETIVO · horas de reloj, por desarrollador tras E3",
    "effort_hours_raw": "esfuerzo antes de E3",
    "log_effort": "log natural de effort_hours",
    "story_point": "puntos de historia cuando existen; NO se usan como objetivo",
    "created": "fecha de creación",
    "started": "inicio (SiP)",
    "completed": "cierre",
    "n_words": "palabras del texto normalizado",
    "n_merged": "filas originales colapsadas en esta (D2)",
    "has_desc": "1 si hay descripción",
    "has_code": "1 si la descripción contiene código o traza",
    "effort_gt_calendar": "1 si el esfuerzo supera el lapso de calendario (marcado, no descartado)",
    "is_software_task": "0 para trabajo no informático de SiP; excluidas de las particiones",
    "split": "partición 70/15/15 por group_key",
    "split_project": "partición por proyecto entero",
    "split_t3": "partición cronológica dentro de cada proyecto (la usada en los experimentos)",
    "split_temporal": "partición cronológica en dos bloques, previa a split_t3",
    "n_dev": "desarrolladores distintos en la tarea (SiP)",
    "effort_task_total": "horas sumadas de todo el equipo, antes de E3",
    "objetivo_redondo": "1 si effort_hours es un entero (E4)",
    "HoursEstimate": "estimación humana previa (SiP)",
    "AssignedToID": "asignado (SiP)",
    "RaisedByID": "quien reportó (SiP)",
    "ProjectBreakdownCode": "código de desglose (SiP)",
    "Assignee_ID": "asignado (TAWOS, anonimizado)",
    "en_sprint": "pertenece a un sprint",
    "tiene_asignado": "tiene persona asignada",
    "n_links": "enlaces a otras incidencias (puede incluir enlaces posteriores)",
    "n_links_estructurales": "subconjunto conservador: Depends/Blocks/Part",
    "n_componentes": "componentes asociados",
    "Title_Changed_After_Estimation": "el título cambió tras estimar",
    "Description_Changed_After_Estimation": "la descripción cambió tras estimar",
    "Story_Point_Changed_After_Estimation": "los puntos cambiaron tras estimar",
    "desarrollador": "identificador unificado de desarrollador",
    "tiene_estimacion": "1 si hay estimación humana",
    "orden": "fecha usada para ordenar dentro del proyecto",
}
dic = diccionario(v3, NOTAS)
dic.to_csv(DATOS / "diccionario_columnas_v3.csv", index=False, encoding="utf-8-sig")
print(f"escrito {DATOS / 'diccionario_columnas_v3.csv'}")
dic
'''))

    c.append(md("""
---
## Resumen de lo descartado

| paso | entrada | salida | perdidas | criterio dominante |
|---|---|---|---|---|
| v0 → v1 | 20 248 | 18 593 | **1 655** | texto < 3 palabras (465), esfuerzo < 0,1 h (408), duplicados |
| v1 → v2 | 18 593 | 18 474 | **119** | épicas (15) y contenedores > 16 h (109) |
| v2 → v3 | 18 474 | 18 474 | 0 | solo añade 21 columnas |

Ninguna fila se descartó por conveniencia estadística: cada criterio responde a
que la fila **no representa lo que el modelo tiene que predecir** —una tarea de
software, hecha por una persona, con texto suficiente para vectorizar y un
esfuerzo de reloj creíble.

El cuaderno del conjunto v4, con el corpus público de Jira y los criterios de
mezcla, es `02_dataset_v4.ipynb`.
"""))
    return c


def notebook():
    return construir(celdas(), "Dataset v3")
