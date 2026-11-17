"""v4 = v3 plus the Public Jira Dataset, merged without disturbing the protocol.

The one decision that governs everything here: **the v3 test block is left
byte-identical**. New material can only enter training, validation, or a new
external block. That keeps the pre-registered single-test-read result comparable
and turns "does more data help?" into a clean A/B on a fixed test set, instead
of a comparison across two different test sets that would prove nothing.

Filters applied to the incoming rows, each one measured and reported:
  D1  timespent outside [0.05, 200] h -- 6 s and 208,000 h are not tasks.
  D2  container issue types (Epic, New Project, ...).
  D3  rollup rows: aggregatetimespent > timespent AND subtasks > 0, i.e. the
      logged time belongs to the children, not to this issue.
  D4  no usable text.
  D5  issue keys we already hold through TAWOS (both mine Apache, Atlassian,
      Sonatype, Hyperledger and Spring, so this overlap is real: 5,100 rows).
  D6  text already present in the v3 validation or test blocks -- a pure
      leakage guard, since an added training row repeating a test text would
      inflate the very comparison we are running.

Rows keeping a human estimate are flagged `estimacion_igual_a_real` when
timespent equals timeoriginalestimate exactly (29% of them). Jira's "adjust
remaining estimate automatically" produces that equality mechanically, so those
rows cannot be used as evidence that a human estimated well.
"""
import glob
import hashlib
import re

import numpy as np
import pandas as pd

pd.set_option("display.width", 240)
V3 = "data/processed/pbe_effort_v3.parquet"
OUT = "data/processed/pbe_effort_v4.parquet"
SEED = 20260910
CONTENEDOR = {"Epic", "New Project", "Initiative", "Program", "Umbrella"}
MIN_H, MAX_H = 0.05, 200.0

TIPO_KIND = {
    "Bug": "bug", "Defect": "bug",
    "Improvement": "improvement", "Enhancement": "improvement",
    "Suggestion": "improvement", "Enhancement Request": "improvement",
    "New Feature": "feature", "Story": "feature", "Feature Request": "feature",
    "Wish": "feature",
    "Task": "task", "Sub-task": "task", "Technical task": "task",
    "Sub-Task": "task", "Dependency upgrade": "task", "Refactoring": "task",
    "Epic": "epic",
    "Documentation": "docs", "Publishing Support": "docs",
    "Test": "test", "Quality Risk": "test",
    "Support Request": "support", "Support Patch": "support",
}
RE_CODE = re.compile(r"\{code|\{noformat|```|\bat [\w.$]+\([\w.]+:\d+\)")


def norm(s):
    return (s.fillna("").astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def cargar_crudo():
    fr = []
    for p in sorted(glob.glob("data/interim/jira_publico/*.parquet")):
        d = pd.read_parquet(p)
        d["fuente_jira"] = p.replace("\\", "/").split("/")[-1][:-8]
        fr.append(d)
    return pd.concat(fr, ignore_index=True)


def limpiar(df, v3):
    n0 = len(df)
    reg = []

    def paso(mask, etiqueta):
        nonlocal df
        antes = len(df)
        df = df[mask].copy()
        reg.append((etiqueta, antes - len(df), len(df)))

    df["horas"] = df.timespent / 3600.0
    paso((df.horas >= MIN_H) & (df.horas <= MAX_H),
         "D1 esfuerzo fuera de [{}, {}] h".format(MIN_H, MAX_H))
    paso(~df.issuetype.isin(CONTENEDOR), "D2 tipo contenedor")
    roll = ((df.agg_timespent.fillna(0) > df.timespent)
            & (df.n_subtareas.fillna(0) > 0))
    paso(~roll, "D3 agregacion de subtareas")

    df["title"] = df.summary.fillna("").astype(str).str.strip()
    df["description"] = df.description.fillna("").astype(str).str.strip()
    df["text"] = (df.title + ". " + df.description).str.strip()
    df["text"] = df.text.str.replace(r"^\. ", "", regex=True).str.strip(". ")
    df["text_norm"] = norm(df.text)
    paso(df.text_norm.str.len() >= 3, "D4 sin texto utilizable")

    claves = set(v3[v3.source == "TAWOS"].task_id
                 .str.replace("TAWOS-", "", regex=False))
    paso(~df.key.isin(claves), "D5 issue ya presente via TAWOS")

    df["text_hash"] = [hashlib.md5(s.encode()).hexdigest()[:16]
                       for s in df.text_norm]
    reservado = set(v3[(v3.is_software_task == 1)
                       & v3.split_t3.isin(["val", "test"])].text_hash)
    paso(~df.text_hash.isin(reservado), "D6 texto presente en val/test de v3")

    print("filas de entrada: {:,}".format(n0))
    for etiqueta, quitadas, quedan in reg:
        print("  {:44s} -{:>6,}   quedan {:>7,}".format(etiqueta, quitadas, quedan))
    print("  {:44s}  {:>6,}   ({:.1f}% de lo extraido)".format(
        "sobreviven", len(df), 100 * len(df) / n0))
    return df


def a_esquema(df):
    def fecha(s):
        return pd.to_datetime(s, format="ISO8601", errors="coerce",
                              utc=True).dt.tz_localize(None)

    creado = fecha(df.created)
    resuelto = fecha(df.resolved)
    cal = (resuelto - creado).dt.total_seconds() / 3600
    est_h = df.estimate_orig / 3600.0
    out = pd.DataFrame({
        "task_id": "JP-" + df.fuente_jira + "-" + df.key,
        "source": "JiraPublico",
        "domain": "open_source_jira",
        "project": df.fuente_jira + ":" + df.project_key.fillna("?"),
        "title": df.title,
        "description": df.description,
        "text": df.text,
        "text_raw": df.text,
        "text_hash": df.text_hash,
        "raw_kind": df.issuetype.fillna("(none)"),
        "task_kind": df.issuetype.map(TIPO_KIND).fillna("other"),
        "priority": df.priority.fillna("(none)"),
        "effort_hours": df.horas,
        "effort_hours_raw": df.horas,
        "log_effort": np.log(df.horas),
        "story_point": np.nan,
        "created": creado,
        "started": pd.NaT,
        "completed": resuelto,
        "n_words": df.text.str.split().str.len().fillna(0).astype(int),
        "n_merged": 1,
        "has_desc": (df.description.str.len() > 0).astype(int),
        "has_code": df.description.str.contains(RE_CODE).astype(int),
        "effort_gt_calendar": (cal.notna() & (df.horas > cal)).astype(int),
        "is_software_task": 1,
        "n_dev": 1,
        "effort_task_total": df.horas,
        "objetivo_redondo": (df.timespent % 3600 == 0).astype(int),
        "HoursEstimate": est_h,
        "AssignedToID": np.nan,
        "RaisedByID": np.nan,
        "ProjectBreakdownCode": pd.NA,
        "Assignee_ID": pd.NA,
        "en_sprint": np.nan,
        "tiene_asignado": df.assignee.notna().astype(float),
        "n_links": df.n_enlaces.fillna(0).astype(float),
        "n_links_estructurales": df.n_subtareas.fillna(0).astype(float),
        "n_componentes": df.n_componentes.fillna(0).astype(float),
        "Title_Changed_After_Estimation": pd.NA,
        "Description_Changed_After_Estimation": pd.NA,
        "Story_Point_Changed_After_Estimation": pd.NA,
        "desarrollador": np.where(df.assignee.notna(),
                                  df.fuente_jira + ":" + df.assignee.astype(str),
                                  None),
        "tiene_estimacion": df.estimate_orig.notna().astype(int),
        "estimacion_igual_a_real": (df.estimate_orig == df.timespent)
                                   .fillna(False).astype(int),
        "n_cambios": df.n_cambios.fillna(0).astype(float),
        "fuente_jira": df.fuente_jira,
    })
    return out


def repartir(nuevo, rng):
    """Whole projects held out as an external block; the rest goes to train.

    Holding out projects rather than rows is what makes the external block a
    real cold-start test: the model has never seen the organisation, which is
    exactly PBE's situation on a new customer.
    """
    tam = nuevo.groupby("project").size().sort_values()
    cand = tam[tam <= len(nuevo) * 0.10].index.tolist()
    rng.shuffle(cand)
    fuera, total = [], 0
    for p in cand:
        if total >= len(nuevo) * 0.15:
            break
        fuera.append(p)
        total += int(tam[p])
    s = pd.Series("train", index=nuevo.index)
    s[nuevo.project.isin(fuera)] = "externo"
    tr = s == "train"
    grupos = np.asarray(nuevo.loc[tr, "text_hash"].unique(), dtype=object)
    rng.shuffle(grupos)
    assert len(set(grupos)) == len(grupos), "el barajado duplico grupos"
    val = set(grupos[:int(len(grupos) * 0.10)])
    s[tr & nuevo.text_hash.isin(val)] = "val"
    return s, fuera, total


def main():
    v3 = pd.read_parquet(V3)
    crudo = cargar_crudo()
    print("=" * 78)
    print("LIMPIEZA DEL BLOQUE PUBLIC JIRA")
    print("=" * 78)
    limpio = limpiar(crudo, v3)
    nuevo = a_esquema(limpio)

    rng = np.random.default_rng(SEED)
    split, fuera, n_fuera = repartir(nuevo, rng)
    nuevo["split_t3"] = split.to_numpy()
    print("\n--- reparto del bloque nuevo ---")
    print("  {}".format(nuevo.split_t3.value_counts().to_dict()))
    print("  proyectos externos retenidos: {} ({:,} filas)".format(
        len(fuera), n_fuera))
    print("   " + ", ".join(fuera[:12]) + (" ..." if len(fuera) > 12 else ""))

    v3 = v3.copy()
    v3["estimacion_igual_a_real"] = 0
    v3["n_cambios"] = np.nan
    v3["fuente_jira"] = pd.NA
    faltan = [c for c in v3.columns if c not in nuevo.columns]
    for c in faltan:
        nuevo[c] = pd.NA
    print("\n  columnas rellenadas en el bloque nuevo: {}".format(faltan))

    v4 = pd.concat([v3[v3.columns], nuevo[v3.columns]], ignore_index=True)
    v4["split_t3"] = v4.split_t3.astype(str)

    print("\n" + "=" * 78)
    print("COMPROBACIONES DE INTEGRIDAD")
    print("=" * 78)
    sw = v4.is_software_task == 1
    te3 = set(v3[(v3.is_software_task == 1) & (v3.split_t3 == "test")].task_id)
    te4 = set(v4[sw & (v4.split_t3 == "test")].task_id)
    print("  prueba de v3 intacta            : {}  ({:,} filas)".format(
        te3 == te4, len(te4)))
    h_te = set(v4[sw & (v4.split_t3 == "test")].text_hash)
    h_tr = set(v4[sw & (v4.split_t3 == "train")].text_hash)
    h_va = set(v4[sw & (v4.split_t3 == "val")].text_hash)
    print("  textos compartidos train/test   : {}  (debe ser 0)".format(
        len(h_te & h_tr)))
    print("  textos compartidos val/test     : {}  (debe ser 0)".format(
        len(h_te & h_va)))
    print("  ids duplicados                  : {}".format(
        int(v4.task_id.duplicated().sum())))
    ext = set(v4[v4.split_t3 == "externo"].project)
    print("  proyectos externos vistos en entrenamiento: {}".format(
        len(ext & set(v4[v4.split_t3 == "train"].project))))

    print("\n--- composicion de v4 (tareas de software) ---")
    t = v4[sw].groupby(["source", "split_t3"]).size().unstack(fill_value=0)
    t["total"] = t.sum(axis=1)
    print(t.to_string())
    print("\n--- esfuerzo por fuente ---")
    print(v4[sw].groupby("source").effort_hours
          .describe(percentiles=[.25, .5, .75, .95]).round(2).to_string())

    v4.to_parquet(OUT, compression="zstd", index=False)
    print("\nescrito {}  ({:,} filas, {} columnas)".format(
        OUT, len(v4), v4.shape[1]))
    n_tr = int((sw & (v4.split_t3 == "train")).sum())
    n_v3 = int((sw & (v4.split_t3 == "train") & (v4.source != "JiraPublico")).sum())
    print("  entrenamiento disponible: {:,} filas (v3 solo: {:,})".format(n_tr, n_v3))


if __name__ == "__main__":
    main()
