"""One place with every headline number and where it came from.

The figures are spread across 26 CSVs and several logs, and some moved while
the work advanced (the pipeline median went 4.54 -> 4.74 s when the bench grew
from 28 to 64 projects). Slides that quote different files drift apart, so this
rebuilds the whole set from the artefacts on disk and records, for each number,
the file and script that produced it.

out: results/REGISTRO.csv  +  results/REGISTRO.md
"""
import json
import os

import numpy as np
import pandas as pd

BASE = r"F:\Software Task generation Model"
R = []


def add(bloque, nombre, valor, fuente, nota=""):
    R.append(dict(bloque=bloque, nombre=nombre, valor=valor, fuente=fuente, nota=nota))


def col(r, *nombres):
    """the CSVs were written by different scripts: cobertura/sCob, precision/sPrec..."""
    for n in nombres:
        if n in r.index and pd.notna(r[n]):
            return r[n]
    return ""


def leer(p):
    return pd.read_csv(f"{BASE}/results/{p}") if os.path.exists(f"{BASE}/results/{p}") else None


# ------------------------------------------------------------------- datos
N = pd.read_parquet(f"{BASE}/data/interim/nodes.parquet", columns=["uid"])
E = pd.read_parquet(f"{BASE}/data/interim/edges.parquet")
S = pd.read_parquet(f"{BASE}/data/processed/split_projects.parquet")
add("datos", "incidencias unificadas", len(N), "data/interim/nodes.parquet", "TAWOS + Public Jira, src/03")
add("datos", "aristas padre-hijo", len(E), "data/interim/edges.parquet",
    f"{int((E.relation == 'epic').sum())} epic->hijo, {int((E.relation == 'subtask').sum())} padre->subtarea")
add("datos", "proyectos", S.project_uid.nunique(), "data/processed/split_projects.parquet",
    f"train {int((S.split == 'train').sum())} / val {int((S.split == 'val').sum())} / test {int((S.split == 'test').sum())}")
for sp in ["train", "val", "test"]:
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_hist_{sp}.parquet", columns=["mitad", "estilo"])
    add("datos", f"ejemplos abajo ({sp})", int(((d.mitad == 'tasks') & (d.estilo == 'jira')).sum()),
        f"data/processed/s2s_hist_{sp}.parquet", "historia -> tareas")
o = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_train.parquet", columns=["estilo"])
add("datos", "pares One2One arriba (train)", len(o), "data/processed/s2s_o2o_train.parquet",
    f"{int((o.estilo == 'historia').sum())} en formato historia")
H = pd.read_parquet(f"{BASE}/data/interim/historias_formato.parquet", columns=["uid", "en_titulo"])
add("datos", "incidencias con formato de historia", len(H), "data/interim/historias_formato.parquet",
    f"{int(H.en_titulo.sum())} con el formato en el titulo; src/12")

# ------------------------------------------------------------------ sbert
s7 = leer("07_sbert_test.csv")
if s7 is not None:
    for _, r in s7[s7.relacion == "todas"].iterrows():
        add("sbert", f"nDCG@10 consultas {r.lang}", f"{r.ndcg10_congelado} -> {r.ndcg10_ajustado}",
            "results/07_sbert_test.csv", f"delta {r.delta} IC95 [{r.ic95_lo}, {r.ic95_hi}]")

# ------------------------------------------------------------- generadores
c = leer("23_cobertura.csv")
if c is not None:
    b = c[c.estrategia == "beam (actual)"].iloc[0]
    add("abajo", "un modelo (val)", f"sF1 {b.sF1}, cobertura {col(b, 'cobertura', 'sCob')}",
        "results/23_cobertura.csv", "")
t = leer("11_test_metricas.csv")
if t is not None:
    for _, r in t[t.mitad == "todas"].iterrows():
        add("abajo", f"prueba: {r.modelo}", f"sF1 {r.sF1}, cobertura {col(r, 'cobertura', 'sCob')}",
            "results/11_test_metricas.csv", f"n={r.n}")
add("abajo", "conjunto de dos modelos (prueba)", "sF1 0.4852, cobertura 0.4673",
    "logs/23 + comparacion pareada", "+0.012 sF1 [0.009, 0.015]; +0.029 cobertura [0.026, 0.033]")
e18 = leer("18_test_o2o.csv")
if e18 is not None:
    for _, r in e18.iterrows():
        add("arriba", r.etiqueta, f"sF1 {r.sF1}, cobertura {col(r, 'cobertura', 'sCob')}, "
            f"formato {col(r, 'formato')}",
            "results/18_test_o2o.csv", f"n_gen {r.n_gen}")
q = leer("24_calidad_arriba.csv")
if q is not None:
    for _, r in q.iterrows():
        add("arriba", f"variante {r.variante} (val)", f"sF1 {r.sF1}, fuga {col(r, 'fuga')}",
            "results/24_calidad_arriba.csv", "")

# --------------------------------------------------------------- idiomas
e26 = leer("26_eval_es.csv")
if e26 is not None:
    for _, r in e26.iterrows():
        add("espanol", f"arriba {r.modelo}/{r.idioma}", f"sF1 {r.sF1}", "results/26_eval_es.csv", "")
add("espanol", "entrada es traducida al ingles", "sF1 0.370 -> 0.416", "medicion directa 2026-09-23",
    "+0.045 IC95 [0.027, 0.065]; 175 ms; ida y vuelta, numero optimista")
t27 = leer("27_traduccion_salida.csv")
if t27 is not None:
    for _, r in t27.iterrows():
        add("espanol", f"salida traducida ({r.tipo})",
            f"fidelidad {r.fidelidad} (piso {r.piso}), terminos {r.terminos}",
            "results/27_traduccion_salida.csv", f"{r.ms_por_item} ms/item")

# ------------------------------------------------------------------ lodo
l2 = leer("29_lodo_v2.csv")
if l2 is not None:
    p = l2.drop_duplicates(["dominio", "conjunto"], keep="last").pivot(
        index="dominio", columns="conjunto", values="sF1")
    for d, row in p.iterrows():
        add("dominio", f"dominio {int(d)} fuera", f"fuera {row.fuera} vs dentro {row.dentro}",
            "results/29_lodo_v2.csv", f"diferencia {round(row.fuera - row.dentro, 4)}")

# ----------------------------------------------------------------- banco
B = pd.DataFrame([json.loads(l) for l in open(f"{BASE}/results/20_banco.jsonl", encoding="utf-8")])
comunes = B.groupby("caso").sistema.nunique()
casos = set(comunes[comunes == B.sistema.nunique()].index)
add("banco", "proyectos medidos", len(casos), "results/20_banco.jsonl", "con los 4 sistemas presentes")
for (sis, cond), g in B[B.caso.isin(casos)].groupby(["sistema", "condicion"]):
    add("banco", f"{sis} {cond}".strip(), f"{g.segundos.median():.2f} s (p90 {g.segundos.quantile(.9):.1f})",
        "results/20_banco.jsonl", f"n={len(g)}, tareas {g.n_tareas.mean():.0f}")
P = pd.read_csv(f"{BASE}/results/22_banco_pareado.csv")
add("banco", "fidelidad nuestra (64 proyectos)", f"sF1 {P.sF1_nuestro.iloc[0]:.4f}",
    "results/22_banco_pareado.csv", "media por proyecto, mismos casos que los LLM")
for r in P.itertuples():
    add("banco", f"fidelidad {r.sistema}", f"sF1 {r.sF1_otro:.4f}", "results/22_banco_pareado.csv",
        f"delta {r.delta:+.4f} IC95 [{r.ic95_lo:+.4f}, {r.ic95_hi:+.4f}] pareado n={r.n}; "
        f"{'el LLM gana' if r.gana_otro else 'empate: el IC cruza cero'}")
add("banco", "tokens de salida por backlog (LLM)", int(B.dropna(subset=['tokens']).tokens.median()),
    "results/20_banco.jsonl", "base del costo por API")

# ---------------------------------------------------------------- costos
add("costo", "API tipo Groq por proyecto", "~0.0005 USD", "precios publicos 2026",
    "~1000 tok entrada + ~1400 salida; GPT OSS 20B 0.075/0.30 por millon")
add("costo", "PC propia 24/7", "~58 kWh/mes = ~S/ 40", "medicion nvidia-smi + tarifa OSINERGMIN",
    "GPU 14 W en reposo, 80 W generando")
add("costo", "serverless GPU (L4)", "0.10 centavos caliente / 0.44 frio", "Modal 0.000222 USD/s",
    "carga de modelos 10-17 s es el costo dominante en frio")

D = pd.DataFrame(R)
D.to_csv(f"{BASE}/results/REGISTRO.csv", index=False)
with open(f"{BASE}/results/REGISTRO.md", "w", encoding="utf-8") as fh:
    fh.write("# Registro de resultados\n\nCada cifra con su archivo de origen. "
             "Generado por src/32_registro_resultados.py\n")
    for bloque, g in D.groupby("bloque", sort=False):
        fh.write(f"\n## {bloque}\n\n| medida | valor | fuente | nota |\n|---|---|---|---|\n")
        for _, r in g.iterrows():
            fh.write(f"| {r.nombre} | {r.valor} | `{r.fuente}` | {r.nota} |\n")
print(D.to_string(index=False))
print(f"\n{len(D)} cifras registradas en results/REGISTRO.csv y .md")
