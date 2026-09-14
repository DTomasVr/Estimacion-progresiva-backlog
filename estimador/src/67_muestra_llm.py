"""Fixed benchmark sample for the model-versus-LLM comparison.

Everything a competitor needs is frozen here, once, before any LLM is called:
the tasks, the text each LLM will read, the few-shot examples, the per-project
history, and our own model's predictions on the same rows. No LLM result can
then influence what is being compared.

Three conditions, matched on information rather than on method:
  A  cold start       the task text and its category
  B  few-shot         + 10 fixed solved tasks from the training block
  C  project history  + the last k tasks of THAT project, with their hours
Our model competes in A/B with the absolute regressor and in C with the ratio
model scaled by the median of exactly the same k tasks the LLM is shown, so both
sides see the same history.

Contamination guards: issue keys are masked, and project names are never shown.
Order inside a project uses the creation date (start date for SiP rows without
one) as a proxy for completion order, the same proxy as the deployment curve.
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

DATA = "data/processed/agix_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
OUT = Path("data/processed/llm_bench")
NUM = ["n_links", "n_links_estructurales", "n_componentes", "tiene_asignado"]
CAT = ["task_kind", "source"]
SEED, K_HIST, MAX_CHARS = 20260914, 10, 1200
CORTES = [0, 1, 4, 16, np.inf]
CLAVE = re.compile(r"\b[A-Z][A-Z0-9]{1,14}-\d+\b")
GRUPOS_PRUEBA = {("prueba_d1", "TAWOS"): 150, ("prueba_d1", "SiP"): 150,
                 ("externo", "JiraPublico"): 300}
GRUPOS_CAL = {("val", "TAWOS"): 40, ("val", "SiP"): 40, ("val", "JiraPublico"): 40}


def texto_llm(t):
    t = CLAVE.sub("[ISSUE]", str(t or ""))
    t = re.sub(r"\s+", " ", t).strip()
    return t[:MAX_CHARS] + (" …" if len(t) > MAX_CHARS else "")


def estratificada(d, n, rng):
    """Proportional over effort buckets, so unweighted metrics stay unbiased."""
    d = d.assign(tramo=pd.cut(d.effort_hours, CORTES, labels=False))
    cuota = (d.tramo.value_counts(normalize=True) * n).round().astype(int)
    partes = []
    for tr, q in cuota.items():
        pool = d[d.tramo == tr]
        q = min(max(q, min(5, len(pool))), len(pool))
        partes.append(pool.iloc[rng.permutation(len(pool))[:q]])
    return pd.concat(partes).drop(columns="tramo")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    emb = np.load(f"{EMB}.npy")
    ids = np.load(f"{EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fe"] = [pos[t] for t in sw.task_id]
    sw["orden_llm"] = (sw.created.fillna(sw.started).fillna(sw.completed)
                       .fillna(pd.Timestamp("2000-01-01")))
    sw["bloque"] = np.select(
        [(sw.source != "JiraPublico") & (sw.split_t3 == "test"),
         sw.split_t3 == "externo", sw.split_t3 == "val"],
        ["prueba_d1", "externo", "val"], "train")

    # ---------------------------------------------------------- muestras
    def sacar(grupos):
        return pd.concat([estratificada(sw[(sw.bloque == b) & (sw.source == s)],
                                        n, rng) for (b, s), n in grupos.items()])
    muestra = sacar(GRUPOS_PRUEBA)
    cal = sacar(GRUPOS_CAL)
    assert not set(muestra.task_id) & set(cal.task_id)

    # ---------------------------------------------------------- historial
    orden = sw.sort_values(["project", "orden_llm", "task_id"])
    por_proy = {p: d for p, d in orden.groupby("project", sort=False)}

    def historial(r):
        d = por_proy[r.project]
        if r.bloque == "externo":               # causal dentro del bloque externo
            d = d[d.bloque == "externo"]
        previas = d[(d.orden_llm < r.orden_llm)
                    | ((d.orden_llm == r.orden_llm) & (d.task_id < r.task_id))]
        previas = previas.tail(K_HIST)
        return [{"texto": texto_llm(t)[:300], "horas": round(float(h), 2)}
                for t, h in zip(previas.text, previas.effort_hours)]

    for d in (muestra, cal):
        d["texto_llm"] = [texto_llm(t) for t in d.text]
        d["hist"] = [historial(r) for r in d.itertuples()]
        d["n_hist"] = d["hist"].str.len()

    # ---------------------------------------------------------- pocos ejemplos
    tr = sw[sw.bloque == "train"]
    tr_corto = tr[tr.text.str.len().between(40, 400)]
    ejemplos = []
    for q in np.linspace(.05, .95, 10):
        objetivo = tr_corto.effort_hours.quantile(q)
        cand = tr_corto.iloc[(tr_corto.effort_hours - objetivo).abs().argsort()[:50]]
        r = cand.iloc[rng.integers(len(cand))]
        ejemplos.append({"texto": texto_llm(r.text)[:300], "tipo": r.task_kind,
                         "horas": round(float(r.effort_hours), 2)})
    json.dump(ejemplos, open(OUT / "pocos_ejemplos.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---------------------------------------------------------- nuestro modelo
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                       min_frequency=25).fit(tr[CAT].astype(str))
    sc = StandardScaler().fit(np.log1p(tr[NUM].to_numpy(float)))
    completo = lambda d: np.hstack([emb[d.fe.to_numpy()],
                                    oh.transform(d[CAT].astype(str)),
                                    sc.transform(np.log1p(d[NUM].to_numpy(float)))])
    solo_texto = lambda d: emb[d.fe.to_numpy()]
    hgb = lambda: HistGradientBoostingRegressor(max_iter=300, max_depth=6,
                                                learning_rate=.06,
                                                l2_regularization=1., random_state=0)
    y = np.log(tr.effort_hours.to_numpy())
    m_abs = hgb().fit(completo(tr), y)
    m_txt = hgb().fit(solo_texto(tr), y)
    med = tr.groupby("project").effort_hours.median()
    glob = float(tr.effort_hours.median())
    m_raz = hgb().fit(completo(tr),
                      y - np.log(tr.project.map(med).fillna(glob).to_numpy()))

    for d in (muestra, cal):
        d["modelo_abs"] = np.exp(m_abs.predict(completo(d)))
        d["modelo_solo_texto"] = np.exp(m_txt.predict(solo_texto(d)))
        d["mediana_global"] = glob
        med_hist = np.array([np.median([e["horas"] for e in h]) if len(h) >= 3
                             else np.nan for h in d["hist"]])
        d["mediana_historial"] = med_hist
        d["modelo_razon_C"] = np.exp(m_raz.predict(completo(d))) * med_hist

    cols = ["task_id", "bloque", "source", "project", "task_kind", "texto_llm",
            "hist", "n_hist", "effort_hours", "modelo_abs", "modelo_solo_texto",
            "modelo_razon_C", "mediana_historial", "mediana_global"]
    for nombre, d in (("muestra", muestra), ("calibracion", cal)):
        d = d[cols].copy()
        d["hist"] = d["hist"].map(lambda h: json.dumps(h, ensure_ascii=False))
        d.to_parquet(OUT / f"{nombre}.parquet", index=False)

    print(f"muestra de prueba: {len(muestra)} tareas")
    print(muestra.groupby(["bloque", "source"]).size().to_string())
    print("\ntramos de esfuerzo:")
    print(pd.cut(muestra.effort_hours, CORTES).value_counts().sort_index().to_string())
    print(f"\ncon historial >= 3 (condicion C): {int((muestra.n_hist >= 3).sum())}")
    print(f"calibracion: {len(cal)} tareas   pocos ejemplos: {len(ejemplos)}")
    print(f"claves de incidencia enmascaradas: "
          f"{int(muestra.texto_llm.str.contains(r'\[ISSUE\]').sum())} textos")
    print(f"\nescrito en {OUT}/")


if __name__ == "__main__":
    main()
