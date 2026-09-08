"""Apply the agreed cleaning decisions. Explicit, ordered, and reported.

  D9  join SiP dates from est-act-dates.csv          (enrich, 0 rows lost)
  D8  normalise text: URLs, emails, hashes, traces   (rewrite, 0 rows lost)
  D6  drop text with < 3 words
  D5  drop effort < 0.1 h
  D4  drop SiP tasks > 200 h (umbrella budget lines)
  D1a drop exact-text groups whose effort spans > 10x; collapse the rest
  D2  collapse remaining exact-text duplicates to one row at the median
  D7  merge task kinds with < 50 rows into `other`
  D3a group near-duplicates (cosine >= 0.95) so they cannot straddle splits
  D10 keep title-only rows (no action)

Order matters: text is normalised BEFORE duplicates are detected, so that two
issues differing only by a stack trace are recognised as the same text.
Original values are preserved in `text_raw` and `effort_hours_raw`.
"""
import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

pd.set_option("display.width", 240)
SEED = 20260830
OUT = "data/processed"

URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
HASH = re.compile(r"\b[0-9a-f]{7,}\b", re.I)
STACK = re.compile(r"(?:^|\n)\s*at\s+[\w$.]+\([^)]*\)", re.M)
NUM = re.compile(r"\b\d{4,}\b")
WS = re.compile(r"\s+")


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def normalise(s):
    s = s.fillna("").astype(str)
    s = s.str.replace(URL, " <URL> ", regex=True)
    s = s.str.replace(EMAIL, " <EMAIL> ", regex=True)
    s = s.str.replace(STACK, " <STACK> ", regex=True)
    s = s.str.replace(r"(?:\s*<STACK>\s*){2,}", " <STACK> ", regex=True)
    s = s.str.replace(HASH, " <HASH> ", regex=True)
    s = s.str.replace(NUM, " <NUM> ", regex=True)
    s = s.str.replace(WS, " ", regex=True).str.strip()
    return s


def log(step, df, before, note=""):
    delta = len(df) - before  # negative when rows are removed
    print(f"  {step:5s} {len(df):7,} filas  ({delta:+,})  {note}")
    return len(df)


def main():
    df = pd.read_parquet(f"{OUT}/agix_effort_dataset.parquet")
    print(f"entrada: {len(df):,} filas ({(df.is_software_task==1).sum():,} de software)\n")
    df["text_raw"] = df.text
    df["effort_hours_raw"] = df.effort_hours
    n = len(df)
    print("--- pasos ---")

    # ---- D9: recover SiP dates -------------------------------------------
    # the date file mirrors the per-developer row structure of the task file;
    # every duplicated TaskNumber group was verified to be byte-identical
    dates = pd.read_csv("data/raw/jones/SiP/est-act-dates.csv").drop_duplicates()
    assert not dates.TaskNumber.duplicated().any(), "SiP dates still ambiguous"
    for c in ["EstimateOn", "StartedOn", "CompletedOn"]:
        dates[c] = pd.to_datetime(dates[c], format="%d-%b-%y", errors="coerce")
    dates["task_id"] = "SIP-" + dates.TaskNumber.astype(str)
    dmap = dates.set_index("task_id")
    df["created"] = df.created.fillna(df.task_id.map(dmap.StartedOn))
    df["started"] = df.task_id.map(dmap.StartedOn)
    df["completed"] = df.task_id.map(dmap.CompletedOn)
    got = df.task_id.isin(dmap.index).sum()
    n = log("D9", df, n, f"fechas SiP recuperadas para {got:,} tareas")

    # ---- D8: normalise text ----------------------------------------------
    df["text"] = normalise(df.text_raw)
    df["title"] = normalise(df.title)
    changed = (df.text != df.text_raw).sum()
    n = log("D8", df, n, f"texto normalizado, {changed:,} filas modificadas")

    df["n_words"] = df.text.str.split().str.len().fillna(0).astype(int)
    df["text_hash"] = [hashlib.md5(s.lower().encode()).hexdigest()[:16] for s in df.text]

    # ---- D6 / D5 / D4: row drops -----------------------------------------
    df = df[df.n_words >= 3].copy()
    n = log("D6", df, n, "texto con menos de 3 palabras")

    df = df[df.effort_hours >= 0.1].copy()
    n = log("D5", df, n, "esfuerzo menor a 0,1 h")

    df = df[~((df.source == "SiP") & (df.effort_hours > 200))].copy()
    n = log("D4", df, n, "tareas paraguas de SiP mayores a 200 h")

    # ---- D1a + D2: exact-text groups -------------------------------------
    g = df.groupby("text_hash").effort_hours.agg(["min", "max", "size"])
    dup = g[g["size"] > 1]
    wide = dup[(dup["max"] / dup["min"]) > 10].index
    df = df[~df.text_hash.isin(wide)].copy()
    n = log("D1a", df, n, f"{len(wide):,} grupos de texto con dispersion > 10x")

    df = df.sort_values(["text_hash", "task_id"])
    grp = df.groupby("text_hash", sort=False)
    med = grp.effort_hours.transform("median")
    cnt = grp.effort_hours.transform("size")
    df["n_merged"] = cnt.astype(int)
    df["effort_hours"] = med
    df = df.drop_duplicates("text_hash", keep="first").copy()
    df["log_effort"] = np.log(df.effort_hours)
    n = log("D2", df, n, "duplicados exactos colapsados a la mediana")

    # ---- D7: merge thin categories ---------------------------------------
    sw_mask = df.is_software_task == 1
    counts = df.loc[sw_mask, "task_kind"].value_counts()
    thin = list(counts[counts < 50].index)
    df["task_kind"] = df.task_kind.where(~df.task_kind.isin(thin), "other")
    n = log("D7", df, n, f"categorias fusionadas en 'other': {thin}")

    # ---- D3a: near-duplicate clusters as the split key -------------------
    sw = df[df.is_software_task == 1]
    vec = TfidfVectorizer(min_df=2, max_features=60000, ngram_range=(1, 2),
                          sublinear_tf=True, strip_accents="unicode")
    X = vec.fit_transform(sw.text)
    nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    uf = UF(len(sw))
    pairs = 0
    for i in range(len(sw)):
        for j in range(1, dist.shape[1]):
            if 1 - dist[i, j] >= 0.95:
                uf.union(i, idx[i, j])
                pairs += 1
    cluster = np.array([uf.find(i) for i in range(len(sw))])
    df["group_key"] = df.text_hash
    df.loc[sw.index, "group_key"] = ["nd%d" % c for c in cluster]
    merged = len(sw) - len(set(cluster))
    n = log("D3a", df, n, f"{pairs:,} pares >= 0.95, {merged:,} filas agrupadas con un vecino")

    # ---- re-split on the new grouping key ---------------------------------
    rng = np.random.default_rng(SEED)
    sw_mask = df.is_software_task == 1
    keys = np.asarray(df.loc[sw_mask, "group_key"].unique(), dtype=object)
    rng.shuffle(keys)
    assert len(set(keys)) == len(keys)
    n_tr, n_va = int(len(keys) * .70), int(len(keys) * .85)
    lut = {k: ("train" if i < n_tr else "val" if i < n_va else "test")
           for i, k in enumerate(keys)}
    df["split"] = pd.NA
    df.loc[sw_mask, "split"] = df.loc[sw_mask, "group_key"].map(lut)

    sizes = df[sw_mask].groupby(["source", "project"]).size().sort_values()
    cand = [k for k in sizes.index if sizes[k] <= sw_mask.sum() * 0.12]
    rng.shuffle(cand)
    held, tot = [], 0
    for k in cand:
        if tot >= sw_mask.sum() * .15:
            break
        held.append(k)
        tot += sizes[k]
    idx2 = df.set_index(["source", "project"]).index
    df["split_project"] = pd.NA
    df.loc[sw_mask, "split_project"] = np.where(
        idx2.isin(set(held)), "test", "train")[sw_mask.to_numpy()]

    # ---- report -----------------------------------------------------------
    sw = df[df.is_software_task == 1]
    print(f"\n--- resultado ---")
    print(f"  filas totales      : {len(df):,}")
    print(f"  tareas de software : {len(sw):,}")
    print(sw.groupby("source").agg(
        n=("effort_hours", "size"), mediana=("effort_hours", "median"),
        media=("effort_hours", "mean")).round(2).to_string())
    print(f"\n  particion split       : {sw.split.value_counts().to_dict()}")
    print(f"  particion split_project: {sw.split_project.value_counts().to_dict()}")

    chk = sw.groupby("group_key").split.nunique()
    print(f"\n  fuga: grupos en mas de un split = {int((chk>1).sum())} (debe ser 0)")
    h = sw.groupby("text_hash").split.nunique()
    print(f"  fuga: texto exacto en mas de un split = {int((h>1).sum())} (debe ser 0)")
    print(f"  duplicados de texto restantes = {int(sw.text_hash.duplicated().sum())}")

    print(f"\n  categorias:\n{sw.task_kind.value_counts().to_string()}")

    cols = ["task_id", "source", "domain", "project", "title", "description",
            "text", "text_raw", "text_hash", "group_key", "raw_kind",
            "task_kind", "priority", "effort_hours", "effort_hours_raw",
            "log_effort", "story_point", "created", "started", "completed",
            "n_words", "n_merged", "has_desc", "has_code",
            "effort_gt_calendar", "is_software_task", "split", "split_project"]
    df[cols].to_parquet(f"{OUT}/agix_effort_clean.parquet",
                        compression="zstd", index=False)
    df[cols].to_csv(f"{OUT}/agix_effort_clean.csv", index=False)
    print(f"\nescrito {OUT}/agix_effort_clean.parquet y .csv")


if __name__ == "__main__":
    main()
