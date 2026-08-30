"""Read-only data quality diagnostic. Reports problems, changes nothing.

Every finding is printed with the exact row count and a suggested action, so
the cleaning decisions can be made deliberately rather than by default.
Nothing here writes to the dataset.
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

pd.set_option("display.width", 240)
PATH = "data/processed/agix_effort_dataset.parquet"


def head(n, title):
    print(f"\n{'='*78}\n{n}. {title}\n{'='*78}")


def main():
    df = pd.read_parquet(PATH)
    sw = df[df.is_software_task == 1].copy()
    print(f"rows: {len(df):,}   software tasks: {len(sw):,}")

    # ------------------------------------------------------------------
    head(1, "MISSING VALUES per column")
    miss = []
    for c in df.columns:
        s = df[c]
        nulls = s.isna().sum()
        empty = (s.astype(str).str.strip() == "").sum() if s.dtype == object or str(s.dtype).startswith("str") else 0
        if nulls or empty:
            miss.append((c, nulls, round(nulls / len(df) * 100, 2), empty))
    print(pd.DataFrame(miss, columns=["columna", "nulos", "%", "vacios"])
          .to_string(index=False) if miss else "  ninguna")

    # ------------------------------------------------------------------
    head(2, "DUPLICATES")
    print(f"  filas identicas en todas las columnas : {df.duplicated().sum():,}")
    print(f"  task_id repetido                      : {df.task_id.duplicated().sum():,}")
    for col in ["text_hash", "title"]:
        d = sw[col].duplicated(keep=False)
        print(f"  [software] {col:10s} repetido           : {d.sum():,} filas "
              f"({d.mean()*100:.1f}%) en {sw[d][col].nunique():,} grupos")

    # exact text duplicates that also agree on effort -> pure redundancy
    g = sw[sw.text_hash.duplicated(keep=False)].groupby("text_hash")
    same_eff = g.effort_hours.nunique() == 1
    print(f"\n  grupos de texto duplicado           : {len(same_eff):,}")
    print(f"    con el MISMO esfuerzo (redundante): {same_eff.sum():,}")
    print(f"    con esfuerzo DISTINTO (ruido)     : {(~same_eff).sum():,}")

    # ------------------------------------------------------------------
    head(3, "CONFLICTING LABELS -- same text, different hours")
    conf = g.agg(n=("effort_hours", "size"),
                 lo=("effort_hours", "min"),
                 hi=("effort_hours", "max"),
                 med=("effort_hours", "median"))
    conf = conf[conf.hi > conf.lo].copy()
    conf["ratio"] = (conf.hi / conf.lo).round(1)
    print(f"  grupos con etiquetas en conflicto: {len(conf):,} "
          f"(afectan a {conf.n.sum():,} filas)")
    if len(conf):
        print(f"  razon max/min  mediana {conf.ratio.median():.1f}x, "
              f"p90 {conf.ratio.quantile(.9):.1f}x, max {conf.ratio.max():.1f}x")
        print(f"  grupos con razon > 10x: {(conf.ratio > 10).sum():,}")
        worst = conf.sort_values("ratio", ascending=False).head(5)
        print("\n  peores casos:")
        for h, r in worst.iterrows():
            t = sw[sw.text_hash == h].text.iloc[0][:88]
            print(f"    {r.lo:7.2f}h .. {r.hi:7.2f}h  ({int(r.n)} filas)  {t}")

    # ------------------------------------------------------------------
    head(4, "NEAR-DUPLICATES (cosine > 0.90 on TF-IDF)")
    txt = sw.text.fillna("")
    vec = TfidfVectorizer(min_df=2, max_features=60000, ngram_range=(1, 2),
                          strip_accents="unicode", lowercase=True)
    X = vec.fit_transform(txt)
    nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    sim = 1 - dist[:, 1]
    for thr in (0.99, 0.95, 0.90, 0.80):
        n = int((sim >= thr).sum())
        print(f"  filas con un vecino a similitud >= {thr:.2f}: {n:6,} ({n/len(sw)*100:5.2f}%)")
    near = np.where(sim >= 0.90)[0]
    exact = set(sw.index[sw.text_hash.duplicated(keep=False)])
    only_near = [i for i in near if sw.index[i] not in exact]
    print(f"  de esas, NO son duplicado exacto  : {len(only_near):,}")
    if only_near:
        print("\n  ejemplos de casi-duplicado (par):")
        for i in only_near[:4]:
            a, b = sw.iloc[i], sw.iloc[idx[i, 1]]
            print(f"    sim {sim[i]:.3f} | {a.effort_hours:6.2f}h  {a.text[:70]}")
            print(f"                 | {b.effort_hours:6.2f}h  {b.text[:70]}")

    # ------------------------------------------------------------------
    head(5, "EFFORT OUTLIERS AND IMPLAUSIBLE VALUES")
    e = sw.effort_hours
    print(sw.groupby("source").effort_hours.describe(
        percentiles=[.001, .01, .5, .99, .999]).round(3).to_string())
    bands = [("< 6 min (0.1 h)", e < 0.1), ("< 15 min", e < 0.25),
             ("> 40 h (1 semana)", e > 40), ("> 100 h", e > 100),
             ("> 200 h", e > 200), ("> 500 h", e > 500)]
    print()
    for name, m in bands:
        print(f"  {name:22s}: {m.sum():6,} ({m.mean()*100:5.2f}%)  "
              f"[TAWOS {int((m & (sw.source=='TAWOS')).sum()):5,} | "
              f"SiP {int((m & (sw.source=='SiP')).sum()):5,}]")
    print(f"\n  IQR-based outliers (log scale): ", end="")
    lg = np.log(e)
    q1, q3 = lg.quantile(.25), lg.quantile(.75)
    iqr = q3 - q1
    out = (lg < q1 - 1.5 * iqr) | (lg > q3 + 1.5 * iqr)
    print(f"{out.sum():,} ({out.mean()*100:.2f}%)")
    print("\n  valores mas extremos:")
    for _, r in sw.nlargest(5, "effort_hours").iterrows():
        print(f"    {r.effort_hours:9.2f}h [{r.source}] {r.text[:70]}")

    # ------------------------------------------------------------------
    head(6, "TEXT QUALITY")
    sw["n_words"] = sw.text.str.split().str.len()
    print(f"  palabras por texto: mediana {sw.n_words.median():.0f}, "
          f"p10 {sw.n_words.quantile(.1):.0f}, p90 {sw.n_words.quantile(.9):.0f}")
    checks = [
        ("texto < 3 palabras", sw.n_words < 3),
        ("texto < 5 palabras", sw.n_words < 5),
        ("texto > 500 palabras", sw.n_words > 500),
        ("contiene U+FFFD (mojibake)", sw.text.str.contains("�", regex=False)),
        ("parece traza de pila", sw.text.str.count(r"\bat [a-z]+\.[A-Za-z]") >= 3),
        ("contiene 'Exception'/'Traceback'",
         sw.text.str.contains(r"Exception|Traceback|stack trace", case=False, regex=True)),
        ("mayoritariamente URL", sw.text.str.count(r"https?://") >= 3),
        ("titulo solo numeros/ID", sw.title.fillna("").str.fullmatch(r"[\W\d_]+")),
        ("no-ASCII > 20% del texto",
         sw.text.map(lambda s: sum(ord(c) > 127 for c in s) / max(len(s), 1) > .2)),
    ]
    for name, m in checks:
        m = m.fillna(False)
        print(f"  {name:34s}: {m.sum():6,} ({m.mean()*100:5.2f}%)")

    print("\n  titulos mas repetidos (posible plantilla / bot):")
    vc = sw.title.value_counts().head(8)
    for t, n in vc.items():
        med = sw[sw.title == t].effort_hours.median()
        print(f"    {n:4d}x  mediana {med:6.2f}h  {str(t)[:64]}")

    # ------------------------------------------------------------------
    head(7, "CONSISTENCY FLAGS ALREADY IN THE DATA")
    print(f"  effort_gt_calendar = 1        : {int(sw.effort_gt_calendar.sum()):,} "
          f"({sw.effort_gt_calendar.mean()*100:.2f}%)")
    print(f"  sin descripcion (has_desc=0)  : {int((sw.has_desc==0).sum()):,} "
          f"({(sw.has_desc==0).mean()*100:.2f}%)")
    print(f"    de ellas, de TAWOS          : {int(((sw.has_desc==0)&(sw.source=='TAWOS')).sum()):,}")

    # ------------------------------------------------------------------
    head(8, "RARE CATEGORIES (thin classes for the categoriser)")
    k = sw.task_kind.value_counts()
    print(k.to_string())
    print(f"\n  categorias con < 100 filas: {list(k[k < 100].index)}")
    print(f"  categorias con < 50  filas: {list(k[k < 50].index)}")

    # ------------------------------------------------------------------
    head(9, "SPLIT INTEGRITY")
    print(f"  filas software sin split: {int(sw.split.isna().sum()):,}")
    chk = sw.groupby("text_hash").split.nunique()
    print(f"  grupos de texto en mas de un split: {int((chk > 1).sum()):,}")
    print("\n  distribucion de esfuerzo por split (mediana h):")
    print(sw.groupby("split").effort_hours.agg(["size", "median", "mean"]).round(2).to_string())


if __name__ == "__main__":
    main()
