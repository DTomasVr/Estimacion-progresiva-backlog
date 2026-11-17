"""TAWOS has no parent/epic column, but its Change_Log records every edit of
the Epic Link / Parent fields. Stream the 1.9 GB of Change_Log INSERTs out of
the zip and keep only the hierarchy-bearing rows; count every Field name so we
can see what we filtered away.

Output:
  data/interim/tawos_changelog_hier.parquet   (hierarchy rows)
  results/02_tawos_changelog_fields.csv       (Field -> count, all fields)
"""
import os
import sys
import time
from collections import Counter

import pandas as pd

PBE = r"C:\Users\User\Documents\Programacao\PBE-Dataset"
sys.path.insert(0, PBE + r"\src")
import tawos_sql as T  # noqa: E402

ZIP = PBE + r"\data\raw\TAWOS.sql.zip"
SCHEMA = PBE + r"\data\interim\tawos_schema.sql"
BASE = r"F:\Software Task generation Model"
TOTAL_ESPERADO = 10_116_894
HIER = {"epic link", "parent", "parent issue", "epic child", "epic name",
        "issueparentassociation", "parent link", "sub-task", "subtask", "epic"}


def main():
    cols = T.column_names(SCHEMA, "Change_Log")
    fi = cols.index("Field")
    campos = Counter()
    filas = []
    t0 = time.time()
    siguiente = 500_000
    for n, row in enumerate(T.iter_rows(ZIP, "Change_Log"), 1):
        f = row[fi]
        campos[f] += 1
        if f and f.strip().lower() in HIER:
            filas.append(dict(zip(cols, row)))
        if n >= siguiente:
            seg = time.time() - t0
            eta = (TOTAL_ESPERADO - n) / (n / seg)
            print(f"  {n:>11,} filas  {seg:6.0f}s  ETA {eta/60:5.1f} min  "
                  f"jerarquicas {len(filas):,}", flush=True)
            siguiente += 500_000
    seg = time.time() - t0
    print(f"\n{n:,} filas en {seg:.0f}s ({seg/60:.1f} min); jerarquicas {len(filas):,}")
    df = pd.DataFrame(filas)
    df.to_parquet(f"{BASE}/data/interim/tawos_changelog_hier.parquet", index=False)
    c = pd.Series(campos, name="n").sort_values(ascending=False)
    c.index.name = "Field"
    c.to_csv(f"{BASE}/results/02_tawos_changelog_fields.csv")
    print(c.head(60).to_string())
    if len(df):
        print(df.Field.value_counts().to_string())


if __name__ == "__main__":
    main()
