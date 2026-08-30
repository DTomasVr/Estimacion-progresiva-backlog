"""Extract the TAWOS tables we need from the dump into Parquet.

Streams the zip, batches rows, and appends to a ParquetWriter so peak memory
stays bounded regardless of table size.
"""
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "src")
import tawos_sql as T  # noqa: E402

ZIP = "data/raw/TAWOS.sql.zip"
SCHEMA = "data/interim/tawos_schema.sql"
OUT = "data/interim"
BATCH = 25_000

# column -> pandas dtype for the non-string fields
NUMERIC = {
    "ID": "Int64", "Jira_ID": "Int64", "Story_Point": "float64",
    "Timespent": "float64", "In_Progress_Minutes": "float64",
    "Total_Effort_Minutes": "float64", "Resolution_Time_Minutes": "float64",
    "Title_Changed_After_Estimation": "Int8",
    "Description_Changed_After_Estimation": "Int8",
    "Story_Point_Changed_After_Estimation": "Int8",
    "Creator_ID": "Int64", "Reporter_ID": "Int64", "Assignee_ID": "Int64",
    "Project_ID": "Int64", "Sprint_ID": "Int64", "Repository_ID": "Int64",
    "JiraID": "Int64",
}
DATES = {
    "Creation_Date", "Estimation_Date", "Resolution_Date", "Last_Updated",
    "Start_Date", "End_Date", "Activated_Date", "Complete_Date",
    "Last_Update_Date",
}


def frame(rows, cols):
    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:
        if c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(NUMERIC[c])
        elif c in DATES:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def extract(table, drop=()):
    cols = T.column_names(SCHEMA, table)
    keep = [c for c in cols if c not in drop]
    path = f"{OUT}/{table}.parquet"
    writer = None
    batch, n, t0 = [], 0, time.time()

    def flush(rows):
        nonlocal writer, n
        df = frame(rows, cols)[keep]
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        n += len(df)

    for row in T.iter_rows(ZIP, table):
        if len(row) != len(cols):
            raise ValueError(f"{table}: got {len(row)} values, expected {len(cols)}")
        batch.append(row)
        if len(batch) >= BATCH:
            flush(batch)
            batch = []
            print(f"  {table}: {n:,} rows", file=sys.stderr)
    if batch:
        flush(batch)
    if writer:
        writer.close()
    print(f"{table}: {n:,} rows -> {path}  ({time.time()-t0:.0f}s)")
    return n


if __name__ == "__main__":
    extract("Project")
    extract("Repository")
    extract("Sprint")
    extract("User")
    extract("Issue", drop=("URL", "Pull_Request_URL"))
