"""One streaming pass over the 5.87 GB mongodump: pull the effort fields out.

Nothing is restored into MongoDB. The archive is read straight from inside the
zip, ungzipped on the fly, and each issue is walked at the byte level so that
the ~300 mostly-null customfields, the embedded changelog and the comments are
skipped by their own length prefixes instead of being decoded.

Two things are written:
  * per-source counters for every issue seen (the denominators we need to report
    honestly what fraction of a public Jira actually logs work), and
  * one row per issue that has timespent > 0, which is the only subset that can
    contribute a real-hours label.
"""
import json
import sys
import time
from collections import defaultdict

import pandas as pd

sys.path.insert(0, "src")
import jira_archive as A  # noqa: E402

ZIP = "data/raw/ThePublicJiraDataset.zip"
OUT = "data/interim/jira_publico"

# names compared as raw bytes: decoding every element name would dominate the
# runtime, and these are the only ones we ever look at
ESCALARES = {b"timespent": "timespent",
             b"timeoriginalestimate": "estimate_orig",
             b"timeestimate": "estimate_rest",
             b"aggregatetimespent": "agg_timespent",
             b"aggregatetimeoriginalestimate": "agg_estimate",
             b"summary": "summary",
             b"description": "description",
             b"created": "created",
             b"resolutiondate": "resolved",
             b"updated": "updated",
             b"workratio": "workratio"}
SUBDOCS = {b"issuetype": "issuetype", b"project": "project",
           b"status": "status", b"assignee": "assignee",
           b"priority": "priority", b"resolution": "resolution"}
ARRAYS = {b"components": "n_componentes", b"labels": "n_etiquetas",
          b"issuelinks": "n_enlaces", b"subtasks": "n_subtareas",
          b"fixVersions": "n_versiones"}
NOMBRE = {b"name", b"key", b"displayName"}


def leer_sub(b, off):
    """Pull name/key out of a small subdocument (issuetype, project, ...)."""
    r = {}
    for name, t, v, ve in A.walk(b, off):
        if t == 0x02 and name in ("name", "key", "displayName", "id"):
            r[name] = b[v + 4:ve - 1].decode("utf-8", "replace")
    return r


def procesar(doc, cont):
    """Return a row dict when the issue has logged time, else None."""
    key = None
    campos = None
    n_cambios = 0
    for name, t, v, ve in A.walk(doc, 0):
        if name == "key" and t == 0x02:
            key = doc[v + 4:ve - 1].decode("utf-8", "replace")
        elif name == "fields" and t == 0x03:
            campos = v
        elif name == "changelog" and t == 0x03:
            for n2, t2, v2, ve2 in A.walk(doc, v):
                if n2 == "total" and t2 == 0x10:
                    n_cambios = A._i32(doc, v2)[0]
    if campos is None:
        return None

    fila = {"key": key, "n_cambios": n_cambios}
    subs = {}
    end = campos + A._i32(doc, campos)[0] - 1
    i = campos + 4
    while i < end:
        t = doc[i]
        j = doc.index(0, i + 1)
        nm = doc[i + 1:j]
        v = j + 1
        ve, _ = A._value_span(doc, t, v)
        if t != 0x0A:                      # nulls carry nothing
            d = ESCALARES.get(nm)
            if d is not None:
                fila[d] = A._decode(doc, t, v, ve)
            elif nm in SUBDOCS:
                subs[SUBDOCS[nm]] = v
            elif nm in ARRAYS:
                fila[ARRAYS[nm]] = sum(1 for _ in A.walk(doc, v))
        i = ve

    cont["issues"] += 1
    ts = fila.get("timespent")
    if fila.get("estimate_orig"):
        cont["con_estimacion"] += 1
    if ts:
        cont["con_tiempo"] += 1
        if fila.get("estimate_orig"):
            cont["con_ambos"] += 1
    else:
        return None

    for etiqueta, off in subs.items():
        s = leer_sub(doc, off)
        if etiqueta == "project":
            fila["project_key"] = s.get("key")
            fila["project_name"] = s.get("name")
        elif etiqueta == "assignee":
            fila["assignee"] = s.get("name") or s.get("key") or s.get("displayName")
        else:
            fila[etiqueta] = s.get("name")
    return fila


def main():
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    import os
    os.makedirs(OUT, exist_ok=True)
    fh = A.open_archive(ZIP)
    A.read_prelude(fh)

    cont = defaultdict(lambda: defaultdict(int))
    filas = defaultdict(list)
    t0 = time.time()
    total = 0
    for db, col, eof, body in A.blocks(fh):
        if eof or not body:
            continue
        c = cont[col]
        for doc in body:
            f = procesar(doc, c)
            if f is not None:
                filas[col].append(f)
        total += len(body)
        if total % 200_000 < len(body):
            seg = time.time() - t0
            print(f"  {total:>9,} issues  {seg:6.0f}s  "
                  f"{total/max(seg,1):>7,.0f}/s  "
                  f"con tiempo {sum(v['con_tiempo'] for v in cont.values()):,}",
                  flush=True)
        if limite and total >= limite:
            break

    print(f"\n{total:,} issues en {time.time()-t0:.0f}s")
    resumen = []
    for col, c in sorted(cont.items()):
        resumen.append(dict(fuente=col, issues=c["issues"],
                            con_estimacion=c["con_estimacion"],
                            con_tiempo=c["con_tiempo"],
                            con_ambos=c["con_ambos"]))
        if filas[col]:
            pd.DataFrame(filas[col]).to_parquet(f"{OUT}/{col}.parquet", index=False)
    r = pd.DataFrame(resumen).sort_values("con_tiempo", ascending=False)
    r["pct_tiempo"] = (100 * r.con_tiempo / r.issues).round(2)
    print(r.to_string(index=False))
    print(f"\nTOTAL issues {r.issues.sum():,}  con tiempo {r.con_tiempo.sum():,}  "
          f"con estimacion+tiempo {r.con_ambos.sum():,}")
    r.to_csv("results/jira_publico_cobertura.csv", index=False)
    json.dump({k: dict(v) for k, v in cont.items()},
              open("results/jira_publico_conteos.json", "w"), indent=1)


if __name__ == "__main__":
    main()
