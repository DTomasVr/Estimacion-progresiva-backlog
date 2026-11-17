"""Full re-extraction of the Public Jira mongodump for task generation.

Unlike Agix-Dataset/src/50_extract_jira.py (which kept only timespent > 0,
2.9 % of the dump), this keeps EVERY issue and the fields that carry the
work-breakdown hierarchy:

  * fields.parent.key        -> sub-task to parent
  * fields.subtasks[].key    -> parent to sub-tasks
  * Epic Link customfield    -> story/task to epic   (id differs per instance,
                                read from 0. DataDefinition/jira_field_information.json)
  * Parent Link customfield  -> Advanced Roadmaps parent (epic to initiative)
  * Epic Name customfield
  * issuelinks[]             -> "type|in/out|KEY" triples

Rows are flushed to data/interim/jira_full/<source>/part-NNNN.parquet every
FLUSH rows so memory stays flat across the 2.7M issues.

usage: python src/01_extract_jira_full.py [limit]
"""
import json
import os
import sys
import time
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import jira_archive as A  # noqa: E402

AGIX = r"C:\Users\User\Documents\Programacao\Agix-Dataset"
ZIP = AGIX + r"\data\raw\ThePublicJiraDataset.zip"
DEF = AGIX + r"\data\raw\jira_public\ThePublicJiraDataset\0. DataDefinition\jira_field_information.json"
OUT = r"F:\Software Task generation Model\data\interim\jira_full"
RES = r"F:\Software Task generation Model\results"
TOTAL_ESPERADO = 2_686_282          # measured in the previous full pass
FLUSH = 100_000
MAX_DESC = 20_000                   # stack traces can run to MBs; keep the prose head

ESCALARES = {b"summary": "summary", b"description": "description",
             b"created": "created", b"resolutiondate": "resolved",
             b"updated": "updated"}
SUBDOCS = {b"issuetype": "issuetype", b"status": "status",
           b"priority": "priority", b"resolution": "resolution"}


def campos_custom():
    """Per source: {bytes field id: column} for Epic Link / Epic Name / Parent Link."""
    info = json.load(open(DEF, encoding="utf-8"))
    quiero = {"Epic Link": "epic_link", "Epic Name": "epic_name", "Parent Link": "parent_link"}
    out = {}
    for src, fields in info.items():
        m = {}
        for f in fields:
            col = quiero.get((f.get("name") or "").strip())
            if col and f.get("id", "").startswith("customfield_"):
                m[f["id"].encode()] = col
        out[src] = m
    return out


def _s(doc, v, ve):
    return doc[v + 4:ve - 1].decode("utf-8", "replace")


def sub(doc, off, keys=("name", "key")):
    r = {}
    for name, t, v, ve in A.walk(doc, off):
        if name in keys:
            r[name] = _s(doc, v, ve) if t == 0x02 else A._decode(doc, t, v, ve)
    return r


def links(doc, off):
    out = []
    for _, t, v, _ in A.walk(doc, off):
        if t != 0x03:
            continue
        tipo, dirn, key = None, None, None
        for n2, t2, v2, _ in A.walk(doc, v):
            if n2 == "type" and t2 == 0x03:
                tipo = sub(doc, v2, ("name",)).get("name")
            elif n2 in ("inwardIssue", "outwardIssue") and t2 == 0x03:
                dirn = "in" if n2[0] == "i" else "out"
                key = sub(doc, v2, ("key",)).get("key")
        if key:
            out.append(f"{tipo}|{dirn}|{key}")
    return ";".join(out)


def procesar(doc, custom):
    key, campos, jid = None, None, None
    for name, t, v, ve in A.walk(doc, 0):
        if name == "key" and t == 0x02:
            key = _s(doc, v, ve)
        elif name == "id" and t == 0x02:
            jid = _s(doc, v, ve)
        elif name == "fields" and t == 0x03:
            campos = v
    if campos is None:
        return None
    fila = {"key": key, "jira_id": jid}
    end = campos + A._i32(doc, campos)[0] - 1
    i = campos + 4
    while i < end:
        t = doc[i]
        j = doc.index(0, i + 1)
        nm = doc[i + 1:j]
        v = j + 1
        ve, _ = A._value_span(doc, t, v)
        if t != 0x0A:
            if nm in ESCALARES:
                x = A._decode(doc, t, v, ve)
                fila[ESCALARES[nm]] = x[:MAX_DESC] if isinstance(x, str) else x
            elif nm in SUBDOCS and t == 0x03:
                s = sub(doc, v, ("name", "subtask"))
                fila[SUBDOCS[nm]] = s.get("name")
                if nm == b"issuetype":
                    fila["is_subtask"] = bool(s.get("subtask"))
            elif nm == b"project" and t == 0x03:
                s = sub(doc, v, ("key", "name", "projectCategory"))
                fila["project_key"] = s.get("key")
                fila["project_name"] = s.get("name")
                pc = s.get("projectCategory")
                if isinstance(pc, (bytes, bytearray)):
                    c = sub(bytes(pc), 0, ("name", "description"))
                    fila["project_category"] = c.get("name")
                    fila["project_category_desc"] = c.get("description")
            elif nm == b"parent" and t == 0x03:
                fila["parent_key"] = sub(doc, v, ("key",)).get("key")
            elif nm == b"subtasks" and t == 0x04:
                ks = [sub(doc, v2, ("key",)).get("key") for _, t2, v2, _ in A.walk(doc, v) if t2 == 0x03]
                fila["subtask_keys"] = ";".join(k for k in ks if k)
            elif nm == b"issuelinks" and t == 0x04:
                fila["links"] = links(doc, v)
            elif nm == b"components" and t == 0x04:
                fila["components"] = ";".join(filter(None, (sub(doc, v2, ("name",)).get("name")
                                                            for _, t2, v2, _ in A.walk(doc, v) if t2 == 0x03)))
            elif nm == b"labels" and t == 0x04:
                fila["labels"] = ";".join(A._decode(doc, t2, v2, ve2) for _, t2, v2, ve2 in A.walk(doc, v) if t2 == 0x02)
            elif nm in custom:
                x = A._decode(doc, t, v, ve)
                if isinstance(x, str):
                    fila[custom[nm]] = x
        i = ve
    return fila


def main():
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    custom = campos_custom()
    fh = A.open_archive(ZIP)
    A.read_prelude(fh)
    filas = defaultdict(list)
    partes = defaultdict(int)
    cont = defaultdict(lambda: defaultdict(int))

    def flush(col):
        if not filas[col]:
            return
        d = f"{OUT}/{col}"
        os.makedirs(d, exist_ok=True)
        pd.DataFrame(filas[col]).to_parquet(f"{d}/part-{partes[col]:04d}.parquet", index=False)
        partes[col] += 1
        filas[col].clear()

    t0 = time.time()
    total, siguiente = 0, 100_000
    for db, col, eof, body in A.blocks(fh):
        if eof or not body:
            continue
        cu = custom.get(col, {})
        c = cont[col]
        for doc in body:
            f = procesar(doc, cu)
            if f is None:
                continue
            f["source"] = col
            c["issues"] += 1
            c["con_parent"] += bool(f.get("parent_key"))
            c["con_subtasks"] += bool(f.get("subtask_keys"))
            c["con_epic_link"] += bool(f.get("epic_link"))
            c["con_parent_link"] += bool(f.get("parent_link"))
            c["es_epic"] += f.get("issuetype") == "Epic"
            filas[col].append(f)
        if len(filas[col]) >= FLUSH:
            flush(col)
        total += len(body)
        if total >= siguiente:
            seg = time.time() - t0
            ritmo = total / seg
            eta = (TOTAL_ESPERADO - total) / ritmo
            print(f"  {total:>9,} issues  {seg:6.0f}s  {ritmo:>7,.0f}/s  "
                  f"ETA {eta/60:5.1f} min  ({100*total/TOTAL_ESPERADO:4.1f} %)", flush=True)
            siguiente += 100_000
        if limite and total >= limite:
            break
    for col in list(filas):
        flush(col)

    seg = time.time() - t0
    print(f"\n{total:,} issues en {seg:.0f}s ({seg/60:.1f} min)")
    r = pd.DataFrame([dict(fuente=k, **v) for k, v in cont.items()]).fillna(0)
    r = r.sort_values("issues", ascending=False)
    print(r.to_string(index=False))
    tag = f"_limit{limite}" if limite else ""
    r.to_csv(f"{RES}/01_jira_full_conteos{tag}.csv", index=False)


if __name__ == "__main__":
    main()
