"""Stream the TAWOS MySQL dump and extract only the DDL (CREATE TABLE blocks).

The dump is 4.31 GB uncompressed; we never write it to disk. We read it as a
stream out of the zip, keep a small rolling text buffer, and harvest the
CREATE TABLE ... ; blocks plus a count of INSERT statements per table.
"""
import zipfile, re, sys, json, collections

ZIP = "data/raw/TAWOS.sql.zip"
OUT = "data/interim/tawos_schema.sql"
CHUNK = 8 << 20  # 8 MB

ddl = []
insert_bytes = collections.Counter()
insert_stmts = collections.Counter()
in_ddl = False
cur = []
total = 0

with zipfile.ZipFile(ZIP) as z, z.open("TAWOS.sql") as f:
    tail = ""
    while True:
        raw = f.read(CHUNK)
        if not raw:
            break
        total += len(raw)
        text = tail + raw.decode("utf8", "replace")
        # keep last partial line for the next round
        nl = text.rfind("\n")
        tail, text = text[nl + 1:], text[: nl + 1]
        for line in text.split("\n"):
            if line.startswith("CREATE TABLE "):
                in_ddl = True
                cur = [line]
            elif in_ddl:
                cur.append(line)
                if line.rstrip().endswith(";"):
                    ddl.append("\n".join(cur))
                    in_ddl = False
            elif line.startswith("INSERT INTO "):
                m = re.match(r"INSERT INTO `([^`]+)`", line)
                if m:
                    insert_stmts[m.group(1)] += 1
                    insert_bytes[m.group(1)] += len(line)
        print(f"\r  scanned {total/1e9:.2f} GB", end="", file=sys.stderr)

print(file=sys.stderr)
with open(OUT, "w", encoding="utf8") as fh:
    fh.write("\n\n".join(ddl))

report = {
    t: {"insert_stmts": insert_stmts[t], "insert_MB": round(insert_bytes[t] / 1e6, 1)}
    for t in sorted(insert_bytes, key=lambda k: -insert_bytes[k])
}
with open("data/interim/tawos_table_sizes.json", "w") as fh:
    json.dump(report, fh, indent=2)

print(f"tables with DDL: {len(ddl)}")
for t, v in report.items():
    print(f"  {t:28s} {v['insert_stmts']:6d} stmts  {v['insert_MB']:9.1f} MB")
