"""Streaming reader for the TAWOS mysqldump.

The dump is 4.31 GB uncompressed, so we read it straight out of the zip and
never materialise it. mysqldump escapes newlines inside string literals as the
two-character sequence backslash-n, therefore every INSERT statement occupies
exactly one physical line -- we can stream line-by-line and tokenise each
statement.
"""
import re
import zipfile

# A run of ordinary chars OR a backslash escape, wrapped in single quotes.
# Written as alternating runs so the regex engine never backtracks char-by-char.
_STR = re.compile(r"'((?:[^'\\]+|\\.)*)'", re.S)
_ATOM = re.compile(r"[^,)]*")

_UNESCAPE = {
    "0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t",
    "Z": "\x1a", "\\": "\\", "'": "'", '"': '"',
}


def _unescape(s):
    if "\\" not in s:
        return s
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            out.append(_UNESCAPE.get(nxt, nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def parse_values(stmt, start):
    """Yield one list of python values per tuple in an INSERT ... VALUES line."""
    n = len(stmt)
    i = start
    while i < n:
        while i < n and stmt[i] != "(":
            if stmt[i] == ";":
                return
            i += 1
        if i >= n:
            return
        i += 1
        row = []
        while True:
            if stmt[i] == "'":
                m = _STR.match(stmt, i)
                row.append(_unescape(m.group(1)))
                i = m.end()
            else:
                m = _ATOM.match(stmt, i)
                v = m.group(0).strip()
                row.append(None if v == "NULL" else v)
                i = m.end()
            if stmt[i] == ",":
                i += 1
            elif stmt[i] == ")":
                i += 1
                break
        yield row


def iter_rows(zip_path, table, member="TAWOS.sql", encoding="utf8"):
    """Yield every row of `table` as a list of raw python values."""
    prefix = "INSERT INTO `%s` VALUES " % table
    plen = len(prefix)
    with zipfile.ZipFile(zip_path) as z, z.open(member) as fh:
        for raw in fh:
            if not raw.startswith(b"INSERT INTO `"):
                continue
            line = raw.decode(encoding, "replace")
            if not line.startswith(prefix):
                continue
            yield from parse_values(line, plen)


def column_names(schema_sql_path, table):
    """Column names, in declaration order, from the harvested DDL."""
    with open(schema_sql_path, encoding="utf8") as fh:
        ddl = fh.read()
    m = re.search(
        r"CREATE TABLE `%s` \((.*?)\n\) ENGINE" % re.escape(table), ddl, re.S
    )
    if not m:
        raise KeyError(table)
    cols = []
    for line in m.group(1).split("\n"):
        line = line.strip()
        cm = re.match(r"`([^`]+)` ", line)
        if cm:
            cols.append(cm.group(1))
    return cols
