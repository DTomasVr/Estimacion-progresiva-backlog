"""Streaming reader for the mongodump archive of The Public Jira Dataset.

The dump was produced with `mongodump --gzip --archive`, which wraps the whole
archive in ONE gzip stream (confirmed: the file starts with 1f 8b). Inside, the
layout is

    magic 0x8199e26d
    prelude: header doc, then one metadata doc per collection, then terminator
    blocks:  namespace-header doc, then body docs, then terminator  (repeated)

where a terminator is the four bytes ff ff ff ff sitting where a BSON length
prefix would be. A namespace header carrying "EOF": true closes that namespace.

Documents are walked lazily. A full BSON decode is unaffordable here: issues
embed their changelog (32M entries overall) and comments (9M), so decoding
everything would cost far more than the handful of fields we actually read. The
walker uses each element's own length prefix to skip whatever was not asked for.
"""
import gzip
import struct
import zipfile

MAGIC = 0x8199E26D
TERM = b"\xff\xff\xff\xff"

_i32 = struct.Struct("<i").unpack_from
_i64 = struct.Struct("<q").unpack_from
_f64 = struct.Struct("<d").unpack_from


# --------------------------------------------------------------- BSON walking
def _cstring_end(b, i):
    return b.index(0, i)


def _value_span(b, t, i):
    """Return (end_offset, kind) for the value of type `t` starting at `i`."""
    if t == 0x01:                       # double
        return i + 8, "d"
    if t in (0x02, 0x0D, 0x0E):         # string / code / symbol
        return i + 4 + _i32(b, i)[0], "s"
    if t in (0x03, 0x04, 0x0F):         # doc / array / code-with-scope
        return i + _i32(b, i)[0], "o"
    if t == 0x05:                       # binary
        return i + 5 + _i32(b, i)[0], "b"
    if t in (0x06, 0x0A, 0xFF, 0x7F):   # undefined / null / min / max
        return i, "n"
    if t == 0x07:                       # ObjectId
        return i + 12, "b"
    if t == 0x08:                       # bool
        return i + 1, "?"
    if t in (0x09, 0x11, 0x12):         # datetime / timestamp / int64
        return i + 8, "l"
    if t == 0x0B:                       # regex: two cstrings
        j = _cstring_end(b, i) + 1
        return _cstring_end(b, j) + 1, "n"
    if t == 0x0C:                       # dbpointer
        return i + 4 + _i32(b, i)[0] + 12, "b"
    if t == 0x10:                       # int32
        return i + 4, "i"
    if t == 0x13:                       # decimal128
        return i + 16, "b"
    raise ValueError(f"tipo BSON desconocido 0x{t:02x} en {i}")


def _decode(b, t, i, end):
    if t == 0x01:
        return _f64(b, i)[0]
    if t in (0x02, 0x0D, 0x0E):
        return b[i + 4:end - 1].decode("utf-8", "replace")
    if t == 0x08:
        return b[i] == 1
    if t == 0x10:
        return _i32(b, i)[0]
    if t in (0x09, 0x11, 0x12):
        return _i64(b, i)[0]
    if t in (0x06, 0x0A):
        return None
    return b[i:end]          # raw slice: caller decides (subdocs, arrays, oids)


def walk(b, off=0):
    """Yield (name, type, value_start, value_end) for one BSON document."""
    end = off + _i32(b, off)[0] - 1
    i = off + 4
    while i < end:
        t = b[i]
        j = _cstring_end(b, i + 1)
        name = b[i + 1:j].decode("utf-8", "replace")
        v = j + 1
        ve, _ = _value_span(b, t, v)
        yield name, t, v, ve
        i = ve


def pick(b, off, wanted):
    """Top-level scalar/raw fetch of the named fields from one document."""
    out = {}
    for name, t, v, ve in walk(b, off):
        if name in wanted:
            out[name] = _decode(b, t, v, ve)
    return out


def array_len(b, off):
    return sum(1 for _ in walk(b, off))


# ------------------------------------------------------------ archive framing
def _read_exact(fh, n):
    out = fh.read(n)
    while len(out) < n:
        more = fh.read(n - len(out))
        if not more:
            raise EOFError(f"faltan {n - len(out)} bytes")
        out += more
    return out


def _next_doc(fh):
    """Read one BSON doc, or None when the terminator is reached."""
    head = fh.read(4)
    if len(head) < 4:
        return None
    if head == TERM:
        return None
    n = _i32(head, 0)[0]
    return head + _read_exact(fh, n - 4)


def open_archive(zip_path, member=None):
    z = zipfile.ZipFile(zip_path)
    if member is None:
        member = next(i.filename for i in z.infolist()
                      if i.filename.endswith(".archive"))
    return gzip.GzipFile(fileobj=z.open(member), mode="rb")


def read_prelude(fh):
    magic = _i32(_read_exact(fh, 4), 0)[0] & 0xFFFFFFFF
    if magic != MAGIC:
        raise ValueError(f"magic inesperado 0x{magic:08x}")
    docs = []
    while (d := _next_doc(fh)) is not None:
        docs.append(d)
    return docs


def blocks(fh):
    """Yield (db, collection, is_eof, [raw docs]) block by block."""
    while True:
        head = _next_doc(fh)
        if head is None:
            return
        h = {}
        for name, t, v, ve in walk(head, 0):
            h[name] = _decode(head, t, v, ve)
        body = []
        while (d := _next_doc(fh)) is not None:
            body.append(d)
        yield h.get("db"), h.get("collection"), bool(h.get("EOF")), body
