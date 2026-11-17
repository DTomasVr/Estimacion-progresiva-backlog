"""Shared text cleaner for Jira wiki markup (Public Jira) and the CSV-quoted
prose of TAWOS Description_Text. Used by the SBERT pairs and, later, by the
seq2seq preprocessing, so both see identical text."""
import re

_BLOQUES = re.compile(r"\{(code|noformat|panel|quote)[^}]*\}.*?\{\1\}", re.S | re.I)
_BLOQUE_ABIERTO = re.compile(r"\{(code|noformat)[^}]*\}.*", re.S | re.I)   # unclosed: drop tail
_MACRO = re.compile(r"\{[a-zA-Z]+(?::[^}]*)?\}")                           # {color:red} {html} ...
_LINK = re.compile(r"\[([^|\]\[]*)\|[^\]]*\]")                             # [text|url] -> text
_LINK_SOLO = re.compile(r"\[(?:https?|mailto|file):[^\]]*\]|\[~[^\]]*\]")  # [http://..] [~user]
_URL = re.compile(r"https?://\S+|www\.\S+")
_IMG = re.compile(r"![^!\s]+(?:\|[^!]*)?!")
_HEAD = re.compile(r"(?m)^\s*(?:h[1-6]\.|bq\.|#+|\*+|-+)\s+")
_TABLA = re.compile(r"\|\|?")
_ENF = re.compile(r"(?<![\w*])([*_+^~-])(\S(?:.*?\S)?)\1(?![\w*])")
_HTML = re.compile(r"<[^>]{1,200}>")
_TRAZA = re.compile(r"(?m)^\s*(?:at [\w$.<>]+\(.*\)|Caused by:.*|\.\.\. \d+ more)\s*$")
_ESP = re.compile(r"\s+")


def limpiar(texto, max_chars=None):
    if not isinstance(texto, str) or not texto:
        return ""
    t = texto.replace("\r", "\n").replace("�", " ")
    # TAWOS Description_Text: CSV-escaped, wrapped in """ ... """
    t = t.replace('""""', '"').replace('"""', "").replace('""', '"')
    t = _BLOQUES.sub(" ", t)
    t = _BLOQUE_ABIERTO.sub(" ", t)
    t = _TRAZA.sub(" ", t)
    t = _IMG.sub(" ", t)
    t = _LINK.sub(r"\1", t)
    t = _LINK_SOLO.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _MACRO.sub(" ", t)
    t = _HTML.sub(" ", t)
    t = _HEAD.sub(" ", t)
    t = _TABLA.sub(" ", t)
    t = _ENF.sub(r"\2", t)
    t = _ESP.sub(" ", t).strip()
    if len(t) > 1 and t[0] == t[-1] == '"':                               # TAWOS titles
        t = t[1:-1].strip()
    if max_chars and len(t) > max_chars:
        corte = t.rfind(" ", 0, max_chars)
        t = t[:corte if corte > max_chars // 2 else max_chars]
    return t


def texto_issue(summary, description, max_desc=300):
    """summary + head of description, the unit both SBERT and mT5 see."""
    s = limpiar(summary)
    d = limpiar(description, max_desc)
    if d and d.lower() != s.lower():
        return f"{s}. {d}" if s and not s.endswith((".", "?", "!", ":")) else f"{s} {d}".strip()
    return s
