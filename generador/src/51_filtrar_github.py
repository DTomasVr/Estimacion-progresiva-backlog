"""Separate real work decompositions from the noise in the scraped GitHub issues.

Reading a sample showed the problem: most "task lists" in issue bodies are the
SUBMISSION CHECKLIST that the repo's issue template asks the reporter to tick
("I have searched the existing issues", "I'm a real person"). Those are not
decompositions of anything and would poison training data.

What is kept must look like work someone will do:
  * the parent is a feature or enhancement, not a bug or a support request
  * the items are imperative work items, not first-person declarations
  * the items are not just references to other issues
  * at least 3 usable items survive, because 2 is not a decomposition

out: data/reales/github_filtrado.jsonl  +  a report of what was dropped and why
"""
import json
import os
import re
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = f"{BASE}/data/reales/github_issues.jsonl"
SALIDA = f"{BASE}/data/reales/github_filtrado.jsonl"
MIN_TAREAS = 3

# --- lo que descalifica a la incidencia entera
ETIQUETA_MALA = re.compile(r"bug|support|triage|question|duplicate|invalid|wontfix|stale|"
                           r"needs.?(info|logs|triage|repro)|troubleshoot|regression|crash", re.I)
ETIQUETA_BUENA = re.compile(r"feature|enhancement|epic|improvement|proposal|rfc|story|"
                            r"roadmap|meta|tracking", re.I)
TITULO_MALO = re.compile(r"^\s*(\[?bug\]?|fix|crash|error|broken|regression|\[support\])\b|"
                         r"\bdoes ?n[o']t work\b|\bnot working\b", re.I)

# --- lo que descalifica a un item concreto
DECLARACION = re.compile(r"^\s*(i|we|my|i'?ve|i'?m|this)\b.{0,80}$", re.I)   # "I have searched..."
PLANTILLA = re.compile(r"searched|duplicate|read (the|and)|followed the steps|latest version|"
                       r"provided|guidelines|checked the faq|real person|reproduc|"
                       r"agree to|code of conduct|willing to|checklist|see the wiki|"
                       r"troubleshooting steps|deactivat", re.I)
SOLO_REF = re.compile(r"^\s*(#\d+|https?://\S+|\[[^\]]*\]\(\S+\))[\s,;.]*$")
REF_FINAL = re.compile(r"\s*[-–—]?\s*(#\d+|https?://github\.com/\S+)\s*$")
IMPERATIVO = re.compile(r"^(add|create|implement|update|remove|delete|allow|support|build|make|"
                        r"move|rename|refactor|expose|handle|show|display|store|send|write|fix|"
                        r"migrate|replace|convert|introduce|extend|improve|enable|disable|apply|"
                        r"validate|test|document|design|define|set|configure|integrate|port|"
                        r"provide|generate|render|load|save|track|check|split|merge|clean)\b", re.I)


def limpiar_item(t):
    t = REF_FINAL.sub("", t).strip(" -–—.:;")
    t = re.sub(r"\s+", " ", t)
    return t


def util(t):
    if len(t) < 10 or len(t) > 180:
        return False, "largo"
    if SOLO_REF.match(t):
        return False, "solo referencia"
    if PLANTILLA.search(t):
        return False, "plantilla de issue"
    if DECLARACION.match(t):
        return False, "declaracion del reportante"
    if not IMPERATIVO.match(t):
        return False, "no es un item de trabajo"
    return True, ""


B = [json.loads(l) for l in open(ENTRADA, encoding="utf-8")]
motivos_issue, motivos_item = Counter(), Counter()
salida = []
for b in B:
    ets = " ".join(b["etiquetas"])
    if ETIQUETA_MALA.search(ets) and not ETIQUETA_BUENA.search(ets):
        motivos_issue["etiqueta de bug o soporte"] += 1
        continue
    if TITULO_MALO.search(b["titulo"]):
        motivos_issue["el titulo es un fallo"] += 1
        continue
    buenas = []
    for t in b["tareas"]:
        t = limpiar_item(t)
        ok, por = util(t)
        if ok:
            buenas.append(t)
        else:
            motivos_item[por] += 1
    # dentro de una misma incidencia no deben repetirse
    vistos, unicas = set(), []
    for t in buenas:
        k = re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
        if k not in vistos:
            vistos.add(k)
            unicas.append(t)
    if len(unicas) < MIN_TAREAS:
        motivos_issue[f"menos de {MIN_TAREAS} items utiles"] += 1
        continue
    salida.append(dict(uid=b["uid"], repo=b["repo"], titulo=re.sub(r"\s+", " ", b["titulo"]).strip(),
                       tareas=unicas, etiquetas=b["etiquetas"]))

with open(SALIDA, "w", encoding="utf-8") as fh:
    for x in salida:
        fh.write(json.dumps(x, ensure_ascii=False) + "\n")

n_items = sum(len(b["tareas"]) for b in B)
print(f"ENTRADA: {len(B)} incidencias, {n_items} items")
print("\nincidencias descartadas:")
for k, v in motivos_issue.most_common():
    print(f"  {k:<34} {v:5d}")
print("\nitems descartados:")
for k, v in motivos_item.most_common():
    print(f"  {k:<34} {v:5d}  ({100*v/n_items:.1f} %)")
print(f"\nSOBREVIVEN: {len(salida)} incidencias ({100*len(salida)/len(B):.1f} %), "
      f"{sum(len(x['tareas']) for x in salida)} tareas "
      f"({sum(len(x['tareas']) for x in salida)/max(1,len(salida)):.1f} por incidencia)")
print(f"repos representados: {len(set(x['repo'] for x in salida))}")
for r, n in Counter(x["repo"] for x in salida).most_common(8):
    print(f"   {r:<36} {n}")
print(f"\nescrito {SALIDA}")
