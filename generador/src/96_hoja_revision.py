"""Hoja de revision humana del pre-etiquetado: una muestra de backlogs, legibles, con la
pre-etiqueta al lado para confirmarla o corregirla.

La muestra se estratifica por grupo y por pre-etiqueta, para que la revision ponga a
prueba los tres niveles (bien / regular / mal) y no solo el mas frecuente. Con semilla.

  python src/96_hoja_revision.py data/calibracion/salidas_v4.json \\
         data/calibracion/etiquetas_pre_v4.csv data/calibracion/REVISION.md 15

out: el markdown que se le pase
"""
import csv
import json
import random
import sys

sys.stdout.reconfigure(encoding="utf-8")
SAL, ETQ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 15
J = json.load(open(SAL, encoding="utf-8"))
E = {r["id"]: r for r in csv.DictReader(open(ETQ, encoding="utf-8"))}
rng = random.Random(20261107)

# estratos (grupo, pre-etiqueta); se reparte la muestra entre estratos lo mas parejo posible
estratos = {}
for k, r in E.items():
    if r["grupo"] in ("vaga", "invalida"):
        continue
    estratos.setdefault((r["grupo"], r["etiqueta"]), []).append(k)
for v in estratos.values():
    rng.shuffle(v)
muestra, ronda = [], 0
while len(muestra) < N and any(len(v) > ronda for v in estratos.values()):
    for clave in sorted(estratos):
        if len(estratos[clave]) > ronda and len(muestra) < N:
            muestra.append(estratos[clave][ronda])
    ronda += 1
muestra.sort()

L = ["# Revisión del pre-etiquetado", "",
     f"{len(muestra)} backlogs elegidos al azar, repartidos entre grupos y entre etiquetas, para",
     "comprobar si el pre-etiquetado automático acierta. Para cada uno: lee el backlog, mira",
     "la pre-etiqueta y anota en **Tu etiqueta** si estás de acuerdo o qué cambiarías.",
     "La rúbrica está en `RUBRICA.md`: gana el peor de los cuatro criterios.", "",
     "| id | grupo | pre-etiqueta | **tu etiqueta** |", "|---|---|---|---|"]
for k in muestra:
    L.append(f"| {k} | {E[k]['grupo']} | {E[k]['etiqueta']} |  |")
for k in muestra:
    v, r = J[k], E[k]
    L += ["", "---", "", f"## {k} · pre-etiqueta: **{r['etiqueta']}**", "",
          f"> {v['texto']}", "",
          f"- **Funciones que nombra:** {'; '.join(v['funciones'])}",
          f"- **Actores que nombra:** {', '.join(v['actores'])}",
          f"- **Sector detectado:** {v['sector']} ({v['similitud']:.3f})",
          f"- **C1** función principal {r['c1_principal']}, cubiertas {r['c1_cubiertas']} de "
          f"{r['c1_total']} · **C2** roles ajenos {r['c2_ajenos']} · **C3** deriva "
          f"{r['c3_deriva']} · **C4** tareas malas {r['c4_malas']} de {r['c4_total']}"]
    if r["nota"]:
        L.append(f"- **Nota automática:** {r['nota']}")
    L += ["", "**Historias**", ""]
    for i, h in enumerate(v["historias"], 1):
        marca = " *(ancla)*" if h.get("origen") == "ancla" else ""
        L.append(f"{i}. {h['es'] or h['en']}{marca}")
        for t in (h["tareas_es"] or h["tareas_en"]):
            L.append(f"    - {t}")
    L += ["", "**Tu etiqueta:** ______  ·  **Qué falla, si algo:** ______"]
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(f"{len(muestra)} backlogs -> {OUT}")
print("estratos:", {f"{g}/{e}": len(v) for (g, e), v in sorted(estratos.items())})
