"""Evalua el extractor de actores desde la descripcion.

Dos pruebas, ninguna circular:

1. Los 6 proyectos reales de data/reales/test_real.json. El plantel VERDADERO son
   los actores que las personas usaron en sus 328 historias ("As a <rol>"), asi
   que se puede medir precision y cobertura de verdad. Se compara el extractor
   contra el planificador por sector, en el mismo sitio.

2. Las 622 descripciones sinteticas v3, donde se conoce el plantel del sector.
   Aqui solo se mide precision (que lo extraido sea un actor real del sector) y
   cuantas descripciones dan al menos dos actores, porque la descripcion no
   nombra a todo el plantel y la cobertura seria injusta.

  AGIX_DEVICE=cpu python src/80_eval_extractor.py

out: results/80_extractor.csv
"""
import importlib.util
import json
import os
import re

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AGIX_DEVICE", "cpu")
spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
from sentence_transformers import SentenceTransformer  # noqa: E402

RE_ROL = re.compile(r"^\s*as\s+(?:an?|the)?\s*([^,]{2,40}?)\s*,", re.I)


def norm(r):
    r = re.sub(r"[^a-z\s'-]", " ", (r or "").lower()).strip()
    r = re.sub(r"^(a|an|the)\s+", "", r)
    return " ".join(pl.singular(w) for w in r.split())


def casa(x, conjunto):
    """x cuenta como acertado si coincide, contiene o esta contenido en alguno"""
    return any(x == y or x in y or y in x for y in conjunto)


sb = SentenceTransformer(pl.SBERT, device=os.environ["AGIX_DEVICE"])
sb.max_seq_length = 128
plan = pl.PlanificadorRoles(sb)

# ---------------------------------------------------------------- 1. reales
T = json.load(open(f"{BASE}/data/reales/test_real.json", encoding="utf-8"))
filas = []
for x in T:
    d = x["descripcion"]
    humanos = {norm(m.group(1)) for h in x["historias"] if (m := RE_ROL.match(h))}
    humanos.discard("")
    e = sb.encode([d], normalize_embeddings=True, show_progress_bar=False)[0]
    sim = float((plan.E @ e).max())
    ext = [norm(r) for r in pl.actores_nombrados(d, 4)]
    # que habria devuelto el planificador por sector, ignorando el umbral
    o = np.argsort(-(plan.E @ e))
    sect = []
    for i in o[:3]:
        for r in plan.roles[int(i)]:
            if r not in sect:
                sect.append(r)
    sect = [norm(r) for r in sect[:4]]
    for etq, plantel in [("extractor", ext), ("sector", sect)]:
        ok = [p for p in plantel if casa(p, humanos)]
        cub = [h for h in humanos if casa(h, plantel)]
        filas.append(dict(prueba="reales", proyecto=x["proyecto"], metodo=etq,
                          sim_sector=round(sim, 3), n_humanos=len(humanos),
                          plantel=" | ".join(plantel), n=len(plantel),
                          aciertos=len(ok),
                          precision=round(len(ok) / max(1, len(plantel)), 3),
                          cobertura=round(len(cub) / max(1, len(humanos)), 3)))
R1 = pd.DataFrame(filas)
print("=== 1. seis proyectos reales, contra los actores que usaron las personas\n")
for x, g in R1.groupby("proyecto", sort=False):
    h = g.iloc[0]
    print(f"[{x}]  sim de sector {h.sim_sector}  ({h.n_humanos} actores humanos)")
    for _, r in g.iterrows():
        print(f"   {r.metodo:<10} prec {r.precision:.2f} cob {r.cobertura:.2f}  "
              f"{r.plantel}")
print("\n" + R1.groupby("metodo")[["precision", "cobertura"]].mean().round(3).to_string())
print("\nsolo donde el sector NO llega (sim < 0.65):")
sub = R1[R1.sim_sector < 0.65]
if len(sub):
    print(sub.groupby("metodo")[["precision", "cobertura"]].mean().round(3).to_string())

# ------------------------------------------------------------- 2. v3 sinteticas
D = json.load(open(f"{BASE}/data/synth/descripciones_v3.json", encoding="utf-8"))
f2 = []
for x in D:
    roster = [norm(r) for r in x["roles"]]
    ext = [norm(r) for r in pl.actores_nombrados(x["descripcion"], 4)]
    ok = [p for p in ext if casa(p, roster)]
    f2.append(dict(prueba="v3", sector=x["sector"], n=len(ext), aciertos=len(ok),
                   precision=len(ok) / max(1, len(ext)) if ext else np.nan,
                   dos_o_mas=len(ext) >= 2))
R2 = pd.DataFrame(f2)
print(f"\n=== 2. {len(R2)} descripciones sinteticas v3, contra el plantel del sector")
print(f"  actores extraidos por descripcion: media {R2.n.mean():.2f}  "
      f"(0 en {int((R2.n == 0).sum())}, 1 en {int((R2.n == 1).sum())})")
print(f"  precision media (de las que extraen algo): {R2.precision.mean():.3f}")
print(f"  descripciones con 2 o mas actores: {100*R2.dos_o_mas.mean():.1f} %")

pd.concat([R1, R2], ignore_index=True).to_csv(f"{BASE}/results/80_extractor.csv", index=False)
print("\n-> results/80_extractor.csv")
