"""Genera los backlogs del conjunto de calibracion con una configuracion dada.

La configuracion se fija por variables de entorno ANTES de importar el pipeline, porque
19_pipeline.py las lee al cargarse (PBE_TAXONOMIA, PBE_SPRINT0, PBE_SEED).

  PBE_TAXONOMIA=descripciones_v4.json PBE_SPRINT0=1 PBE_SEED=11 \\
      python src/94_generar_calibracion.py data/calibracion/salidas_v4.json [ids...]

Sin ids genera las 64. Guarda, por backlog, lo necesario para etiquetar con la rubrica y
para calibrar la puerta: la similitud de sector y su margen, el plan de roles, las anclas,
las historias con sus tareas en los dos idiomas y el bloque Sprint 0.

out: el json que se le pase
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", r"F:\PBE-produccion\cache_hf")
sys.stdout.reconfigure(encoding="utf-8")

spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)

SAL = sys.argv[1]
SOLO = set(sys.argv[2:])
D = json.load(open(f"{BASE}/data/calibracion/descripciones.json", encoding="utf-8"))["descripciones"]
if SOLO:
    D = [d for d in D if d["id"] in SOLO]

print(f"taxonomia={pl.TAXONOMIA} sprint0={pl.SPRINT0} semilla={pl.SEMILLA} "
      f"dispositivo={pl.DISPOSITIVO} · {len(D)} descripciones", flush=True)
P = pl.Pipeline(traducir=True)
if pl.DISPOSITIVO == "cuda":
    torch.cuda.reset_peak_memory_stats()


# los sectores de la taxonomia, en el mismo orden que el planificador
_tax = json.load(open(f"{BASE}/data/synth/{pl.TAXONOMIA}", encoding="utf-8"))
_sect = [x["sector"] for x in _tax]

salida = {}
for n, d in enumerate(D, 1):
    t0 = time.time()
    r = P(d["texto"])
    seg = time.time() - t0
    # la descripcion tal como la vio el planificador (traducida si venia en espanol)
    en = getattr(P, "ultima_descripcion_en", None)
    pr = P.planificador
    sim1 = margen = sec1 = sec2 = None
    if pr is not None and getattr(pr, "ok", False):
        txt = en or d["texto"]
        e = pr.sb.encode([txt], normalize_embeddings=True, show_progress_bar=False)[0]
        s = pr.E @ e
        mejor = {}
        for sec, v in zip(_sect, s):
            mejor[sec] = max(mejor.get(sec, -1.0), float(v))
        rk = sorted(mejor.items(), key=lambda z: -z[1])
        (sec1, sim1), (sec2, sim2) = rk[0], rk[1]
        margen = sim1 - sim2
    salida[d["id"]] = dict(
        grupo=d["grupo"], sector_esperado=d.get("sector"), idioma=d["idioma"],
        texto=d["texto"], funciones=d["funciones"], actores=d["actores"],
        sector=sec1, similitud=sim1, segundo_sector=sec2, margen=margen,
        roles=getattr(P, "ultimo_plan", None),
        anclas=list(getattr(P, "ultimas_anclas", []) or []),
        segundos=round(seg, 2),
        historias=[dict(en=h["historia"], es=h.get("historia_es"), origen=h.get("origen"),
                        tareas_en=h["tareas"], tareas_es=h.get("tareas_es"))
                   for h in r["historias"]],
        sprint0=[dict(habilitador=h["habilitador"], en=h["historia"], es=h["historia_es"],
                      tareas_en=h["tareas"], tareas_es=h["tareas_es"])
                 for h in r.get("sprint0", [])],
        n_tareas=r["n_tareas"], n_tareas_sprint0=r.get("n_tareas_sprint0", 0))
    print(f"[{n:>2}/{len(D)}] {d['id']} {seg:5.1f}s  sector={sec1} {sim1 if sim1 is None else round(sim1, 3)}"
          f"  historias={len(r['historias'])} tareas={r['n_tareas']} sprint0={r.get('n_tareas_sprint0', 0)}",
          flush=True)
    json.dump(salida, open(SAL, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

if pl.DISPOSITIVO == "cuda":
    print(f"pico de VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)
print(f"escrito {SAL}", flush=True)
