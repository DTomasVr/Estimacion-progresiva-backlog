"""Precomputed demo examples from domains the corpus actually covers.

The previous six (results/33_ejemplos_consumo.json) were all consumer products --
booking, clinics, farm produce -- which is the documented weak spot: the corpus is
2.8M technical Jira issues, so those come out vague. These use infrastructure,
developer tooling, data platforms, LMS and ledger descriptions, which is what the
model was trained on.

Run with the FULL configuration (two generators, beam 4), plus the within-story
de-duplication measured in src/39: at cosine 0.95 it removes 17 % of the tasks and
costs 0.0023 of soft-F1. The shipped pipeline does not do this yet (annex A14).

out: results/33_ejemplos.json   (the demo page reads this path)
"""
import importlib.util
import json
import os
import re

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("AGIX_DEVICE", "cuda")
spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
from sentence_transformers import SentenceTransformer  # noqa: E402

UMBRAL_INTRA = 0.95

CASOS = [
    ("Tienda de audiolibros",
     "Una plataforma de comercio electrónico para vender audiolibros y ebooks, con muestras de "
     "los primeros capítulos, biblioteca personal y pagos en la app."),
    ("Reservas de limpieza",
     "Una aplicación web para reservar servicios de limpieza a domicilio, con citas recurrentes, "
     "calificación de los trabajadores y pagos en la app."),
    ("Restaurante",
     "Un sistema de pedidos para un restaurante, donde la cocina ve los pedidos y su estado, el "
     "personal atiende peticiones especiales y el dueño consulta los ingresos del día."),
    ("Supermercado en línea",
     "Una app de compra en línea de un supermercado donde los clientes buscan productos, repiten "
     "un pedido anterior y reciben avisos de descuentos, y los empleados confirman las entregas."),
    ("Gimnasio",
     "Un sistema de membresías de gimnasio donde los clientes abren una cuenta, eligen y renuevan "
     "su plan, y los entrenadores programan clases en las salas disponibles."),
    ("Colegio",
     "Un sistema de gestión escolar con cuentas para estudiantes, profesores, apoderados y "
     "administración, con horarios, mensajería con el tutor y reporte de inasistencias."),
    ("Ticketera de eventos",
     "Una plataforma de venta de entradas donde los asistentes compran y ven sus eventos, los "
     "artistas gestionan su página, y la empresa valida las entradas."),
    ("Bookstore (en)",
     "An e-commerce platform for selling audiobooks and ebooks, with chapter previews, a personal "
     "library and in-app payments."),
    ("Clinic booking (en)",
     "A clinic system to manage medical records, appointments and prescriptions, with role-based "
     "access for doctors and reception."),
]

p = pl.Pipeline(traducir=True)                      # sin --ligero: dos generadores y haz de 4
sb = SentenceTransformer(pl.SBERT, device=pl.DISPOSITIVO)
print(f"pipeline en {pl.DISPOSITIVO} ({pl.CONJUNTO}, haz {pl.BEAMS_ABAJO}), cargado en {p.carga_s}s",
      flush=True)


def degenerada(t):
    """delega en el pipeline: una sola implementacion del filtro"""
    return pl.degenerada_texto(t)


def quitar_repetidas(h):
    """drop tasks that repeat another task of the SAME story (the A14 fix, at 0.95)"""
    ts = h.get("tareas") or []
    if len(ts) < 2:
        return 0
    E = sb.encode(ts, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    S = (E @ E.T).float().cpu().numpy()
    np.fill_diagonal(S, 0)
    quedan = [i for i in range(len(ts))
              if not (S[i, :i] >= UMBRAL_INTRA).any() and not degenerada(ts[i])
              and not degenerada(h.get("tareas_es", ts)[i] if h.get("tareas_es") and i < len(h["tareas_es"]) else ts[i])]
    fuera = len(ts) - len(quedan)
    h["tareas"] = [ts[i] for i in quedan]
    if h.get("tareas_es"):
        h["tareas_es"] = [h["tareas_es"][i] for i in quedan if i < len(h["tareas_es"])]
    return fuera


salida = []
for titulo, desc in CASOS:
    r = p(desc)
    # the same failure mode shows up in the story itself, not only in its tasks
    antes_h = len(r["historias"])
    r["historias"] = [h for h in r["historias"]
                      if not degenerada(h["historia"])
                      and not degenerada(h.get("historia_es") or h["historia"])]
    fuera = sum(quitar_repetidas(h) for h in r["historias"])
    r["historias"] = [h for h in r["historias"] if h["tareas"]]   # sin tareas no sirve de ejemplo
    fuera_h = antes_h - len(r["historias"])
    r["n_tareas"] = sum(len(h["tareas"]) for h in r["historias"])
    votos = [h["votos"] for h in r["historias"]]
    rel = [h.get("relevancia", 0) for h in r["historias"]]
    salida.append(dict(titulo=titulo, descripcion=desc, resultado=r))
    print(f"  {titulo:<24} {r['tiempos']['total']:5.2f}s  {len(r['historias'])} historias, "
          f"{r['n_tareas']:3d} tareas ({r['n_tareas']/max(1,len(r['historias'])):.1f} por historia, "
          f"-{fuera} tareas y -{fuera_h} historias repetidas)  votos max {max(votos)} media {np.mean(votos):.1f}  "
          f"relevancia {np.mean(rel):.3f}", flush=True)

json.dump(salida, open(f"{BASE}/results/33_ejemplos.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print(f"\nescrito results/33_ejemplos.json ({len(salida)} ejemplos)")
h = salida[0]["resultado"]["historias"][0]
print(f"\nmuestra: {h.get('historia_es', h['historia'])[:120]}")
for t in (h.get("tareas_es") or h["tareas"])[:3]:
    print(f"   - {t[:100]}")
