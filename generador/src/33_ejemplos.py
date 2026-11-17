"""Precomputed examples for the demo page, so a presentation never waits on
generation (or on having a GPU). Six kickoff-style descriptions in Spanish and
English, run through the real pipeline with the Spanish translation on.

out: results/33_ejemplos.json
"""
import importlib.util
import json
import os

BASE = r"F:\Software Task generation Model"
os.environ.setdefault("AGIX_DEVICE", "cuda")
spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)

CASOS = [
    ("Reservas de limpieza",
     "Una aplicación web para reservar servicios de limpieza a domicilio, con citas recurrentes y pagos en la app."),
    ("Salas de estudio",
     "Una aplicación móvil donde los estudiantes de una universidad encuentran y reservan salas de estudio en el campus en tiempo real."),
    ("Cosechas a restaurantes",
     "Plataforma para que pequeños agricultores registren sus cosechas y vendan directamente a restaurantes, con seguimiento de entregas."),
    ("Historias clínicas",
     "Sistema para que una clínica pequeña gestione historias clínicas, citas y recetas, con acceso por roles para médicos y recepción."),
    ("Repositorio de tesis",
     "Un repositorio institucional donde los estudiantes suben sus tesis, se revisan por un comité y se publican con metadatos y búsqueda."),
    ("Study room booking (en)",
     "A mobile app where university students can find and book study rooms on campus in real time, with reminders and check-in."),
]

p = pl.Pipeline(traducir=True)
print(f"pipeline en {pl.DISPOSITIVO}, cargado en {p.carga_s}s", flush=True)
salida = []
for titulo, desc in CASOS:
    r = p(desc)
    salida.append(dict(titulo=titulo, descripcion=desc, resultado=r))
    print(f"  {titulo:<26} {r['tiempos']['total']:.2f} s  "
          f"{len(r['historias'])} historias, {r['n_tareas']} tareas", flush=True)
json.dump(salida, open(f"{BASE}/results/33_ejemplos.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"escrito results/33_ejemplos.json ({len(salida)} ejemplos)")
h = salida[0]["resultado"]["historias"][0]
print(f"\nmuestra: {h.get('historia_es', h['historia'])[:120]}")
for t in h.get("tareas_es", h["tareas"])[:3]:
    print(f"   - {t[:100]}")
