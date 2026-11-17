"""Prueba de extremo a extremo del estado actual, para revisarla A MANO.

Generador tal cual (19_pipeline, traduccion activada) -> seis descripciones de
proyecto -> backlog -> estimador por el adaptador 34_estimacion, dos veces:
con los codificadores fp32 de siempre y con los ONNX int8 de 246.

Sale un markdown legible: cada historia con sus tareas en espanol, y por tarea
horas y valor probable fp32 | int8, y el total del proyecto con su banda.
"""
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

GEN = r"F:\Software Task generation Model"
os.environ.setdefault("PBE_DEVICE", "cuda")
sys.path.insert(0, "src")
OUT = Path("results/prueba_extremo")


def mod(nombre, ruta):
    s = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


CASOS = [
    ("Reservas de limpieza",
     "Una aplicación web para reservar servicios de limpieza a domicilio, con citas recurrentes y pagos en la app."),
    ("Salas de estudio",
     "Una aplicación móvil donde los estudiantes de una universidad encuentran y reservan salas de estudio en el campus en tiempo real."),
    ("Historias clínicas",
     "Sistema para que una clínica pequeña gestione historias clínicas, citas y recetas, con acceso por roles para médicos y recepción."),
    ("Inventario de ferretería",
     "Un sistema de inventario para una ferretería con varias sucursales, con alertas de stock bajo, transferencias entre tiendas y reportes de ventas."),
    ("Bolsa de trabajo",
     "Portal de empleo para egresados de una universidad, donde las empresas publican ofertas y los egresados postulan con su CV."),
    ("Delivery app (en)",
     "A food delivery app for a small city, where restaurants publish menus, customers order and pay online, and couriers get routes."),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pl = mod("pl", f"{GEN}/src/19_pipeline.py")
    p = pl.Pipeline(traducir=True)
    print(f"generador en {pl.DISPOSITIVO}, {p.carga_s}s", flush=True)
    gen = []
    for titulo, desc in CASOS:
        r = p(desc)
        gen.append((titulo, desc, r))
        print(f"  {titulo:<26} {len(r['historias'])} historias, {r['n_tareas']} tareas",
              flush=True)
    del p
    import torch
    torch.cuda.empty_cache()

    e34 = mod("es34", f"{GEN}/src/34_estimacion.py")
    E = e34.Estimacion(por_tarea=True, dispositivo="cuda")
    f32 = [E.estimar(copy.deepcopy(r)) for _, _, r in gen]
    print(f"fp32 estimado · {time.time() - t0:.0f} s", flush=True)

    i8m = mod("i8", "src/246_codificadores_int8.py")
    for (n, ruta, ml, nz), attr in zip(i8m.COD, ("_st", "_aj", "_ac")):
        setattr(E.est, attr, i8m.CodOnnx(Path("models/onnx") / f"{n}.int8.onnx",
                                         ruta, ml, nz))
    i8 = [E.estimar(copy.deepcopy(r)) for _, _, r in gen]
    print(f"int8 estimado · {time.time() - t0:.0f} s", flush=True)

    json.dump([dict(titulo=t, descripcion=d, fp32=a, int8=b)
               for (t, d, _), a, b in zip(gen, f32, i8)],
              open(OUT / "salidas.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False, default=str)

    L = ["# Prueba de extremo a extremo · estado actual", "",
         "Generador tal cual + estimador con codificadores fp32 | int8. "
         "Horas = mediana por tarea; prob = valor probable (rejilla). "
         "Banda al 80 %.", ""]
    for (titulo, desc, _), a, b in zip(gen, f32, i8):
        ea, eb = a["estimacion"], b["estimacion"]
        L += [f"## {titulo}", "", f"> {desc}", "",
              f"**Total** fp32 {ea['total_horas']} h [{ea['banda_baja']}–{ea['banda_alta']}]"
              f" · int8 {eb['total_horas']} h [{eb['banda_baja']}–{eb['banda_alta']}]"
              f" · {ea['tareas']} tareas", ""]
        for ha, hb in zip(a["historias"], b["historias"]):
            L += [f"### {ha.get('historia_es') or ha['historia']}", "",
                  "| tarea | h fp32 | h int8 | prob fp32 | prob int8 | banda fp32 |",
                  "|---|---|---|---|---|---|"]
            tes = ha.get("tareas_es") or ha["tareas"]
            for i, t in enumerate(tes):
                g = lambda h, k: (h.get(k) or [None] * (i + 1))[i]
                L.append(f"| {t} <br><sub>{ha['tareas'][i]}</sub> | {g(ha,'horas_tareas')} | "
                         f"{g(hb,'horas_tareas')} | {g(ha,'horas_probables')} | "
                         f"{g(hb,'horas_probables')} | {g(ha,'horas_banda')} |")
            L.append("")
    (OUT / "revision.md").write_text("\n".join(L), encoding="utf-8")
    print(f"escrito {OUT / 'revision.md'} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
