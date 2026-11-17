"""Prueba de extremo a extremo de la API desplegada en el Space de Hugging Face.

    python src/276_probar_space.py

Lee AGIX_API_KEY de .env.llm (nunca la imprime) y comprueba:

  1 /salud dice listo
  2 /backlog con una descripcion en espanol: historias en espanol, tareas en ingles,
    horas alineadas con sus tareas, total positivo y dentro de su banda
  3 /estimar sobre ese backlog: el frio coincide con el de /backlog; 2 respuestas dan
    el mismo total que 2 cerradas; las cerradas estrechan la banda y las respuestas no;
    el previo escala el total por e^previo; una lista mal alineada da 422
  4 /desplazamiento devuelve un numero finito, y usado como previo reproduce las horas
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import requests

URL = "https://agixapp-agix-app.hf.space"
DESC = ("Una aplicación web para que los vecinos de un edificio reserven las áreas comunes "
        "(gimnasio, parrilla y sala de reuniones), paguen la cuota de mantenimiento y reciban "
        "avisos de la administración.")
OUT = Path("results/probar_space")


def clave():
    for l in open(".env.llm", encoding="utf-8-sig"):
        if l.strip().startswith("AGIX_API_KEY="):
            return l.split("=", 1)[1].strip().strip('"')
    sys.exit("falta AGIX_API_KEY en .env.llm")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["X-API-Key"] = clave()
    fallos = []

    def ok(cond, texto):
        print(("  OK    " if cond else "  FALLO ") + texto, flush=True)
        if not cond:
            fallos.append(texto)

    def post(ruta, cuerpo, timeout=300):
        t = time.time()
        r = s.post(URL + ruta, json=cuerpo, timeout=timeout)
        return r, time.time() - t

    print("1 · /salud")
    sal = requests.get(URL + "/salud", timeout=60).json()
    ok(sal["estado"] == "listo", f"estado {sal['estado']} · estimador {sal.get('estimador')}")

    print("2 · /backlog")
    r, dt = post("/backlog", {"descripcion": DESC, "k_historias": 12}, timeout=400)
    ok(r.status_code == 200, f"HTTP {r.status_code} en {dt:.0f} s")
    b = r.json()
    json.dump(b, open(OUT / "backlog.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    H = b["historias"]
    e = b["estimacion"]
    n = sum(len(h["tareas"]) for h in H)
    ok(len(H) == 12, f"{len(H)} historias · {n} tareas (n_tareas dice {b.get('n_tareas')})")
    ok(all(h.get("historia_es", "").startswith("Como ") for h in H), "las historias en español empiezan por «Como»")
    ok(all(len(h["tareas"]) == len(h.get("tareas_es", [])) == len(h.get("horas_tareas", []))
           == len(h.get("horas_probables", [])) == len(h.get("horas_banda", [])) for h in H),
       "tareas, tareas_es, horas y bandas alineadas en cada historia")
    ok(all(x > 0 for h in H for x in h["horas_tareas"]), "todas las horas por tarea son positivas")
    ok(e["disponible"] and e["banda_baja"] < e["total_horas"] < e["banda_alta"],
       f"total {e['total_horas']:.1f} h dentro de su banda [{e['banda_baja']:.1f}, {e['banda_alta']:.1f}]")
    for h in H[:3]:
        print(f"      {h['historia_es']}")
        for t, x in zip(h["tareas_es"][:3], h["horas_probables"][:3]):
            print(f"         {x:5.2f} h  {t}")

    print("3 · /estimar")
    hist = [{"tareas": h["tareas"]} for h in H]
    dos = [8.0, 12.0] + [None] * (n - 2)
    f, _ = post("/estimar", {"historias": hist})
    f = f.json()["estimacion"]
    ok(abs(f["total_horas"] - e["total_horas"]) < 1e-6, f"en frío {f['total_horas']:.1f} h, igual que /backlog")
    c, _ = post("/estimar", {"historias": hist, "cerradas": dos})
    c = c.json()["estimacion"]
    rr, dt = post("/estimar", {"historias": hist, "respuestas": dos})
    rr = rr.json()["estimacion"]
    ok(abs(c["total_horas"] - rr["total_horas"]) < 1e-6,
       f"2 cerradas {c['total_horas']:.1f} h = 2 respuestas {rr['total_horas']:.1f} h ({dt:.1f} s)")
    ok(c.get("cerradas") == 2 and rr.get("respondidas") == 2 and rr.get("cerradas") == 0,
       f"cuenta: cerradas={c.get('cerradas')} · respondidas={rr.get('respondidas')}")
    p, _ = post("/estimar", {"historias": hist, "previo_equipo": 0.5})
    p = p.json()["estimacion"]
    ok(abs(p["total_horas"] / f["total_horas"] - math.exp(.5)) < 1e-3,
       f"previo +0,5: {p['total_horas']:.1f} h = frío x {p['total_horas'] / f['total_horas']:.4f} (e^0,5 = 1,6487)")
    m, _ = post("/estimar", {"historias": hist, "respuestas": [1.0, 2.0]})
    ok(m.status_code == 422, f"lista mal alineada: HTTP {m.status_code}")

    print("4 · /desplazamiento")
    horas = [x * 1.35 for h in H for x in h["horas_tareas"]]
    d, _ = post("/desplazamiento", {"historias": hist, "horas": horas})
    v = d.json().get("desplazamiento")
    ok(d.status_code == 200 and v is not None and math.isfinite(v), f"desplazamiento {v}")
    q, _ = post("/estimar", {"historias": hist, "previo_equipo": v})
    q = q.json()["estimacion"]
    ok(abs(q["total_horas"] - sum(horas)) / sum(horas) < 1e-3,
       f"usado como previo, el total es {q['total_horas']:.1f} h y las horas reales sumaban {sum(horas):.1f} h")

    print(f"\n{'TODO CORRECTO' if not fallos else f'{len(fallos)} FALLOS'} · salida en {OUT}/backlog.json")
    sys.exit(1 if fallos else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
