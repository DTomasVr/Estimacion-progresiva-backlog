"""Ablacion de los dos filtros nuevos, sobre la MISMA poblacion de entradas.

Los dos filtros (validador del hueco del rol y anti-eco en las tareas) se pueden
apagar con PBE_SIN_FILTROS=1, asi que se puede medir lo que aportan sin cambiar
ni el modelo ni las entradas. Esto da el "antes" de la tabla y separa dos cosas
que conviene no confundir:

  - cuanto del defecto se arregla FILTRANDO en inferencia (gratis, hoy)
  - cuanto se arregla con DATOS de entrenamiento mejores (la tanda v3)

  PBE_DEVICE=cpu python src/58_ablacion_filtros.py [n]

out: results/58_ablacion.csv
"""
import collections
import importlib.util
import json
import os
import subprocess
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = next((int(a) for a in sys.argv[1:] if a.isdigit()), 40)


# El defecto del ROL no se ve en las descripciones reservadas del profesor:
# son de sectores que el modelo domina y nombran un publico que SI es un rol
# valido, asi que el modelo lo copia y acierta. Aparece en descripciones de
# producto de fuera de la taxonomia, que es lo que de verdad llega a PBE.
CONSUMO = [
    "A supermarket app where customers browse the catalogue, build a basket and book a delivery slot.",
    "An online store for selling audiobooks and ebooks, with a library and offline listening.",
    "A gym management app with memberships, class booking and a check-in kiosk.",
    "A veterinary clinic system for appointments, vaccination records and reminders.",
    "A hotel booking platform with room availability, seasonal rates and guest reviews.",
    "A food delivery app connecting restaurants, couriers and customers with live tracking.",
    "A school platform where teachers publish assignments and parents follow grades.",
    "A car workshop system for service appointments, repair quotes and spare part stock.",
    "A recycling collection service with household pickup scheduling and weight reports.",
    "A microloan platform where borrowers apply and analysts review credit risk.",
    "An app to find and pay for street parking, with a timer and monthly invoices.",
    "A wedding planning tool to manage the guest list, seating chart and vendor payments.",
]


def entradas(n):
    """las descripciones reservadas del profesor: nunca entrenadas"""
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_mixa_test.parquet")
    ids = {str(x).split("#")[0] for x in d.parent_uid.dropna()
           if not str(x).startswith("Apache")}
    B = [json.loads(l) for l in open(f"{BASE}/data/synth/backlogs_qwen2.5_7b.jsonl",
                                     encoding="utf-8")]
    B = [b for b in B if b["id"] in ids][:n]
    return [(b["sector"], b["descripcion"]) for b in B]


def una_pasada(sin_filtros, casos):
    """subproceso: FILTROS se lee al importar el pipeline, no se puede cambiar en vivo"""
    env = dict(os.environ)
    env["PBE_SIN_FILTROS"] = "1" if sin_filtros else "0"
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run([sys.executable, f"{BASE}/src/58_ablacion_filtros.py", "--hijo"],
                       input=json.dumps(casos), capture_output=True, text=True, env=env,
                       encoding="utf-8")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("fallo la pasada hija")
    return json.loads(r.stdout.split("###JSON###")[1])


def hijo():
    casos = json.loads(sys.stdin.read())
    spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
    pl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pl)
    for v, at in [("PBE_ARRIBA", "M_ARRIBA"), ("PBE_ABAJO", "M_ABAJO")]:
        if os.environ.get(v):
            setattr(pl, at, os.environ[v])
    P = pl.Pipeline(traducir=False)
    out = []
    for sector, desc in casos:
        r = P(desc, k_historias=12, n_muestras=36)
        out.append(dict(sector=sector, descripcion=desc,
                        historias=[dict(h=x["historia"], t=list(x["tareas"]))
                                   for x in r["historias"]]))
        print(f"  {sector}", flush=True)
    print("###JSON###" + json.dumps(out, ensure_ascii=False))


def resumen(nombre, salidas):
    spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
    # se cargan solo las funciones de texto, sin modelos
    import re
    src = open(f"{BASE}/src/19_pipeline.py", encoding="utf-8").read()
    ns = {"re": re, "json": json, "os": os, "BASE": BASE}
    ns["os"] = os
    entorno = dict(os.environ)
    os.environ["PBE_SIN_FILTROS"] = "0"      # para juzgar siempre con el mismo criterio
    exec(src[src.index("def separar_tareas"):src.index("class Pipeline")], ns)
    os.environ.clear()
    os.environ.update(entorno)
    nh = nrol = nt = ndeg = 0
    roles = collections.Counter()
    tareas_por_hist = []
    for s in salidas:
        for x in s["historias"]:
            nh += 1
            r = ns["rol_de"](x["h"])
            roles[r] += 1
            nrol += ns["rol_valido"](x["h"])
            tareas_por_hist.append(len(x["t"]))
            for t in x["t"]:
                nt += 1
                ndeg += ns["tarea_degenerada"](t, x["h"])
    print(f"\n--- {nombre}")
    print(f"  historias {nh} | rol valido {100*nrol/max(1,nh):.1f} %")
    print(f"  tareas {nt} | degeneradas {100*ndeg/max(1,nt):.1f} % | "
          f"{sum(tareas_por_hist)/max(1,len(tareas_por_hist)):.1f} por historia")
    print(f"  roles distintos {len(roles)} | invalidos mas frecuentes: "
          + ", ".join(f"{k} ({v})" for k, v in roles.most_common()
                      if not ns["rol_valido"](f"As a {k}, I want x, so that y.")) [:180])
    return dict(caso=nombre, historias=nh, rol_valido=nrol / max(1, nh),
                tareas=nt, degeneradas=ndeg / max(1, nt),
                tareas_por_hist=sum(tareas_por_hist) / max(1, len(tareas_por_hist)),
                roles_distintos=len(roles))


if __name__ == "__main__":
    if "--hijo" in sys.argv:
        hijo()
        sys.exit()
    pob = os.environ.get("PBE_POB", "consumo")
    casos = ([("consumo", d) for d in CONSUMO[:N]] if pob == "consumo"
             else entradas(N) if pob == "reservadas"
             else [("consumo", d) for d in CONSUMO] + entradas(N))
    print(f"poblacion {pob}: {len(casos)} descripciones, dispositivo "
          f"{os.environ.get('PBE_DEVICE', 'cuda')}")
    filas = []
    for etiqueta, sin in [("sin filtros (el defecto crudo)", True),
                          ("con filtros de inferencia", False)]:
        print(f"\n=== {etiqueta}", flush=True)
        S = una_pasada(sin, casos)
        json.dump(S, open(f"{BASE}/results/58_salidas_{'sin' if sin else 'con'}.json",
                          "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        filas.append(resumen(etiqueta, S))
    D = pd.DataFrame(filas)
    D.to_csv(f"{BASE}/results/58_ablacion.csv", index=False)
    print("\n" + D.to_string(index=False))
    print("-> results/58_ablacion.csv y results/58_salidas_{sin,con}.json")
