"""E-266 - ¿Que techo tiene el +-25 % del total en un backlog GENERADO?

E-264 midio que, aun con la escala exacta, un proyecto de 16 a 40 tareas solo cae
dentro de +-25 % el 60 % de las veces. Un backlog de Agix tiene unas 12 historias
con sus tareas, asi que cae justo ahi. Aqui se comprueba con el generador actual
(F:/Agix-produccion, tuberia del 07/11/2026):

  1 se generan backlogs para descripciones de prueba (GPU) y se guarda el JSON;
  2 el v7 les pone la media por tarea (CPU, la cabeza que se suma);
  3 se simulan las horas reales: media x residuo, con residuos remuestreados de
    las tareas reales del corpus ya sin la escala de su proyecto. Las tareas
    generadas no tienen horas reales, asi que la dispersion es la del corpus:
    es un supuesto, y se dice;
  4 se mide el +-25 % con la escala exacta (techo) y con el error de escala en
    frio de los 114 proyectos, y como cambia con el numero de tareas.

    python src/266_techo_backlog_generado.py            genera y simula
    python src/266_techo_backlog_generado.py --simular  reutiliza los backlogs
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/techo_backlog_generado")
PROD = Path("F:/Agix-produccion/generacion")
SIMS = 20000
NS = (12, 24, 36, 60, 100, 150, 250)


def descripciones():
    d = {}
    j = json.load(open("F:/Software Task generation Model/results/92_hu_rev.json", encoding="utf-8"))
    for k, v in j.items():
        d[k] = v["descripcion"]
    for x in json.load(open("results/prueba_extremo/salidas.json", encoding="utf-8")):
        if x["descripcion"] not in d.values():
            d[x["titulo"]] = x["descripcion"]
    return d


def generar():
    t0 = time.time()
    os.environ.setdefault("AGIX_BASE", str(PROD))
    s = importlib.util.spec_from_file_location("pl", PROD / "src" / "19_pipeline.py")
    pl = importlib.util.module_from_spec(s)
    s.loader.exec_module(pl)
    print(f"generador en {pl.DISPOSITIVO} · cargando...", flush=True)
    P = pl.Pipeline(traducir=True)
    print(f"  cargado · {time.time() - t0:.0f} s", flush=True)
    desc = descripciones()
    sal = []
    for i, (nombre, texto) in enumerate(desc.items(), 1):
        t = time.time()
        r = P(texto)
        tareas = [x for h in r["historias"] for x in h["tareas"]]
        sal.append(dict(nombre=nombre, descripcion=texto, historias=len(r["historias"]),
                        tareas=tareas, segundos=round(time.time() - t, 1)))
        print(f"  {i}/{len(desc)} {nombre}: {len(r['historias'])} historias · {len(tareas)} tareas · "
              f"{time.time() - t:.0f} s (total {time.time() - t0:.0f} s)", flush=True)
        json.dump(sal, open(OUT / "backlogs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return sal


def estimar(sal):
    t0 = time.time()
    s = importlib.util.spec_from_file_location("e6", "src/171_estimador_v6.py")
    e6 = importlib.util.module_from_spec(s)
    s.loader.exec_module(e6)
    est = e6.EstimadorV6(paquete=Path("models/estimador_agix_v7.joblib"), dispositivo="cpu")
    for b in sal:
        b["media_h"] = [float(x) for x in est._media_tarea(b["tareas"])]
    json.dump(sal, open(OUT / "backlogs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  v7 (media por tarea) sobre {sum(len(b['tareas']) for b in sal)} tareas · "
          f"{time.time() - t0:.0f} s", flush=True)
    return sal


def corpus():
    """Residuos por tarea sin la escala del proyecto, y error de escala en frio."""
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": sw.effort_hours.to_numpy(),
                      "est": np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= 8].index)]
    g = d.groupby("p")
    c = g.real.transform("sum") / g.est.transform("sum")
    u = (d.real / (c * d.est)).to_numpy()
    escala = np.log((g.real.sum() / g.est.sum()).to_numpy())
    return u, escala, d.est.to_numpy()


def simula(m, u, escala, rng):
    """m: medias por tarea de un backlog. Devuelve (techo, en frio)."""
    y = (m[None, :] * rng.choice(u, (SIMS, len(m)))).sum(1)
    pred = m.sum()
    techo = np.mean(np.abs(pred - y) / y <= .25)
    yf = y * np.exp(rng.choice(escala, SIMS))
    return float(techo), float(np.mean(np.abs(pred - yf) / yf <= .25))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if "--simular" in sys.argv:
        sal = json.load(open(OUT / "backlogs.json", encoding="utf-8"))
    else:
        sal = estimar(generar())
    u, escala, est_corpus = corpus()
    rng = np.random.default_rng(266)
    filas = []
    for b in sal:
        m = np.array(b["media_h"])
        techo, frio = simula(m, u, escala, rng)
        filas.append(dict(backlog=b["nombre"], historias=b["historias"], tareas=len(m),
                          total_h=float(m.sum()), cv_medias=float(m.std() / m.mean()),
                          techo_escala_exacta=techo, en_frio_simulado=frio))
    t = pd.DataFrame(filas)
    t.loc[len(t)] = ["MEDIA", t.historias.mean(), t.tareas.mean(), t.total_h.mean(), t.cv_medias.mean(),
                     t.techo_escala_exacta.mean(), t.en_frio_simulado.mean()]
    t.to_csv(OUT / "por_backlog.csv", index=False)
    print("\nPOR BACKLOG GENERADO")
    print(t.round(3).to_string(index=False))

    todas = np.concatenate([np.array(b["media_h"]) for b in sal])
    curva = []
    for n in NS:
        r = [simula(rng.choice(todas, n), u, escala, rng) for _ in range(30)]
        curva.append(dict(tareas=n, techo_escala_exacta=float(np.mean([x[0] for x in r])),
                          en_frio_simulado=float(np.mean([x[1] for x in r]))))
    c = pd.DataFrame(curva)
    c.to_csv(OUT / "curva_por_tareas.csv", index=False)
    print("\nSEGUN EL NUMERO DE TAREAS DEL BACKLOG (tareas con las medias del generador)")
    print(c.round(3).to_string(index=False))
    json.dump({"backlogs": len(sal), "simulaciones": SIMS,
               "cv_medias_generadas": float(todas.std() / todas.mean()),
               "cv_medias_corpus": float(est_corpus.std() / est_corpus.mean()),
               "segundos": round(time.time() - t0)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\ndispersion de las medias: generadas CV {todas.std() / todas.mean():.2f} · "
          f"corpus CV {est_corpus.std() / est_corpus.mean():.2f}")
    print(f"escrito {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
