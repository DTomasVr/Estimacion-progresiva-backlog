"""E55 - Un intervalo para el total del proyecto, no solo un numero.

"El proyecto son 640 horas" es una frase que nadie deberia decir con un MdMRE de
0,36. "Entre 380 y 1.080 horas, con un 80 % de confianza" si se puede decir, y
ademas es lo que un jefe de proyecto necesita para negociar un plazo.

Con 114 proyectos se calibra el intervalo por prediccion conforme, que da
cobertura garantizada sin suponer nada sobre la forma de la distribucion, solo
que los proyectos son intercambiables. El corte se estima DEJANDO FUERA el
proyecto que se evalua.

Tres cosas que hay que mirar, no solo la cobertura:

  ANCHURA     un intervalo que cubre el 80 % siendo de x100 no sirve de nada. Se
              da en factor multiplicativo, que es como se lee
  REPARTO     por tamano de proyecto. Si cubre de mas a los pequenos y de menos a
              los grandes, la cobertura global esconde las dos cosas
  HISTORIAL   como se estrecha segun el equipo cierra tareas

Con tareas cerradas se aplica lo mismo que en E53: las horas de lo ya cerrado se
SABEN y no se estiman, y la metrica se promedia entre sorteos en vez de promediar
las predicciones, que daba una cobertura falsamente buena.
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "8"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
OUT = Path("results/banda_proyecto")
S = "=" * 116
MIN_TAREAS, SORTEOS, LAMBDA = 8, 30, 12
NIVELES = (.5, .8, .9)
ETIQ = ["8-20", "21-60", "61-200", "mas de 200"]


def bandas(pred, real, q, niv, grupos=None):
    """Intervalos conformes, dejando fuera el proyecto que se evalua."""
    a = (1 - niv) / 2
    lo = np.zeros(len(real))
    hi = np.zeros(len(real))
    for i in range(len(real)):
        k = np.ones(len(real), bool)
        k[i] = False
        if grupos is not None:
            kk = k & (grupos == grupos[i])
            if kk.sum() >= 12:
                k = kk
        lo[i] = pred[i] * np.exp(np.quantile(q[k], a))
        hi[i] = pred[i] * np.exp(np.quantile(q[k], 1 - a))
    return lo, hi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": est})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    nom = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in nom}
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    le = np.log(es_)
    g = d.groupby("p")
    real = g.real.sum().reindex(nom).to_numpy()
    n_t = g.size().reindex(nom).to_numpy()
    estr = np.digitize(n_t, [21, 61, 201])

    esc = {"en frio": [g.est.sum().reindex(nom).to_numpy()]}
    for k in (3, 10):
        rng = np.random.default_rng(2000 + k)
        M = []
        for _ in range(SORTEOS):
            t = []
            for p in nom:
                i = idx[p]
                if len(i) <= k:
                    t.append(float(es_[i].sum()))
                    continue
                s = rng.choice(i, k, replace=False)
                resto = np.setdiff1d(i, s)
                off = np.log(re_[s].sum() / es_[s].sum()) * k / (k + LAMBDA)
                t.append(float(re_[s].sum() + np.exp(le[resto] + off).sum()))
            M.append(np.array(t))
        esc[f"con {k} tareas cerradas"] = M
    print(f"{len(nom)} proyectos · {SORTEOS} sorteos", flush=True)

    filas, cond = [], []
    for nombre, sorteos in esc.items():
        for niv in NIVELES:
            acc = {"unico": [], "por tramo": []}
            anc = {"unico": [], "por tramo": []}
            porT, porU = [], []
            for pred in sorteos:
                q = np.log(real / pred)
                for etq, gr in (("unico", None), ("por tramo", estr)):
                    lo, hi = bandas(pred, real, q, niv, gr)
                    cub = (real >= lo) & (real <= hi)
                    acc[etq].append(float(cub.mean()))
                    anc[etq].append(float(np.median(hi / lo)))
                    if niv == .8 and nombre == "en frio":
                        fila = [(float(cub[estr == s].mean()),
                                 float(np.median((hi / lo)[estr == s])))
                                for s in range(4)]
                        (porT if etq == "por tramo" else porU).append(fila)
            for etq in acc:
                filas.append(dict(escenario=nombre, nivel=f"{niv:.0%}", metodo=etq,
                                  cobertura=float(np.mean(acc[etq])),
                                  sd_cobertura=float(np.std(acc[etq])),
                                  anchura_x=float(np.mean(anc[etq]))))
            if niv == .8 and nombre == "en frio":
                for etq, dat in (("intervalo unico", porU), ("por tramo", porT)):
                    a = np.array(dat).mean(0)
                    for s in range(4):
                        cond.append(dict(tramo=ETIQ[s], metodo=etq,
                                         proyectos=int((estr == s).sum()),
                                         cobertura=float(a[s, 0]),
                                         anchura_x=float(a[s, 1])))
        print(f"  {nombre} listo", flush=True)

    f = pd.DataFrame(filas)
    for nombre in esc:
        print(f"\n{S}\n{nombre}\n{S}")
        print(f[f.escenario == nombre].drop(columns=["escenario"]).round(3)
              .to_string(index=False))
    c = pd.DataFrame(cond).sort_values(["metodo", "tramo"])
    print(f"\n{S}\nCOBERTURA AL 80 % POR TAMANO DE PROYECTO (en frio)\n{S}")
    print(c.round(3).to_string(index=False))

    v = f[(f.escenario == "en frio") & (f.nivel == "80%") & (f.metodo == "unico")].iloc[0]
    v10 = f[(f.escenario == "con 10 tareas cerradas") & (f.nivel == "80%") &
            (f.metodo == "unico")].iloc[0]
    pred0 = esc["en frio"][0]
    q = np.log(real / pred0)
    ej = int(np.argsort(n_t)[len(n_t) // 2])
    lo, hi = bandas(pred0, real, q, .8, None)
    print(f"\n{S}\nLECTURA\n{S}")
    print(f"  al 80 %, en frio: cubre el {v.cobertura:.1%} con anchura x{v.anchura_x:.1f}")
    print(f"  al 80 %, con 10 tareas cerradas: {v10.cobertura:.1%} con "
          f"x{v10.anchura_x:.1f}")
    print(f"\n  ejemplo con un proyecto de tamano mediano ({n_t[ej]} tareas):")
    print(f"    estimacion puntual {pred0[ej]:.0f} h · intervalo al 80 % de "
          f"{lo[ej]:.0f} a {hi[ej]:.0f} h · real {real[ej]:.0f} h")
    f.to_csv(OUT / "resumen.csv", index=False)
    c.to_csv(OUT / "por_tamano.csv", index=False)
    json.dump({"niveles": list(NIVELES), "sorteos": SORTEOS},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
