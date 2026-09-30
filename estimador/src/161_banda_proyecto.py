"""E55 - Un intervalo para el total del proyecto, no solo un numero.

"El proyecto son 640 horas" es una frase que nadie deberia decir con un MdMRE de
0,37. "Entre 380 y 1.080 horas, con un 80 % de confianza" si se puede decir, y
ademas es lo que un jefe de proyecto necesita para negociar un plazo.

Con 114 proyectos se puede calibrar el intervalo por prediccion conforme, que da
cobertura garantizada sin suponer nada sobre la forma de la distribucion, solo
que los proyectos son intercambiables. El corte se estima DEJANDO FUERA el
proyecto que se evalua, para que la cobertura que se reporta sea real y no el
resultado de haberse mirado a si mismo.

Tres cosas que hay que medir, no solo la cobertura:

  ANCHURA     un intervalo que cubre el 80 % siendo de x100 no sirve de nada. Se
              reporta en factor multiplicativo, que es como se lee de verdad
  COBERTURA   marginal y tambien por TAMANO de proyecto: si cubre de mas a los
              pequenos y de menos a los grandes, la garantia global esconde que
              en los grandes falla
  CON HISTORIAL  como se estrecha el intervalo segun el equipo cierra tareas, que
              es el argumento de producto

Se compara con un intervalo de anchura fija, que es el control: si el intervalo
adaptativo no es mas estrecho a igual cobertura, no esta aportando nada.
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
OUT = Path("results/banda_proyecto")
S = "=" * 116
MIN_TAREAS, SORTEOS, LAMBDA = 8, 40, 12
NIVELES = (.5, .8, .9)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    proy = sw.project.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": proy, "real": y, "est": est})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    nombres = list(pd.unique(d.p))
    idx = {p: np.flatnonzero(d.p.to_numpy() == p) for p in nombres}
    g = d.groupby("p")
    real = g.real.sum().reindex(nombres).to_numpy()
    n_t = g.size().reindex(nombres).to_numpy()
    le, re_, es_ = np.log(d.est.to_numpy()), d.real.to_numpy(), d.est.to_numpy()

    escenarios = {"en frio": g.est.sum().reindex(nombres).to_numpy()}
    rng = np.random.default_rng(0)
    for k in (3, 10):
        ac = np.zeros(len(nombres))
        for _ in range(SORTEOS):
            t = []
            for p in nombres:
                i = idx[p]
                if len(i) <= k:
                    t.append(float(es_[i].sum()))
                    continue
                s = rng.choice(i, k, replace=False)
                off = np.log(re_[s].sum() / es_[s].sum()) * k / (k + LAMBDA)
                t.append(float(np.exp(le[i] + off).sum()))
            ac += np.array(t)
        escenarios[f"con {k} tareas cerradas"] = ac / SORTEOS
    print(f"{len(nombres)} proyectos")

    filas, cond = [], []
    for esc, pred in escenarios.items():
        q = np.log(real / pred)                 # residuo en log
        for niv in NIVELES:
            lo = np.zeros(len(real))
            hi = np.zeros(len(real))
            ancho_fijo = np.zeros(len(real))
            for i in range(len(real)):
                o = np.delete(q, i)
                a = (1 - niv) / 2
                # conforme adaptativo: cuantiles del residuo, que se ensanchan
                # segun el propio residuo, no segun el tamano
                lo[i] = pred[i] * np.exp(np.quantile(o, a))
                hi[i] = pred[i] * np.exp(np.quantile(o, 1 - a))
                n = len(o)
                c = np.sort(np.abs(o))[min(int(np.ceil((n + 1) * niv)), n) - 1]
                ancho_fijo[i] = c
            dentro = (real >= lo) & (real <= hi)
            fij = (real >= pred * np.exp(-ancho_fijo)) & (real <= pred * np.exp(ancho_fijo))
            filas.append(dict(escenario=esc, nivel=f"{niv:.0%}", metodo="cuantiles del residuo",
                              cobertura=float(dentro.mean()),
                              anchura_x=float(np.median(hi / lo))))
            filas.append(dict(escenario=esc, nivel=f"{niv:.0%}", metodo="simetrico (control)",
                              cobertura=float(fij.mean()),
                              anchura_x=float(np.median(np.exp(2 * ancho_fijo)))))
            # conforme POR TRAMO DE TAMANO (conforme de Mondrian). El error del
            # total se cancela con la raiz del numero de tareas, asi que un
            # proyecto de 300 tareas no merece el mismo intervalo que uno de 10:
            # con el intervalo unico, los grandes salen cubiertos al 96 % y los
            # pequenos al 70 %, o sea que la garantia global esconde las dos cosas.
            estr = np.digitize(n_t, [21, 61, 201])
            lo2 = np.zeros(len(real))
            hi2 = np.zeros(len(real))
            for i in range(len(real)):
                k = (estr == estr[i])
                k[i] = False
                o = q[k] if k.sum() >= 12 else np.delete(q, i)
                a = (1 - niv) / 2
                lo2[i] = pred[i] * np.exp(np.quantile(o, a))
                hi2[i] = pred[i] * np.exp(np.quantile(o, 1 - a))
            den = (real >= lo2) & (real <= hi2)
            filas.append(dict(escenario=esc, nivel=f"{niv:.0%}",
                              metodo="por tramo de tamano",
                              cobertura=float(den.mean()),
                              anchura_x=float(np.median(hi2 / lo2))))
            if niv == .8 and esc == "en frio":
                for tr in np.unique(estr):
                    k = estr == tr
                    cond.append(dict(tramo=["8-20", "21-60", "61-200", "mas de 200"][tr],
                                     metodo="por tramo", proyectos=int(k.sum()),
                                     cobertura=float(den[k].mean()),
                                     anchura_x=float(np.median((hi2 / lo2)[k]))))
            if niv == .8 and esc == "en frio":
                t = pd.DataFrame({"n": n_t, "cub": dentro,
                                  "anch": hi / lo,
                                  "tramo": pd.cut(n_t, [7, 20, 60, 200, 100000],
                                                  labels=["8-20", "21-60", "61-200",
                                                          "mas de 200"])})
                for tr, gg in t.groupby("tramo", observed=True):
                    cond.append(dict(tramo=str(tr), metodo="intervalo unico",
                                     proyectos=len(gg),
                                     cobertura=float(gg.cub.mean()),
                                     anchura_x=float(gg.anch.median())))
    f = pd.DataFrame(filas)
    for esc in escenarios:
        print(f"\n{S}\n{esc}\n{S}")
        print(f[f.escenario == esc].drop(columns=["escenario"]).round(3).to_string(index=False))
    print(f"\n{S}\nCOBERTURA AL 80 % POR TAMANO DE PROYECTO (en frio)\n{S}")
    c = pd.DataFrame(cond).sort_values(["metodo", "tramo"])
    print(c.round(3).to_string(index=False))

    v = f[(f.escenario == "en frio") & (f.nivel == "80%") &
          (f.metodo == "cuantiles del residuo")].iloc[0]
    v10 = f[(f.escenario == "con 10 tareas cerradas") & (f.nivel == "80%") &
            (f.metodo == "cuantiles del residuo")].iloc[0]
    ej = int(np.argsort(n_t)[len(n_t) // 2])
    q = np.log(real / escenarios["en frio"])
    lo = escenarios["en frio"][ej] * np.exp(np.quantile(np.delete(q, ej), .1))
    hi = escenarios["en frio"][ej] * np.exp(np.quantile(np.delete(q, ej), .9))
    print(f"\n{S}\nLECTURA\n{S}")
    print(f"  al 80 %, en frio: cubre el {v.cobertura:.1%} con un intervalo de "
          f"x{v.anchura_x:.1f}")
    print(f"  al 80 %, con 10 tareas cerradas: {v10.cobertura:.1%} con x{v10.anchura_x:.1f}")
    print(f"\n  ejemplo con un proyecto de tamano mediano ({n_t[ej]} tareas):")
    print(f"    estimacion puntual {escenarios['en frio'][ej]:.0f} h · "
          f"intervalo al 80 % de {lo:.0f} a {hi:.0f} h · real {real[ej]:.0f} h")
    f.to_csv(OUT / "resumen.csv", index=False)
    c.to_csv(OUT / "por_tamano.csv", index=False)
    json.dump({"niveles": list(NIVELES), "min_tareas": MIN_TAREAS},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
