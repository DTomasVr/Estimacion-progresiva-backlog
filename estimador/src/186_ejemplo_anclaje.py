"""E75 - Que cambia exactamente cuando el proyecto va cerrando tareas.

La pregunta que esto contesta: el modelo es estatico, asi que lo unico que puede
cambiar es lo que se hace con su salida. Aqui se ve en un proyecto real.

Los pesos del modelo NO se tocan. Para un texto dado, su prediccion por tarea es
siempre la misma. Lo que cambia es UN escalar por proyecto:

    off   = log( suma de horas REALES de las cerradas / suma de lo PREDICHO para ellas )
            x  k / (k + 12)
    total = suma de las reales  +  suma de (predicho de las abiertas x e^off)

Es decir: las cerradas dejan de estimarse (se saben) y las abiertas se multiplican
todas por el mismo factor e^off. La contraccion k/(k+12) evita fiarse de un cociente
calculado con dos tareas.

Se elige el proyecto de forma honesta -- el de error mediano entre los que tienen
suficientes tareas -- y se usa el ORDEN REAL de cierre si el proyecto lo tiene.

    python src/186_ejemplo_anclaje.py
    python src/186_ejemplo_anclaje.py NOMBRE_DEL_PROYECTO
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
OUT = Path("results/ejemplo_anclaje")
S = "=" * 104
LAMBDA, MIN_TAREAS, KS = 12, 20, (0, 1, 2, 3, 5, 10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    assert len(pred) == len(sw)
    tam = sw.groupby("project").size()
    k_ok = sw.project.isin(tam[tam >= 8].index).to_numpy()
    sw, pred = sw[k_ok].reset_index(drop=True), pred[k_ok]

    g = pd.DataFrame({"p": sw.project.to_numpy(), "real": sw.effort_hours.to_numpy(),
                      "est": pred}).groupby("p")
    tot = pd.DataFrame({"real": g.real.sum(), "est": g.est.sum(), "n": g.size()})
    tot["err"] = (tot.est - tot.real).abs() / tot.real

    if len(sys.argv) > 1:
        proy = sys.argv[1]
    else:
        # el de error MEDIANO entre los que tienen bastantes tareas: ni el mejor
        # ni el peor, para que el ejemplo no sea una seleccion interesada
        cand = tot[tot.n >= MIN_TAREAS].sort_values("err")
        proy = cand.index[len(cand) // 2]
    fila = tot.loc[proy]
    i = np.flatnonzero(sw.project.to_numpy() == proy)
    d = pd.DataFrame({"titulo": sw.title.fillna("").to_numpy()[i],
                      "real": sw.effort_hours.to_numpy()[i], "est": pred[i],
                      "cierre": sw.completed.to_numpy()[i]})

    # orden de cierre: el REAL si lo hay, y si no un sorteo con semilla fija
    if d.cierre.notna().mean() >= .8:
        orden = np.argsort(np.where(pd.isna(d.cierre),
                                    np.datetime64("2999-01-01"), d.cierre),
                           kind="stable")
        fuente = "orden REAL de cierre (el dataset guarda la fecha)"
    else:
        orden = np.random.default_rng(7).permutation(len(d))
        fuente = "sorteo con semilla fija (este proyecto no guarda fecha de cierre)"

    print(f"{S}\nPROYECTO: {proy}\n{S}")
    print(f"  {int(fila.n)} tareas · {fila.real:.0f} h reales · "
          f"{fila.est:.0f} h estimadas en frio · error {fila['err']:.1%}")
    print(f"  tareas cerradas tomadas en {fuente}")

    # una tarea concreta que siga ABIERTA en todos los casos: la ultima en cerrarse
    j_obs = int(orden[-1])
    obs = d.iloc[j_obs]
    print(f"\n  tarea que se sigue (la ultima en cerrarse, abierta en todos los casos):")
    print(f"    «{obs.titulo[:88]}»")
    print(f"    el modelo predice {obs.est:.2f} h · real {obs.real:.2f} h")

    filas = []
    for k in KS:
        sel = orden[:k]
        abiertas = np.setdiff1d(np.arange(len(d)), sel)
        if k:
            off = np.log(d.real.to_numpy()[sel].sum() / d.est.to_numpy()[sel].sum())
            off *= k / (k + LAMBDA)
        else:
            off = 0.
        f = float(np.exp(off))
        total = float(d.real.to_numpy()[sel].sum() +
                      (d.est.to_numpy()[abiertas] * f).sum())
        filas.append(dict(cerradas=k, factor=f,
                          horas_tarea=float(obs.est) * f,
                          total=total, real=float(fila.real),
                          error_pc=abs(total - fila.real) / fila.real * 100))
    t = pd.DataFrame(filas)

    print(f"\n{S}\nLO QUE CAMBIA SEGUN CUANTAS TAREAS LLEVE CERRADAS EL PROYECTO\n{S}")
    print(f"{'cerradas':>9} {'factor':>8} {'horas de esa tarea':>20} "
          f"{'total estimado':>15} {'total real':>11} {'error':>8}")
    for _, r in t.iterrows():
        print(f"{int(r.cerradas):>9} {r.factor:>8.3f} "
              f"{r.horas_tarea:>17.2f} h {r.total:>13.0f} h {r.real:>9.0f} h "
              f"{r.error_pc:>7.1f}%")
    print(f"\n  la prediccion CRUDA del modelo para esa tarea es {obs.est:.2f} h en las "
          f"seis filas:\n  lo unico que cambia es el factor que la multiplica.")
    t.insert(0, "proyecto", proy)
    t.to_csv(OUT / "resumen.csv", index=False)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
