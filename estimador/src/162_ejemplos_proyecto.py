"""E56 - Ejemplos concretos, que es lo que se ensena en una exposicion.

Las tablas de metricas convencen a un jurado tecnico; un ejemplo concreto
convence a todo el mundo. Se eligen tres proyectos reales de los 114 -- uno
pequeno, uno mediano y uno grande -- y se muestra lo que el modelo diria de ellos
en frio, con su intervalo, y unas cuantas tareas sueltas con lo que estimo y lo
que costaron.

Los proyectos se eligen por TAMANO, no por lo bien que salen, y se dice cuanto se
equivoca en cada uno. Un ejemplo escogido por bonito no vale para nada.
"""
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
pd.set_option("display.width", 220)
OUT = Path("results/ejemplos_proyecto")
S = "=" * 116
MIN_TAREAS = 8


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "titulo": sw.title.fillna("").astype(str),
                      "real": sw.effort_hours.to_numpy(), "est": est,
                      "fuente": sw.source.to_numpy()})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    g = d.groupby("p")
    tot = pd.DataFrame({"real": g.real.sum(), "est": g.est.sum(), "n": g.size()})
    q = np.log(tot.real / tot.est).to_numpy()
    estr = np.digitize(tot.n.to_numpy(), [21, 61, 201])
    lo, hi = np.zeros(len(tot)), np.zeros(len(tot))
    for i in range(len(tot)):
        k = estr == estr[i]
        k[i] = False
        o = q[k] if k.sum() >= 12 else np.delete(q, i)
        lo[i] = tot.est.iloc[i] * np.exp(np.quantile(o, .1))
        hi[i] = tot.est.iloc[i] * np.exp(np.quantile(o, .9))
    tot["banda_baja"], tot["banda_alta"] = lo, hi
    tot["error_pc"] = (tot.est / tot.real - 1) * 100

    # Uno de cada tramo de tamano, y dentro del tramo el de ERROR MEDIANO: la
    # mitad de los proyectos de ese tramo salen mejor y la mitad peor. Elegir por
    # numero de tareas daba ejemplos arbitrarios (salio uno con +485 %) y elegir
    # el que mejor sale seria hacer trampa. El error mediano es el representativo.
    elegidos = []
    for a, b, etq in ((8, 25, "pequeno"), (26, 90, "mediano"), (91, 1e9, "grande")):
        c = tot[(tot.n >= a) & (tot.n <= b)]
        if not len(c):
            continue
        elegidos.append((etq, c.index[np.argsort(c.error_pc.abs().to_numpy())[len(c) // 2]]))

    print(f"{S}\nTRES PROYECTOS, EL DE ERROR MEDIANO DE CADA TRAMO DE TAMANO\n{S}")
    print("  (la mitad de los proyectos de su tramo salen mejor y la mitad peor)")
    filas = []
    for etq, p in elegidos:
        r = tot.loc[p]
        filas.append(dict(tamano=etq, proyecto=str(p)[:22], tareas=int(r.n),
                          estimado_h=round(r.est), real_h=round(r.real),
                          error=f"{r.error_pc:+.0f} %",
                          banda_80=f"{r.banda_baja:.0f} a {r.banda_alta:.0f} h",
                          dentro="si" if r.banda_baja <= r.real <= r.banda_alta else "NO"))
    e = pd.DataFrame(filas)
    print(e.to_string(index=False))

    p = elegidos[1][1] if len(elegidos) > 1 else elegidos[0][1]
    sub = d[d.p == p].copy()
    sub["razon"] = sub.est / sub.real
    sub = sub.reindex(sub.razon.sub(1).abs().sort_values().index)
    muestra = pd.concat([sub.head(4), sub.tail(3)])
    print(f"\n{S}\nTAREAS DEL PROYECTO '{p}' · las 4 que mejor y las 3 que peor\n{S}")
    for _, r in muestra.iterrows():
        print(f"  estimado {r.est:6.2f} h · real {r.real:6.2f} h · "
              f"x{r.razon:5.2f}  {r.titulo[:66]}")
    err_t = float(np.median(np.abs(sub.razon - 1))) * 100
    err_p = abs(sub.est.sum() / sub.real.sum() - 1) * 100
    print(f"\n  suma estimada {sub.est.sum():.0f} h · suma real {sub.real.sum():.0f} h "
          f"· error del total {err_p:.0f} %")
    print(f"  el error TIPICO de una tarea suelta de este proyecto es del {err_t:.0f} %")
    print(f"  y el del total es del {err_p:.0f} %: esa diferencia es la cancelacion de")
    print("  errores, que es de lo que vive esta metrica")

    e.to_csv(OUT / "proyectos.csv", index=False)
    muestra[["titulo", "est", "real", "razon"]].to_csv(OUT / "tareas.csv", index=False)
    tot.to_csv(OUT / "todos.csv")
    print(f"\nescrito {OUT}/proyectos.csv")


if __name__ == "__main__":
    main()
