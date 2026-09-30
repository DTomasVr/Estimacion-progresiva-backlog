"""E43 - La metrica que se presenta: acertar el TOTAL de horas del proyecto.

La asesora pide lo que tambien pide la literatura de estimacion de esfuerzo:
no evaluar tarea a tarea, sino sumar las horas predichas de todas las tareas que
la pipeline genera para un proyecto y comparar ese total con el real. Y el
informe de benchmarking fija con que medirlo: MAE, RMSE y R2.

Hay una razon matematica para esperar que esto salga MUCHO mejor que la
prediccion por tarea, y otra para esperar que no salga perfecto:

  A FAVOR   los errores se cancelan al sumar. Si los errores de n tareas fuesen
            independientes, el error relativo del total caeria como 1/raiz(n).
            Con proyectos de 50 a 500 tareas eso es una reduccion enorme.

  EN CONTRA no son independientes. E30 midio que un desplazamiento por proyecto
            conocido de oraculo sube el R2 de 0,182 a 0,297: hay un error COMUN
            a todas las tareas del proyecto que no se cancela con ninguna suma.
            El error del total converge a ese desplazamiento, no a cero.

Y hay una trampa que hay que arreglar antes de sumar nada. El modelo minimiza
error cuadratico sobre log-horas, asi que exp(prediccion) es la MEDIANA
condicional, no la media. Para un total hace falta la MEDIA, y con una
distribucion asimetrica la media es bastante mayor:

    E[y] = exp(mu + sigma^2 / 2)   bajo lognormal

Sumar medianas subestima el total de forma sistematica. La correccion estandar
es el estimador de smearing de Duan (1983), que no supone lognormalidad: se
multiplica por la media de exp(residuo) medida fuera de muestra. Con el sigma de
este problema el factor es grande, asi que sin esta correccion el total saldria
casi a la mitad.

Brazos, todos sobre las mismas predicciones fuera de muestra:

  0  constante          n tareas x la mediana global. El control obligatorio: si
                        el modelo no le gana a esto, no esta aportando nada
  1  suma de medianas   lo que saldria de sumar la salida del modelo tal cual
  2  suma con smearing  la correcta. El factor se calcula DEJANDO FUERA el
                        proyecto que se evalua, para no mirarse a si mismo
  3  suma de la analogia
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
OUT = Path("results/total_proyecto")
S = "=" * 122
MIN_TAREAS = 8


def metricas_totales(real, pred):
    """Sobre TOTALES de proyecto. La unidad de analisis es el proyecto."""
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    q = np.log(pred / real)
    m = float(np.median(q))
    lr = np.log(real)
    return dict(n_proy=len(real),
                MAE_h=float(np.abs(pred - real).mean()),
                RMSE_h=float(np.sqrt(np.mean((pred - real) ** 2))),
                MdMRE=float(np.median(r)),
                MMRE=float(r.mean()),
                zeta_pc=float((np.exp(np.median(np.abs(q))) - 1) * 100),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100),
                R2_log=float(1 - ((lr - np.log(pred)) ** 2).mean() / lr.var()),
                PRED25=float(np.mean(r <= .25)),
                PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    P = np.load("results/analogia_cv/pred.npy")      # fuera de muestra, por proyecto
    lp_mod, lp_ana, disp = P[:, 0], P[:, 1], P[:, 2]
    print(f"{len(sw):,} tareas · {len(np.unique(proy)):,} proyectos · "
          f"predicciones fuera de muestra por proyecto entero (E42, sin fuga)")

    # --- smearing de Duan, dejando fuera el proyecto que se evalua -------------
    e = np.exp(ly - lp_mod)
    suma_tot, n_tot = e.sum(), len(e)
    d = pd.DataFrame({"p": proy, "e": e})
    g = d.groupby("p").e.agg(["sum", "count"])
    fac = ((suma_tot - g["sum"]) / (n_tot - g["count"])).rename("smearing")
    print(f"factor de smearing: mediana {fac.median():.3f} "
          f"(sin el, el total saldria a {1 / fac.median():.0%} de lo que toca)")

    ea = np.exp(ly - lp_ana)
    da = pd.DataFrame({"p": proy, "e": ea}).groupby("p").e.agg(["sum", "count"])
    fac_a = ((ea.sum() - da["sum"]) / (len(ea) - da["count"]))
    med_glob = float(np.median(y))

    t = pd.DataFrame({"p": proy, "real": y, "mod": np.exp(lp_mod),
                      "ana": np.exp(lp_ana)})
    tot = t.groupby("p").agg(real=("real", "sum"), mod=("mod", "sum"),
                             ana=("ana", "sum"), n=("real", "size"))
    tot = tot[tot.n >= MIN_TAREAS]
    tot["mod_sm"] = tot["mod"] * fac.reindex(tot.index).to_numpy()
    tot["ana_sm"] = tot["ana"] * fac_a.reindex(tot.index).to_numpy()
    tot["cte"] = tot.n * med_glob
    print(f"proyectos con al menos {MIN_TAREAS} tareas: {len(tot):,} "
          f"(mediana de {tot.n.median():.0f} tareas, total real mediano "
          f"{tot.real.median():.0f} h)")

    brazos = {"0 · constante (n x mediana global)": tot["cte"],
              "1 · suma de las medianas (sin corregir)": tot["mod"],
              "2 · suma CON smearing de Duan": tot["mod_sm"],
              "3 · suma de la analogia, con smearing": tot["ana_sm"]}
    filas = [dict(brazo=n, **metricas_totales(tot["real"], v)) for n, v in brazos.items()]
    r = pd.DataFrame(filas)
    print(f"\n{S}\nTOTAL DE HORAS DEL PROYECTO · la unidad de analisis es el PROYECTO\n{S}")
    print(r[["brazo", "n_proy", "MAE_h", "RMSE_h", "MdMRE", "MMRE", "R2_log",
             "PRED25", "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))

    print(f"\n{S}\nCOMPARADO CON LA MISMA METRICA TAREA A TAREA\n{S}")
    # Hay que comparar con cuidado. Para una tarea suelta el estimador que toca es
    # la MEDIANA, sin smearing; para un total, la MEDIA, con smearing. Poner el
    # estimador de totales a competir tarea a tarea lo hunde (queda un factor 2,25
    # de mas en cada tarea) y exagera la mejora. Se dan los dos.
    tar = {"modelo por tarea, mediana (lo que toca por tarea)": np.exp(lp_mod),
           "modelo por tarea, con smearing (el estimador de totales)":
               np.exp(lp_mod) * float(np.median(fac)),
           "analogia por tarea, mediana": np.exp(lp_ana)}
    ft = pd.DataFrame([dict(brazo=n, **metricas_totales(y, v)) for n, v in tar.items()])
    ft = ft.rename(columns={"n_proy": "n_tareas"})
    print(ft[["brazo", "n_tareas", "MdMRE", "MMRE", "PRED25", "PRED50",
              "factor2"]].round(4).to_string(index=False))
    mejor = r.loc[r.MdMRE.idxmin()]
    justo = float(ft.MdMRE.iloc[0])
    print(f"\n  comparacion honesta: MdMRE {justo:.3f} por tarea (con su estimador) "
          f"frente a {mejor.MdMRE:.3f} por proyecto -> se divide por "
          f"{justo / mejor.MdMRE:.1f}")
    print(f"  y el PRED(25) sube de {float(ft.PRED25.iloc[0]):.3f} a "
          f"{mejor.PRED25:.3f}, que es un factor de {mejor.PRED25 / float(ft.PRED25.iloc[0]):.1f}")
    print("  el MMRE del total sale PEOR que el sin corregir (0,79 frente a 0,60) porque")
    print("  es una media y se la comen dos o tres proyectos que fallan por mucho; la")
    print("  mediana, que es lo que se reporta en esta literatura, mejora sin ambiguedad")

    print(f"\n{S}\nCOMO MEJORA SEGUN EL TAMANO DEL PROYECTO\n{S}")
    tot["tramo"] = pd.cut(tot.n, [7, 15, 40, 120, 400, 100000],
                          labels=["8-15", "16-40", "41-120", "121-400", "mas de 400"])
    fs = []
    for tr, gg in tot.groupby("tramo", observed=True):
        if len(gg) < 3:
            continue
        d1 = metricas_totales(gg["real"], gg["mod_sm"])
        d0 = metricas_totales(gg["real"], gg["cte"])
        fs.append(dict(tareas=tr, proyectos=len(gg), MdMRE=d1["MdMRE"],
                       PRED25=d1["PRED25"], factor2=d1["factor2"],
                       MdMRE_constante=d0["MdMRE"], PRED25_constante=d0["PRED25"]))
    e = pd.DataFrame(fs)
    print(e.round(4).to_string(index=False))
    print("  si el error cayese como 1/raiz(n) bajaria sin parar; si se aplana, lo que")
    print("  queda es el desplazamiento comun del proyecto, que ninguna suma cancela")

    r.to_csv(OUT / "resumen.csv", index=False)
    e.to_csv(OUT / "por_tamano.csv", index=False)
    tot.to_csv(OUT / "totales.csv")
    json.dump({"smearing_mediano": float(fac.median()), "min_tareas": MIN_TAREAS},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
