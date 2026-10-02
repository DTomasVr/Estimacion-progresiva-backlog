"""E87 - ¿El sesgo de estimacion es del EQUIPO o del PROYECTO?

Si un equipo que ya uso Agix nos deja sus proyectos anteriores, ¿eso sirve para el
siguiente? E76 ya probo pasar HORAS REALES de otros proyectos del mismo equipo y
salio negativo. Pero queda otra cosa distinta y plausible: el SESGO DE ESTIMACION.

    b_p = log( suma de horas reales / suma de lo que el usuario estimo )

Si un equipo subestima siempre un 30 %, eso es una propiedad suya y deberia
transferirse de un proyecto al siguiente. Si el sesgo cambia de proyecto en proyecto
dentro del mismo equipo, no sirve de nada guardarlo.

Se contesta descomponiendo la varianza de b_p: cuanta esta ENTRE organizaciones y
cuanta DENTRO de cada una. Y despues se prueba de verdad: corregir usando el sesgo
de los proyectos HERMANOS, contra usar la mediana global.

AVISO DE POTENCIA: solo hay 3 organizaciones con 2 o mas proyectos que tengan
estimacion (20 proyectos). Esto no decide nada por si solo; sirve para ver si la via
merece que se le dedique mas datos.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
OUT = Path("results/sesgo_del_equipo")
S = "=" * 96
COB = .8


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tam = sw.groupby("project").size()
    sw = sw[sw.project.isin(tam[tam >= 8].index)].reset_index(drop=True)
    est = sw.HoursEstimate.where(sw.HoursEstimate > 0)
    cob = est.notna().groupby(sw.project).mean()
    nom = list(cob[cob >= COB].index)
    d = sw[sw.project.isin(nom)].copy()
    d["est"] = est[sw.project.isin(nom)]
    d = d[d.est.notna()]
    d["org"] = d.project.map(lambda p: p.split(":")[0] if ":" in p else p)

    b = d.groupby("project").apply(
        lambda g: np.log(g.effort_hours.sum() / g.est.sum())).rename("sesgo")
    orgp = d.groupby("project").org.first()
    n_t = d.groupby("project").size()
    t = pd.concat([b, orgp, n_t.rename("tareas")], axis=1).reset_index()
    v = t.org.value_counts()
    con_her = t[t.org.isin(v[v >= 2].index)].copy()
    print(f"{len(t)} proyectos con estimacion · {len(con_her)} en organizaciones con "
          f"hermano · {con_her.org.nunique()} organizaciones")

    print(f"\n{S}\nEL SESGO DE CADA PROYECTO, POR ORGANIZACION\n{S}")
    for o, g in con_her.groupby("org"):
        vals = " ".join(f"{x:+.2f}" for x in sorted(g.sesgo))
        print(f"  {o:16s} n={len(g):2d}  media {g.sesgo.mean():+.3f}  "
              f"sd {g.sesgo.std():.3f}   [{vals}]")

    # descomposicion de la varianza del sesgo
    gran = con_her.sesgo.mean()
    med_o = con_her.groupby("org").sesgo.mean()
    n_o = con_her.groupby("org").size()
    entre = float(np.average((med_o - gran) ** 2, weights=n_o))
    dentro = float(np.average(con_her.groupby("org").sesgo.var(ddof=1).fillna(0),
                              weights=n_o))
    print(f"\n{S}\n¿ES DEL EQUIPO O DEL PROYECTO?\n{S}")
    print(f"  varianza ENTRE organizaciones  {entre:.4f}  {entre/(entre+dentro):6.1%}")
    print(f"  varianza DENTRO de cada una    {dentro:.4f}  {dentro/(entre+dentro):6.1%}")

    # prueba directa: predecir el sesgo con los hermanos, contra la mediana global
    err_her, err_glo = [], []
    for i, r in con_her.iterrows():
        her = con_her[(con_her.org == r.org) & (con_her.project != r.project)]
        otros = con_her[con_her.project != r.project]
        err_her.append(abs(r.sesgo - her.sesgo.mean()))
        err_glo.append(abs(r.sesgo - otros.sesgo.median()))
    eh, eg = float(np.mean(err_her)), float(np.mean(err_glo))
    print(f"\n{S}\nPREDECIR EL SESGO DE UN PROYECTO ({len(con_her)} proyectos)\n{S}")
    print(f"  con el sesgo de sus HERMANOS del mismo equipo   error medio {eh:.4f}")
    print(f"  con la mediana global de todos los proyectos    error medio {eg:.4f}")
    mej = (eg - eh) / eg
    print(f"  los hermanos {'MEJORAN' if mej > 0 else 'EMPEORAN'} un {abs(mej):.1%}")

    print(f"\n{S}\nLECTURA\n{S}")
    if entre / (entre + dentro) > .5 and mej > .1:
        print("  El sesgo parece del EQUIPO: guardar los proyectos anteriores serviria.")
    else:
        print("  El sesgo es sobre todo del PROYECTO, no del equipo: lo que un equipo")
        print("  se equivoco en un proyecto no dice gran cosa del siguiente.")
    print(f"\n  AVISO: {con_her.org.nunique()} organizaciones y {len(con_her)} proyectos "
          f"son muy pocos. Esto orienta, no decide.")
    t.to_csv(OUT / "por_proyecto.csv", index=False)
    json.dump({"proyectos": len(t), "con_hermano": len(con_her),
               "organizaciones": int(con_her.org.nunique()),
               "var_entre": entre, "var_dentro": dentro,
               "err_hermanos": eh, "err_global": eg, "mejora": mej},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/por_proyecto.csv")


if __name__ == "__main__":
    main()
