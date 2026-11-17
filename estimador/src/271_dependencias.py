"""E-271 - ¿Las dependencias entre tareas dicen algo que al modelo le falta?

En E-270 la media de enlaces por tarea fue lo unico de nivel proyecto que
correlaciono con el desplazamiento de escala (+0,28). PBE guarda las
dependencias (tabla task_dependencies), asi que se comprueba si sirven:

  1 por tarea: ¿una tarea con enlaces tarda mas de lo que predice el modelo?
  2 corrector por tarea: residuo del modelo ~ enlaces, ajustado en OTROS
    proyectos (5 pliegues por proyecto), y como queda el total;
  3 corrector por proyecto: desplazamiento ~ media de enlaces, dejando uno fuera.

Aviso: en Jira los enlaces se van anadiendo durante la vida de la incidencia, asi
que parte de la senal puede venir de despues de empezar. Es un techo optimista.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_predict

OUT = Path("results/dependencias")
COLS = ["n_links", "n_links_estructurales", "n_componentes"]


def pred25(t, r):
    return float(np.mean(np.abs(t - r) / r <= .25))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    sw["est"] = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    tam = sw.groupby("project").size()
    sw = sw[sw.project.isin(tam[tam >= 8].index)].reset_index(drop=True)
    sw["res"] = np.log(sw.effort_hours / sw.est)
    g = sw.groupby("project")
    real, frio = g.effort_hours.sum(), g.est.sum()
    print(f"{g.ngroups} proyectos · {len(sw):,} tareas · con algun enlace {100 * (sw.n_links > 0).mean():.1f} % · "
          f"estructural {100 * (sw.n_links_estructurales > 0).mean():.1f} % · en frio {pred25(frio, real):.3f}")

    sw["res_c"] = sw.res - g.res.transform("mean")
    f1 = []
    for c in COLS:
        con = sw[c] > 0
        f1.append(dict(rasgo=c, tareas_con=int(con.sum()),
                       spearman_global=float(sw[c].corr(sw.res, method="spearman")),
                       spearman_dentro_del_proyecto=float(sw[c].corr(sw.res_c, method="spearman")),
                       veces_mas_que_el_modelo_si_tiene=float(np.exp(sw.res[con].median() - sw.res[~con].median())),
                       horas_medianas_con=float(sw.effort_hours[con].median()),
                       horas_medianas_sin=float(sw.effort_hours[~con].median())))
    f1 = pd.DataFrame(f1)
    f1.to_csv(OUT / "por_tarea.csv", index=False)
    print("\n1 · POR TAREA: residuo del modelo frente a los enlaces")
    print(f1.round(3).to_string(index=False))

    X = np.column_stack([np.log1p(sw[c].fillna(0)) for c in COLS] + [(sw[c] > 0).astype(float) for c in COLS])
    p = np.zeros(len(sw))
    for tr, te in GroupKFold(5).split(X, sw.res, sw.project):
        m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X[tr], sw.res.to_numpy()[tr])
        p[te] = m.predict(X[te]) - m.predict(X[tr]).mean()
    sw["est2"] = sw.est * np.exp(p)
    sw["est2"] *= sw.est.sum() / sw.est2.sum()
    t2 = sw.groupby("project").est2.sum()
    r2 = 1 - ((sw.res - p - (sw.res - p).mean()) ** 2).sum() / ((sw.res - sw.res.mean()) ** 2).sum()
    f2 = pd.DataFrame([dict(brazo="en frio", PRED25=pred25(frio, real), MdMRE=float(np.median(np.abs(frio - real) / real))),
                       dict(brazo="corrector por tarea con enlaces", PRED25=pred25(t2, real),
                            MdMRE=float(np.median(np.abs(t2 - real) / real)))])
    print(f"\n2 · CORRECTOR POR TAREA (ajustado en otros proyectos) · R2 del residuo {r2:+.4f}")
    print(f2.round(3).to_string(index=False))

    b = np.log(real / frio)
    Z = pd.DataFrame({c: np.log1p(g[c].mean()) for c in COLS} | {f"frac_{c}": g[c].apply(lambda s: (s > 0).mean()) for c in COLS})
    cor = Z.corrwith(b, method="spearman")
    Zs = (Z - Z.mean()) / Z.std()
    pb = cross_val_predict(RidgeCV(alphas=np.logspace(-1, 3, 20)), Zs, b, cv=LeaveOneOut())
    t3 = frio * np.exp(pb - pb.mean())
    print("\n3 · POR PROYECTO: spearman de cada agregado con el desplazamiento de escala")
    print(cor.round(3).to_string())
    f2.loc[len(f2)] = ["corrector por proyecto con enlaces", pred25(t3, real), float(np.median(np.abs(t3 - real) / real))]
    f2.to_csv(OUT / "total.csv", index=False)
    print(f"  R2 dejando uno fuera {1 - ((b - pb) ** 2).sum() / ((b - b.mean()) ** 2).sum():+.3f}")
    print(f2.round(3).to_string(index=False))
    por_fuente = pd.DataFrame({"fuente": g.source.first(), "enlaces": g.n_links.mean(), "b": b}).groupby("fuente").agg(
        proyectos=("b", "size"), enlaces_medios=("enlaces", "mean"), b_medio=("b", "mean"))
    print("\n¿viene la correlacion de la fuente de datos?")
    print(por_fuente.round(3).to_string())
    print(f"\nescrito {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
