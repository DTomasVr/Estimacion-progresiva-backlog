"""E83 - Una sola pregunta al usuario: ¿cuanto cuesta una tarea tipica en tu equipo?

Todo lo probado hasta ahora intenta DEDUCIR la escala del proyecto a partir del texto,
y esta medido que el texto no la lleva. Pero hay una fuente que no se ha usado: el
propio usuario. Cuando un jefe de proyecto crea el backlog sabe perfectamente si su
equipo trabaja en tareas de media hora o de ocho, aunque no sepa cuanto costara cada
tarea concreta.

Eso es exactamente el 11 % de varianza que falta, y pedirlo es UNA casilla del
formulario, no cinco tareas cerradas con sus horas.

    off   = log( lo que dice el usuario / la mediana que predice el modelo )
    total = suma de las predicciones x e^off

La pregunta que decide si vale la pena ponerlo en el producto no es si el oraculo
funciona -eso ya se sabe- sino CON CUANTA PRECISION tendria que acertar el usuario.
Por eso se mide con ruido creciente y tambien con una respuesta por tramos, que es
lo que de verdad se puede pedir en un formulario.

  A  en frio                              lo que hay
  B  el usuario acierta exacto            el techo
  C  el usuario se equivoca x1,2 / x1,5 / x2   realista
  D  el usuario elige un TRAMO            lo que cabe en un desplegable
  E  cinco tareas cerradas                el mecanismo actual, para comparar

El ruido se simula con 200 sorteos por nivel y se promedia la METRICA.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
OUT = Path("results/una_pregunta")
S = "=" * 104
MIN_TAREAS, LAMBDA, SORTEOS = 8, 12, 200
RUIDOS = (0.0, 0.18, 0.41, 0.69)          # sd en log: ~x1,2 x1,5 x2 de error tipico
TRAMOS = [0, 1, 2, 4, 8, 1e9]
REPRE = [0.5, 1.5, 3.0, 6.0, 12.0]        # el valor que representa a cada tramo


def mt(real, pred):
    real, pred = np.asarray(real, float), np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    m = float(np.median(np.log(pred / real)))
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                MAE_h=float(np.abs(pred - real).mean()),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    assert len(pred) == len(sw)
    tam = sw.groupby("project").size()
    m = sw.project.isin(tam[tam >= MIN_TAREAS].index).to_numpy()
    sw, pred = sw[m].reset_index(drop=True), pred[m]
    proy, real = sw.project.to_numpy(), sw.effort_hours.to_numpy(float)
    nom = list(pd.unique(proy))
    idx = {p: np.flatnonzero(proy == p) for p in nom}

    real_tot = np.array([real[idx[p]].sum() for p in nom])
    frio = np.array([pred[idx[p]].sum() for p in nom])
    # lo que el usuario sabe: la mediana de horas por tarea de su equipo
    med_real = np.array([np.median(real[idx[p]]) for p in nom])
    med_pred = np.array([np.median(pred[idx[p]]) for p in nom])
    print(f"{len(nom)} proyectos · {len(sw):,} tareas · {time.time() - t0:.0f} s")
    print(f"  mediana de h/tarea: real {np.median(med_real):.2f} · "
          f"predicha {np.median(med_pred):.2f}")
    print(f"  dispersion en log: real {np.std(np.log(med_real)):.3f} · "
          f"predicha {np.std(np.log(med_pred)):.3f}")

    # La cabeza de la MEDIA no predice la mediana: predice ~5 h donde la mediana real
    # son 2 h, porque esta calibrada para sumar. Comparar directamente lo que dice el
    # usuario con median(prediccion) mete un desplazamiento global de ~x0,37 que no
    # tiene nada que ver con el proyecto. Se quita con la mediana de los OTROS
    # proyectos, igual que en E71: asi solo queda la parte especifica del proyecto.
    bruto = np.log(med_real) - np.log(med_pred)
    ajuste = np.array([np.median(np.delete(bruto, j)) for j in range(len(nom))])
    print(f"  desfase global mediana real / mediana predicha: "
          f"x{np.exp(np.median(bruto)):.2f} (se quita fuera de muestra)")

    def total_con(dicho):
        """Aplicar el desplazamiento que implica lo que el usuario dice."""
        off = np.log(dicho / med_pred) - ajuste
        return frio * np.exp(off)

    filas = [dict(brazo="A · en frio", detalle="", **mt(real_tot, frio))]

    # B y C: el usuario da un numero, con ruido creciente
    for sd in RUIDOS:
        if sd == 0:
            filas.append(dict(brazo="B · el usuario acierta exacto", detalle="sin error",
                              **mt(real_tot, total_con(med_real))))
            continue
        ms = []
        for s in range(SORTEOS):
            rng = np.random.default_rng(500 + s)
            dicho = med_real * np.exp(rng.normal(0, sd, len(nom)))
            ms.append(mt(real_tot, total_con(dicho)))
        filas.append(dict(brazo=f"C · el usuario falla x{np.exp(sd):.1f}",
                          detalle=f"sd log {sd}",
                          **{c: float(np.mean([x[c] for x in ms])) for c in ms[0]}))

    # D: el usuario elige un tramo de un desplegable
    tr = np.digitize(med_real, TRAMOS[1:-1])
    dicho_tr = np.array([REPRE[i] for i in tr])
    filas.append(dict(brazo="D · el usuario elige un TRAMO",
                      detalle="<1 · 1-2 · 2-4 · 4-8 · >8 h",
                      **mt(real_tot, total_con(dicho_tr))))

    # E: el mecanismo actual, cinco tareas cerradas
    for k in (3, 5):
        ms = []
        for s in range(SORTEOS):
            rng = np.random.default_rng(900 + s)
            est = np.empty(len(nom))
            for j, p in enumerate(nom):
                i = idx[p]
                if len(i) <= k:
                    est[j] = pred[i].sum()
                    continue
                sel = rng.choice(i, k, replace=False)
                resto = np.setdiff1d(i, sel)
                off = np.log(real[sel].sum() / pred[sel].sum()) * k / (k + LAMBDA)
                est[j] = real[sel].sum() + (pred[resto] * np.exp(off)).sum()
            ms.append(mt(real_tot, est))
        filas.append(dict(brazo=f"E · {k} tareas cerradas", detalle="mecanismo actual",
                          **{c: float(np.mean([x[c] for x in ms])) for c in ms[0]}))

    t = pd.DataFrame(filas)
    print(f"\n{S}\nUNA SOLA PREGUNTA AL USUARIO ({len(nom)} proyectos)\n{S}")
    print(t[["brazo", "detalle", "PRED25", "MdMRE", "factor2", "MAE_h",
             "SSPB_pc"]].round(4).to_string(index=False))

    a = float(t.iloc[0].PRED25)
    print(f"\n{S}\nLECTURA\n{S}")
    for _, x in t.iloc[1:].iterrows():
        print(f"  {x.brazo:34s} ±25 % {x.PRED25:6.1%}  ({(x.PRED25 - a) * 100:+5.1f} pts "
              f"sobre el frio)")
    d = float(t[t.brazo.str.startswith("D")].PRED25.iloc[0])
    e5 = float(t[t.brazo.str.startswith("E · 5")].PRED25.iloc[0])
    print(f"\n  un desplegable de 5 tramos vale {'MAS' if d > e5 else 'menos'} que cinco "
          f"tareas cerradas ({d:.1%} contra {e5:.1%})")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": len(nom), "sorteos": SORTEOS, "ruidos": list(RUIDOS)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
