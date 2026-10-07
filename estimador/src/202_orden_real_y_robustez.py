"""E88 - Dos preguntas de validez sobre el anclaje, en una sola corrida.

(A) EL ORDEN. Todo lo medido hasta ahora toma k tareas cerradas AL AZAR. En
    produccion las cerradas son las PRIMERAS del proyecto, no una muestra
    aleatoria. Si las tareas tempranas son sistematicamente distintas -arranque,
    andamiaje, configuracion- nuestras cifras estan infladas y un revisor lo va a
    preguntar. Se compara el sorteo al azar contra el orden real.

    Hay dos maneras de ordenar y las dos se miden:
      por 'completed'  el orden en que de verdad se cerraron (lo correcto, pero
                       solo 58 de los 114 proyectos lo tienen completo)
      por 'created'    el orden en que entraron al backlog (proxy, los 114)

(B) LA ROBUSTEZ. Con k pequeno el cociente de sumas lo domina una sola tarea
    rara. Se prueban cinco estimadores del desplazamiento, todos con la misma
    contraccion k/(k+12):

      cociente de sumas    log( suma real / suma predicha )      <- el actual
      media de logs        media de log(real_i / pred_i)
      mediana de logs      mediana de log(real_i / pred_i)
      Huber                M-estimador con c=1,345 sobre los logs
      sumas recortadas     cociente de sumas quitando el mayor y el menor

El modelo no se toca: se leen las predicciones ya calculadas. La METRICA se
promedia sobre los sorteos, no la prediccion.
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
pd.set_option("display.width", 210)
OUT = Path("results/orden_real_y_robustez")
S = "=" * 104
LAMBDA, SORTEOS, MIN_TAREAS = 12, 60, 8
KS = (1, 2, 3, 5, 10, 20)
ESTIMADORES = ("cociente de sumas", "media de logs", "mediana de logs", "Huber",
               "sumas recortadas")


def mt(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    m = float(np.median(np.log(pred / real)))
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                MAE_h=float(np.abs(pred - real).mean()),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def huber(x, c=1.345, iters=12):
    """M-estimador de localizacion sobre los logs."""
    mu = float(np.median(x))
    s = float(np.median(np.abs(x - mu))) * 1.4826
    if s <= 1e-9:
        return mu
    for _ in range(iters):
        r = (x - mu) / s
        w = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-9))
        nuevo = float(np.sum(w * x) / np.sum(w))
        if abs(nuevo - mu) < 1e-10:
            break
        mu = nuevo
    return mu


def desplazamiento(real_k, pred_k, como):
    """El offset en log ANTES de contraer."""
    if como == "cociente de sumas":
        return float(np.log(real_k.sum() / pred_k.sum()))
    if como == "sumas recortadas":
        if len(real_k) < 3:
            return float(np.log(real_k.sum() / pred_k.sum()))
        o = np.argsort(np.log(real_k / pred_k))[1:-1]
        return float(np.log(real_k[o].sum() / pred_k[o].sum()))
    l = np.log(real_k / pred_k)
    if como == "media de logs":
        return float(l.mean())
    if como == "mediana de logs":
        return float(np.median(l))
    if como == "Huber":
        return huber(l)
    raise ValueError(como)


def total(idx_p, nom, real, pred, elegir, como, k):
    """Total de cada proyecto anclando en las k tareas que devuelve `elegir`."""
    t = np.empty(len(nom))
    for j, p in enumerate(nom):
        i = idx_p[p]
        if len(i) <= k:
            t[j] = pred[i].sum()
            continue
        sel = elegir(p, i, k)
        resto = np.setdiff1d(i, sel)
        off = desplazamiento(real[sel], pred[sel], como) * k / (k + LAMBDA)
        t[j] = real[sel].sum() + (pred[resto] * np.exp(off)).sum()
    return t


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
    proy = sw.project.to_numpy()
    real = sw.effort_hours.to_numpy(float)
    nom = list(pd.unique(proy))
    idx_p = {p: np.flatnonzero(proy == p) for p in nom}
    real_tot = np.array([real[idx_p[p]].sum() for p in nom])
    frio = np.array([pred[idx_p[p]].sum() for p in nom])

    # orden de cada tarea dentro de su proyecto, por las dos fechas
    orden = {}
    for col in ("created", "completed"):
        f = sw[col]
        orden[col] = {p: idx_p[p][np.argsort(f.iloc[idx_p[p]].to_numpy())]
                      for p in nom}
    # proyectos donde 'completed' esta en TODAS las tareas: los unicos donde el
    # orden de cierre es de fiar
    full = [p for p in nom if sw.completed.iloc[idx_p[p]].notna().all()]
    print(f"{len(nom)} proyectos · {len(sw):,} tareas · "
          f"{len(full)} con fecha de cierre completa · {time.time() - t0:.0f} s",
          flush=True)
    print(f"  en frio: ±25 % {mt(real_tot, frio)['PRED25']:.1%} · "
          f"MdMRE {mt(real_tot, frio)['MdMRE']:.4f}", flush=True)

    sub = np.array([p in set(full) for p in nom])
    filas = []

    def anota(bloque, orden_etq, est, k, t):
        sel = slice(None) if bloque == "114 proyectos" else sub
        filas.append(dict(bloque=bloque, orden=orden_etq, estimador=est, k=k,
                          **mt(real_tot[sel], t[sel])))

    BLOQUES = (("114 proyectos", slice(None)), ("58 con fecha de cierre", sub))
    for bloque, sel in BLOQUES:
        filas.append(dict(bloque=bloque, orden="-- en frio --", estimador="-", k=0,
                          **mt(real_tot[sel], frio[sel])))

    for est in ESTIMADORES:
        for k in KS:
            # al azar: el total se calcula UNA vez por sorteo y se mide en los dos
            # bloques; antes se repetia entero por bloque, que era el doble de trabajo
            ms = {b: [] for b, _ in BLOQUES}
            for s in range(SORTEOS):
                rng = np.random.default_rng(1100 + s)
                t = total(idx_p, nom, real, pred,
                          lambda p, i, kk, rng=rng: rng.choice(i, kk, replace=False),
                          est, k)
                for b, sl in BLOQUES:
                    ms[b].append(mt(real_tot[sl], t[sl]))
            for b, _ in BLOQUES:
                filas.append(dict(bloque=b, orden="al azar", estimador=est, k=k,
                                  **{c: float(np.mean([x[c] for x in ms[b]]))
                                     for c in ms[b][0]}))
            # las k primeras: deterministas, un solo calculo por fecha
            for col, etq in (("created", "las primeras (creacion)"),
                             ("completed", "las primeras (cierre)")):
                t = total(idx_p, nom, real, pred,
                          lambda p, i, kk, c=col: orden[c][p][:kk], est, k)
                for b, sl in BLOQUES:
                    if col == "completed" and b == "114 proyectos":
                        continue
                    filas.append(dict(bloque=b, orden=etq, estimador=est, k=k,
                                      **mt(real_tot[sl], t[sl])))
        print(f"  {est} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    t.to_csv(OUT / "resumen.csv", index=False)

    # ---------- (A) el orden importa? ----------
    print(f"\n{S}\n(A) ¿CAMBIA ALGO SI LAS CERRADAS SON LAS PRIMERAS, "
          f"NO UNA MUESTRA AL AZAR?\n{S}")
    for bloque, ordenes in (("114 proyectos", ("al azar", "las primeras (creacion)")),
                            ("58 con fecha de cierre",
                             ("al azar", "las primeras (creacion)",
                              "las primeras (cierre)"))):
        d = t[(t.bloque == bloque) & (t.estimador == "cociente de sumas")]
        frio_b = float(t[(t.bloque == bloque) & (t.k == 0)].PRED25.iloc[0])
        print(f"\n  {bloque} · en frio ±25 % = {frio_b:.1%}")
        piv = d.pivot(index="k", columns="orden", values="PRED25")
        print(piv[list(ordenes)].round(4).to_string())
        az = piv["al azar"]
        for o in ordenes[1:]:
            dif = (piv[o] - az).mean() * 100
            print(f"    '{o}' frente al azar: {dif:+.1f} puntos de media")

    # ---------- (B) que estimador aguanta mejor con k pequeno ----------
    print(f"\n{S}\n(B) ¿QUE ESTIMADOR DEL DESPLAZAMIENTO AGUANTA MEJOR?\n{S}")
    for orden_etq in ("al azar", "las primeras (cierre)"):
        bloque = "114 proyectos" if orden_etq == "al azar" else "58 con fecha de cierre"
        d = t[(t.bloque == bloque) & (t.orden == orden_etq)]
        if not len(d):
            continue
        print(f"\n  {bloque} · {orden_etq} · ±25 %")
        print(d.pivot(index="k", columns="estimador",
                      values="PRED25").round(4).to_string())
        chico = d[d.k <= 3].groupby("estimador").PRED25.mean().sort_values()
        print(f"    mejor con k<=3: {chico.index[-1]} ({chico.iloc[-1]:.1%}) · "
              f"el actual: {chico.get('cociente de sumas', float('nan')):.1%}")

    # ---------- veredicto ----------
    print(f"\n{S}\nLECTURA\n{S}")
    d58 = t[(t.bloque == "58 con fecha de cierre") &
            (t.estimador == "cociente de sumas")]
    az = d58[d58.orden == "al azar"].set_index("k").PRED25
    ci = d58[d58.orden == "las primeras (cierre)"].set_index("k").PRED25
    dif = float((ci - az).mean()) * 100
    if abs(dif) < 1.5:
        print(f"  (A) El orden NO cambia el resultado ({dif:+.1f} puntos de media).")
        print("      Lo medido al azar vale como medida de produccion.")
    elif dif < 0:
        print(f"  (A) CUIDADO: con el orden real el anclaje rinde {abs(dif):.1f} puntos")
        print("      MENOS. Las cifras al azar estaban infladas; hay que reportar esto.")
    else:
        print(f"  (A) Con el orden real rinde {dif:+.1f} puntos MAS que al azar.")
    g = t[(t.bloque == "114 proyectos") & (t.orden == "al azar") & (t.k <= 3)]
    r = g.groupby("estimador").PRED25.mean().sort_values()
    mejor, act = r.index[-1], float(r.get("cociente de sumas"))
    if r.iloc[-1] - act > .01:
        print(f"  (B) '{mejor}' gana al cociente de sumas con k<=3: "
              f"{r.iloc[-1]:.1%} contra {act:.1%}. Merece cambiarlo.")
    else:
        print(f"  (B) Ningun estimador mejora al cociente de sumas con k<=3 "
              f"(mejor {mejor}: {r.iloc[-1]:.1%} contra {act:.1%}).")

    json.dump({"proyectos": len(nom), "con_fecha_cierre": len(full),
               "sorteos": SORTEOS, "lambda": LAMBDA,
               "dif_orden_puntos": dif},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
