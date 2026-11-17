"""E109 - Arranque en caliente SIN pedir estimaciones: el historial pasado.

Lo que mas sube la precision es la estimacion del usuario, pero exige que
estime casi todo el backlog (E84). La alternativa de menos rozamiento es
pedirle, al crear el proyecto, unas cuantas TAREAS PASADAS con sus horas
reales: datos que ya tiene, no juicios que ha de emitir.

La literatura lo trata como within-company contra cross-company (Mendes,
Kitchenham; Minku y Yao con Dycom): los datos propios ganan cuando existen, los
externos sirven de arranque, y hay riesgo documentado de NEGATIVE TRANSFER. Lo
que aqui se mide cae en un nivel intermedio que esa literatura no cubre:
historial de la MISMA PERSONA en OTRO proyecto.

Hay precedentes contradictorios en el propio proyecto:

  E76  pasar horas reales de proyectos HERMANOS al total: -3,0 puntos
  E87  el sesgo de estimacion es del PROYECTO, no del equipo (11,6 % entre
       organizaciones, los hermanos mejoran un 2,2 %, o sea ruido)
  E100 pero el DESARROLLADOR explica un 9,8 % de la varianza del residuo, y
       anclar por persona mejora SA, orden y MAE

La diferencia entre E76/E87 y E100 es la granularidad: proyecto hermano contra
PERSONA. Esto lo separa.

Brazos, todos en FRIO sobre el proyecto objetivo (cero tareas suyas cerradas):

  A  en frio                                    lo que hay hoy
  B  k tareas pasadas de SUS DESARROLLADORES    lo que se le pediria al usuario
  C  k tareas pasadas de SU ORGANIZACION        si no se sabe quien las hizo
  D  k tareas pasadas AL AZAR de otros proyectos  CONTROL de negative transfer
  E  k tareas cerradas DEL PROPIO proyecto      la referencia de arriba

Y para B, dos formas de usar el historial: el cociente de sumas plano, y uno
ponderado por parecido de texto con las tareas del backlog.

El desplazamiento se contrae con k/(k+lambda) y se barre lambda, porque el
historial externo merece contraerse mas que el propio.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/arranque_con_historial")
S = "=" * 112
MIN_TAREAS, SORTEOS = 8, 40
KS = (5, 10, 20)
LAMBDAS = (12, 24, 48)


def met(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    mar = np.abs(np.median(y) - y).mean()
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                MAE_h=float(np.abs(p - y).mean()),
                SA=float(1 - np.abs(p - y).mean() / mar),
                sesgo=float(np.median(p / y)),
                spearman=float(spearmanr(p, y).statistic))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    sw = todo[~todo.fuente_jira.eq("Apache")].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy").astype(np.float32)
    V = V[(~todo.fuente_jira.eq("Apache")).to_numpy()]
    p_tar = np.clip(np.load("results/experto_cortas/predicciones.npz")["p_glob"],
                    1e-6, None)
    p_tot = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    tam = sw.groupby("project").size()
    m = sw.project.isin(tam[tam >= MIN_TAREAS].index).to_numpy()
    sw, V, p_tar, p_tot = (sw[m].reset_index(drop=True), V[m], p_tar[m], p_tot[m])
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    dev = sw.desarrollador.fillna("?").astype(str).to_numpy()
    org = np.array([p.split(":")[0] if ":" in p else p for p in proy])
    nom = list(pd.unique(proy))
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    real_tot = np.array([y[idx[p]].sum() for p in nom])

    # para cada proyecto, de donde puede salir su historial
    fuentes = {}
    for p in nom:
        i = idx[p]
        fuera = proy != p
        dv = set(dev[i]) - {"?"}
        f_dev = np.flatnonzero(fuera & np.isin(dev, list(dv))) if dv else np.array([], int)
        f_org = np.flatnonzero(fuera & (org == org[i[0]]))
        fuentes[p] = dict(dev=f_dev, org=f_org, azar=np.flatnonzero(fuera),
                          propio=i)
    con_dev = [p for p in nom if len(fuentes[p]["dev"]) >= max(KS)]
    con_org = [p for p in nom if len(fuentes[p]["org"]) >= max(KS)]
    print(f"{len(nom)} proyectos · {len(sw):,} tareas · "
          f"{len(con_dev)} con historial de SUS desarrolladores · "
          f"{len(con_org)} con historial de su organizacion · "
          f"{time.time() - t0:.0f} s", flush=True)

    # se evalua sobre los proyectos que tienen las tres fuentes, para comparar justo
    ev = [p for p in nom if p in set(con_dev) and p in set(con_org)]
    print(f"  se evalua sobre los {len(ev)} que tienen las dos", flush=True)
    sel_t = np.isin(proy, ev)
    jj = [nom.index(p) for p in ev]

    def corre(fuente, k, lam, rng, pesado=False):
        """Un desplazamiento por proyecto a partir de k tareas de `fuente`."""
        off = np.zeros(len(y))
        for p in ev:
            cand = fuentes[p][fuente]
            if len(cand) < k:
                continue
            s = rng.choice(cand, k, replace=False)
            if pesado:
                # pondera el historial por parecido con el backlog objetivo
                w = (V[idx[p]] @ V[s].T).max(0)
                w = np.maximum(w - w.min(), 1e-6)
                w = w / w.sum()
                o = float(np.log((w * y[s]).sum() / (w * p_tar[s]).sum()))
                kef = float(1.0 / np.sum(w ** 2))
                o *= kef / (kef + lam)
            else:
                o = float(np.log(y[s].sum() / p_tar[s].sum())) * k / (k + lam)
            off[idx[p]] = o
        return off

    FUENTES = [("A · en frio", None, False),
               ("B · pasadas de SUS desarrolladores", "dev", False),
               ("B' · idem, ponderado por parecido", "dev", True),
               ("C · pasadas de su ORGANIZACION", "org", False),
               ("D · pasadas AL AZAR (control)", "azar", False),
               ("E · cerradas del PROPIO proyecto", "propio", False)]
    filas = []
    for etq, f, pes in FUENTES:
        if f is None:
            filas.append(dict(brazo=etq, k=0, lam=0,
                              **met(y[sel_t], p_tar[sel_t])))
            continue
        for k in KS:
            for lam in LAMBDAS:
                ms = []
                for s_ in range(SORTEOS):
                    rng = np.random.default_rng(9100 + s_)
                    o = corre(f, k, lam, rng, pes)
                    ms.append(met(y[sel_t], (p_tar * np.exp(o))[sel_t]))
                filas.append(dict(brazo=etq, k=k, lam=lam,
                                  **{c: float(np.mean([x[c] for x in ms]))
                                     for c in ms[0]}))
        print(f"  {etq} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    cols = ["brazo", "k", "lam", "PRED25", "MdMRE", "MAE_h", "SA", "sesgo",
            "spearman"]
    print(f"\n{S}\nPOR TAREA · {int(sel_t.sum()):,} tareas de {len(ev)} proyectos "
          f"en FRIO\n{S}")
    mejor = (t[t.k > 0].sort_values("SA", ascending=False)
             .groupby("brazo").head(1).sort_values("SA", ascending=False))
    print(pd.concat([t[t.k == 0], mejor])[cols].round(4).to_string(index=False))
    print(f"\n  (de cada brazo, su mejor k y lambda por SA)")

    print(f"\n{S}\nEL CONTROL DE NEGATIVE TRANSFER\n{S}")
    a = float(t[t.k == 0].SA.iloc[0])
    for etq, _, _ in FUENTES[1:]:
        b = t[t.brazo == etq]
        bb = b.loc[b.SA.idxmax()]
        print(f"  {etq:38s} SA {a:.4f} -> {bb.SA:.4f} ({bb.SA - a:+.4f}) · "
              f"k={int(bb.k)} lambda={int(bb.lam)}")

    print(f"\n{S}\nLECTURA\n{S}")
    bd = t[t.brazo.str.startswith("B ·")]
    bd = bd.loc[bd.SA.idxmax()]
    az = t[t.brazo.str.startswith("D")]
    az = az.loc[az.SA.idxmax()]
    pr = t[t.brazo.str.startswith("E")]
    pr = pr.loc[pr.SA.idxmax()]
    print(f"  en frio                      SA {a:.4f}")
    print(f"  historial de sus devs        SA {bd.SA:.4f} ({bd.SA - a:+.4f})")
    print(f"  historial al azar (control)  SA {az.SA:.4f} ({az.SA - a:+.4f})")
    print(f"  cerradas del propio proyecto SA {pr.SA:.4f} ({pr.SA - a:+.4f})")
    neto = float(bd.SA - az.SA)
    if neto > .01:
        print(f"\n  El historial de SUS desarrolladores aporta {neto:+.4f} de SA")
        print("  POR ENCIMA del control al azar: no es recalibracion global, es")
        print("  informacion de la persona. Se puede pedir al crear el proyecto.")
    else:
        print(f"\n  Lo que gana el historial propio ({bd.SA - a:+.4f}) no supera al")
        print(f"  azar ({az.SA - a:+.4f}): es recalibracion global, no transferencia.")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos_evaluados": len(ev), "sorteos": SORTEOS,
               "SA_frio": a, "SA_devs": float(bd.SA), "SA_azar": float(az.SA),
               "SA_propio": float(pr.SA)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
