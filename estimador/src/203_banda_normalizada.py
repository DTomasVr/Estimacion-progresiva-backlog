"""E89 - Una banda que se adapta al proyecto, no una talla unica.

E55 dejo el intervalo del total calibrado EN CONJUNTO pero roto POR TAMANO: al
80 % nominal cubre el 63,6 % de los proyectos de 8-20 tareas y el 96,4 % de los
de mas de 200. Es decir, miente justo donde el equipo es pequeno y desperdicia
anchura donde el modelo ya sabe. La anchura es la misma para todos, x5,57.

La causa esta medida y es la descomposicion de varianza de E80/E85. El residuo
en log del total de un proyecto tiene dos partes:

    log(real/pred) = u_p            desplazamiento del proyecto, comun a todas
                                    sus tareas, que el anclaje corrige
                   + media de e_i   ruido por tarea, que se promedia al sumar

de donde sale la forma de la varianza:

    var(n, k) = A * (lambda/(k+lambda))^2 + B / n_abiertas + C

  el primer termino   lo que queda del desplazamiento del proyecto despues de
                      anclar con k tareas: la contraccion es k/(k+lambda), asi
                      que lo NO corregido es lambda/(k+lambda)
  el segundo          el ruido por tarea, que se diluye con el numero de tareas
                      que aun hay que estimar
  el tercero          un suelo irreducible

Se ajustan A, B y C por minimos cuadrados sobre los residuos al cuadrado, FUERA
de muestra, y se usa esa desviacion para normalizar el residuo antes de sacar el
cuantil conforme. La garantia de cobertura marginal del conforme se mantiene
-solo se ha cambiado la funcion de no-conformidad- y la cobertura CONDICIONAL
mejora porque cada proyecto recibe la anchura que le toca.

  A  banda unica                el actual
  B  banda por tramo de tamano  el actual, segunda variante
  C  banda NORMALIZADA parametrica  <- lo nuevo
  D  banda normalizada con ajuste libre (log|residuo| ~ log n + contraccion)

Todo con dejar-fuera-el-proyecto-que-se-evalua, y la METRICA promediada sobre
los sorteos de que tareas se cerraron primero.
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
pd.set_option("display.width", 215)
OUT = Path("results/banda_normalizada")
S = "=" * 112
MIN_TAREAS, SORTEOS, LAMBDA = 8, 30, 12
NIVELES = (.5, .8, .9)
KS = (0, 3, 5, 10)
CORTES = [20, 60, 200]
ETIQ = ["8-20", "21-60", "61-200", "mas de 200"]
VENTANA = 60


def tramo(n):
    return np.digitize(n, CORTES)


def ajusta_var(v2, contrac, n_ab):
    """A*contrac^2 + B/n_ab + C por minimos cuadrados no negativos (burdo pero
    suficiente: tres parametros, y se fuerza C>=0 recortando al final)."""
    X = np.column_stack([contrac ** 2, 1.0 / np.maximum(n_ab, 1), np.ones(len(v2))])
    coef, *_ = np.linalg.lstsq(X, v2, rcond=None)
    coef = np.maximum(coef, 0)
    if coef.sum() <= 0:
        coef = np.array([0., 0., float(np.mean(v2))])
    return coef


def sigma_param(coef, contrac, n_ab):
    v = (coef[0] * contrac ** 2 + coef[1] / np.maximum(n_ab, 1) + coef[2])
    return np.sqrt(np.maximum(v, 1e-8))


def sigma_libre(res, contrac, n_ab, fuera):
    """log|residuo| ~ log n_abiertas + contraccion, ajustado sin el proyecto i."""
    y = np.log(np.maximum(np.abs(res), 1e-4))
    X = np.column_stack([np.log(np.maximum(n_ab, 1)), contrac, np.ones(len(res))])
    coef, *_ = np.linalg.lstsq(X[fuera], y[fuera], rcond=None)
    return np.exp(X @ coef)


def qconf(x, p, arriba):
    """Cuantil CONFORME, no el muestral.

    El conforme de nivel 1-a no es el cuantil empirico: es el estadistico de
    orden ceil((m+1)(1-a)), que es lo que da la garantia de cobertura con m
    puntos de calibracion. Usar np.quantile cubre de MENOS, y el sesgo crece
    cuanto mas pequeno es el grupo: con 113 puntos se pierde ~2 puntos y con 30
    se pierden ~5. Es la razon de que la banda por tramo quedara en 74,6 %.
    """
    s = np.sort(x)
    m = len(s)
    if arriba:
        j = int(np.ceil((m + 1) * p)) - 1
    else:
        j = int(np.floor((m + 1) * p)) - 1
    return float(s[int(np.clip(j, 0, m - 1))])


def banda(pred, res, niv, sig=None, grupos=None, min_grupo=12, vecinos=None):
    """Intervalo conforme dejando fuera el proyecto evaluado.

    sig != None -> no-conformidad normalizada: el cuantil se toma sobre res/sig
    y luego se vuelve a multiplicar por el sig del proyecto.
    """
    a = (1 - niv) / 2
    n = len(res)
    lo, hi = np.empty(n), np.empty(n)
    base = res if sig is None else res / sig
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        if vecinos is not None:
            m = np.zeros(n, bool)
            m[vecinos[i]] = True
            m[i] = False
        elif grupos is not None:
            g = m & (grupos == grupos[i])
            if g.sum() >= min_grupo:
                m = g
        q1 = qconf(base[m], a, arriba=False)
        q2 = qconf(base[m], 1 - a, arriba=True)
        f = 1.0 if sig is None else sig[i]
        lo[i] = pred[i] * np.exp(q1 * f)
        hi[i] = pred[i] * np.exp(q2 * f)
    return lo, hi


def mide(real, lo, hi, tr):
    dentro = (real >= lo) & (real <= hi)
    anch = hi / np.maximum(lo, 1e-9)
    d = dict(cobertura=float(dentro.mean()),
             anchura_x=float(np.mean(anch)),
             anchura_mediana_x=float(np.median(anch)))
    for g, e in enumerate(ETIQ):
        s = tr == g
        d[f"cob_{e}"] = float(dentro[s].mean()) if s.any() else np.nan
        d[f"anch_{e}"] = float(np.mean(anch[s])) if s.any() else np.nan
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
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
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    real_tot = np.array([real[idx[p]].sum() for p in nom])
    n_tar = np.array([len(idx[p]) for p in nom], float)
    tr = tramo(n_tar)
    # vecindario local en log(n): el tramo duro deja solo ~30 proyectos de
    # calibracion, y con tan pocos el estadistico de orden conforme se va al
    # extremo y ensancha la banda. Una ventana de los VENTANA proyectos de tamano
    # mas parecido conserva el caracter condicional con el doble de calibracion.
    ln = np.log(n_tar)
    vec = np.array([np.argsort(np.abs(ln - ln[i]))[:VENTANA] for i in range(len(nom))])
    print(f"{len(nom)} proyectos · {len(sw):,} tareas · ventana local {VENTANA} · "
          f"{time.time() - t0:.0f} s", flush=True)
    print("  reparto por tramo: " +
          " · ".join(f"{e} n={int((tr == g).sum())}" for g, e in enumerate(ETIQ)),
          flush=True)

    filas = []
    for k in KS:
        contrac = np.full(len(nom), 1.0) if k == 0 else np.array(
            [LAMBDA / (min(k, len(idx[p]) - 1) + LAMBDA) for p in nom])
        n_ab = np.maximum(n_tar - (0 if k == 0 else k), 1)
        sorteos = 1 if k == 0 else SORTEOS
        acum = {}
        for s in range(sorteos):
            rng = np.random.default_rng(1500 + s)
            tot = np.empty(len(nom))
            for j, p in enumerate(nom):
                i = idx[p]
                if k == 0 or len(i) <= k:
                    tot[j] = pred[i].sum()
                    continue
                sel = rng.choice(i, k, replace=False)
                resto = np.setdiff1d(i, sel)
                off = np.log(real[sel].sum() / pred[sel].sum()) * k / (k + LAMBDA)
                tot[j] = real[sel].sum() + (pred[resto] * np.exp(off)).sum()
            res = np.log(real_tot / np.maximum(tot, 1e-9))

            # escalas, ajustadas dejando fuera el proyecto evaluado
            sig_p = np.empty(len(nom))
            sig_l = np.empty(len(nom))
            for i in range(len(nom)):
                fuera = np.ones(len(nom), bool)
                fuera[i] = False
                coef = ajusta_var(res[fuera] ** 2, contrac[fuera], n_ab[fuera])
                sig_p[i] = sigma_param(coef, contrac[i:i + 1], n_ab[i:i + 1])[0]
                sig_l[i] = sigma_libre(res, contrac, n_ab, fuera)[i]
            sig_p = np.maximum(sig_p, 1e-3)
            sig_l = np.maximum(sig_l, 1e-3)

            for niv in NIVELES:
                for etq, sig, gr in (("A · banda unica", None, None),
                                     ("B · por tramo de tamano", None, tr),
                                     ("C · normalizada (parametrica)", sig_p, None),
                                     ("D · normalizada (ajuste libre)", sig_l, None),
                                     ("E · normalizada + por tramo", sig_p, tr),
                                     ("F · local en log(n)", None, None),
                                     ("G · local + normalizada", sig_p, None)):
                    vv = vec if etq.startswith(("F", "G")) else None
                    lo, hi = banda(tot, res, niv, sig=sig, grupos=gr, vecinos=vv)
                    d = mide(real_tot, lo, hi, tr)
                    key = (k, niv, etq)
                    acum.setdefault(key, []).append(d)
        for (kk, niv, etq), ds in acum.items():
            filas.append(dict(k=kk, nivel=niv, metodo=etq,
                              **{c: float(np.nanmean([x[c] for x in ds]))
                                 for c in ds[0]}))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    t.to_csv(OUT / "resumen.csv", index=False)

    print(f"\n{S}\nCOBERTURA Y ANCHURA AL 80 %\n{S}")
    d = t[t.nivel == .8]
    print(d.pivot(index="k", columns="metodo",
                  values="cobertura").round(4).to_string())
    print("\n  anchura media (factor x):")
    print(d.pivot(index="k", columns="metodo",
                  values="anchura_x").round(2).to_string())

    print(f"\n{S}\nLO QUE IMPORTA: COBERTURA POR TAMANO AL 80 % (objetivo 0,80 "
          f"en las CUATRO)\n{S}")
    for k in KS:
        print(f"\n  k={k} tareas cerradas")
        cols = [f"cob_{e}" for e in ETIQ]
        g = t[(t.nivel == .8) & (t.k == k)].set_index("metodo")[cols]
        g.columns = ETIQ
        g["desvio max"] = (g - .8).abs().max(axis=1)
        print(g.round(3).to_string())

    print(f"\n{S}\nVEREDICTO\n{S}")
    for k in KS:
        g = t[(t.nivel == .8) & (t.k == k)].set_index("metodo")
        cols = [f"cob_{e}" for e in ETIQ]
        desv = (g[cols] - .8).abs().max(axis=1)
        a_des, a_anch = float(desv["A · banda unica"]), float(
            g.loc["A · banda unica", "anchura_x"])
        mej = desv.idxmin()
        print(f"  k={k:2d}: el actual se desvia hasta {a_des:.3f} con anchura "
              f"x{a_anch:.2f}")
        print(f"        mejor: {mej} · desvio {desv[mej]:.3f} · anchura "
              f"x{g.loc[mej, 'anchura_x']:.2f} · cobertura global "
              f"{g.loc[mej, 'cobertura']:.3f}")
    json.dump({"proyectos": len(nom), "sorteos": SORTEOS, "lambda": LAMBDA,
               "cortes": CORTES}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
