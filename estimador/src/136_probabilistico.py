"""E36 - Evaluar el pronostico como lo que es: una distribucion, no un numero.

E32 y E33 dejan algo que el proyecto no tenia: una distribucion condicional
P(horas | texto) para cada tarea. Eso permite evaluar con las reglas de
puntuacion propias del pronostico probabilistico, que son las que usan la
meteorologia, la hidrologia y la prediccion de demanda, y que estan por encima
de lo que reporta la literatura de estimacion de esfuerzo, donde practicamente
todo el mundo se queda en un numero y su error.

Gneiting y Raftery (2007) fijan el criterio: maximizar la NITIDEZ sujeto a
CALIBRACION. O sea, el intervalo mas estrecho que siga cubriendo lo que dice
cubrir. Se mide con tres cosas:

  CRPS       regla de puntuacion propia que generaliza el error absoluto al caso
             de una distribucion. Es comparable entre modelos y, a diferencia de
             las metricas puntuales, no se puede mejorar haciendo trampa con el
             funcional que se reporta
  PIT        el valor de la funcion de distribucion en el dato observado. Si el
             modelo esta bien calibrado, los PIT se reparten uniformes; si salen
             en forma de U, el modelo es demasiado estrecho (exceso de confianza)
  COBERTURA  que fraccion cae dentro del intervalo al 50, 80 y 90 %, y cuanto
             mide ese intervalo. Es la lectura que le sirve al usuario del
             producto: "entre 2 y 9 horas, con un 80 % de confianza"

Referencias obligadas de comparacion, porque un CRPS suelto no dice nada:

  climatologia   la distribucion empirica del entrenamiento, igual para todos.
                 Es el equivalente probabilistico del predictor constante
  lognormal fija el apilado como centro y una sigma constante ajustada en
                 validacion. Si el modelo condicional no le gana a esto, no esta
                 aprendiendo nada sobre la incertidumbre, solo sobre la posicion
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kstest, norm

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/probabilistico")
S = "=" * 122


def crps(sop, P, y):
    """CRPS exacto de una distribucion discreta: E|X-y| - 0,5 E|X-X'|.

    sop es el soporte (n, k) o (k,), P las masas (n, k), y el observado (n,)."""
    sop = np.atleast_2d(sop)
    if sop.shape[0] == 1:
        sop = np.repeat(sop, len(y), 0)
    t1 = (P * np.abs(sop - y[:, None])).sum(1)
    o = np.argsort(sop, 1)
    s = np.take_along_axis(sop, o, 1)
    p = np.take_along_axis(P, o, 1)
    c = p.cumsum(1)
    # E|X-X'| = 2 * suma_j (s_{j+1}-s_j) * C_j * (1-C_j)
    t2 = 2 * ((np.diff(s, axis=1)) * c[:, :-1] * (1 - c[:, :-1])).sum(1)
    return t1 - .5 * t2


def pit(sop, P, y):
    """PIT con aleatorizacion, que es lo que corresponde con soporte discreto."""
    sop = np.atleast_2d(sop)
    if sop.shape[0] == 1:
        sop = np.repeat(sop, len(y), 0)
    o = np.argsort(sop, 1)
    s = np.take_along_axis(sop, o, 1)
    p = np.take_along_axis(P, o, 1)
    c = p.cumsum(1)
    menor = np.where(s < y[:, None], p, 0).sum(1)
    igual = np.where(np.isclose(s, y[:, None]), p, 0).sum(1)
    u = np.random.default_rng(0).random(len(y))
    return np.clip(menor + u * igual, 0, 1)


def cuantil(sop, P, y, q):
    sop = np.atleast_2d(sop)
    if sop.shape[0] == 1:
        sop = np.repeat(sop, len(y), 0)
    o = np.argsort(sop, 1)
    s = np.take_along_axis(sop, o, 1)
    p = np.take_along_axis(P, o, 1)
    return np.take_along_axis(s, (p.cumsum(1) >= q).argmax(1)[:, None], 1).ravel()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = joblib.load("models/estimador_agix_v3.joblib")
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    tr = (sw.split_t3 == "train").to_numpy()
    va = (sw.split_t3 == "val").to_numpy()
    bl = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
          "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
          "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    E5 = np.load(CACHE / "e5-large.npy")
    AJ = np.load(CACHE / "e5-ajustado.npy")
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    Xs = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    Z = np.column_stack([np.mean([m.predict(np.hstack([E5, SV])) for m in f["bolsa"]], 0),
                         f["ridge"].predict(Xs), f["svr"].predict(Xs),
                         f["hgb_ajustado"].predict(AJ)])
    lp = f["apilado"].predict(Z)

    Pabs = np.load("results/distribucional/P.npy").astype(float)
    val = np.load("results/distribucional/val.npy")
    Pres = np.load("results/forma_residual/P_residuo.npy").astype(float)
    bordes = np.linspace(-4.0, 4.0, 42)
    centros = (bordes[:-1] + bordes[1:]) / 2
    lp_h = np.load("results/forma_residual/lp_honesto.npy")

    # climatologia: la distribucion empirica del entrenamiento sobre la rejilla
    clim = np.zeros(len(val))
    j = np.abs(np.log(val)[None, :] - ly[tr][:, None]).argmin(1)
    for k in j:
        clim[k] += 1
    clim /= clim.sum()
    Pclim = np.repeat(clim[None, :], len(sw), 0)

    # lognormal centrada en el apilado con sigma constante elegida en validacion
    mejor_s, mejor_v = None, 1e9
    rej_l = np.log(val)
    for s in np.arange(.6, 2.01, .05):
        P = np.exp(-.5 * ((rej_l[None, :] - lp[va][:, None]) / s) ** 2)
        P /= P.sum(1, keepdims=True)
        v = float(crps(rej_l, P, ly[va]).mean())
        if v < mejor_v:
            mejor_s, mejor_v = float(s), v
    Plog = np.exp(-.5 * ((rej_l[None, :] - lp[:, None]) / mejor_s) ** 2)
    Plog /= Plog.sum(1, keepdims=True)
    print(f"sigma de la lognormal de referencia elegida en validacion: {mejor_s:.2f}")

    sop_res = lp_h[:, None] + centros[None, :]
    modelos = {
        "climatologia (el constante probabilistico)": (np.repeat(rej_l[None, :], len(sw), 0), Pclim),
        f"lognormal sobre el apilado, sigma {mejor_s:.2f}": (np.repeat(rej_l[None, :], len(sw), 0), Plog),
        "distribucional directo (E32)": (np.repeat(np.log(val)[None, :], len(sw), 0), Pabs),
        "apilado + forma del residuo (E33)": (sop_res, Pres),
    }
    filas = []
    for b, k in bl.items():
        for n, (sop, P) in modelos.items():
            sk, Pk, yk = sop[k], P[k], ly[k]
            u = pit(sk, Pk, yk)
            d = dict(bloque=b, modelo=n, n=int(k.sum()),
                     CRPS_log=float(crps(sk, Pk, yk).mean()),
                     KS_PIT=float(kstest(u, "uniform").statistic),
                     p_KS=float(kstest(u, "uniform").pvalue))
            for q, etq in ((.5, "50"), (.8, "80"), (.9, "90")):
                a = (1 - q) / 2
                lo, hi = cuantil(sk, Pk, yk, a), cuantil(sk, Pk, yk, 1 - a)
                d[f"cob{etq}"] = float(np.mean((yk >= lo) & (yk <= hi)))
                d[f"anch{etq}_x"] = float(np.exp(np.median(hi - lo)))
            filas.append(d)
    t = pd.DataFrame(filas)
    c1 = ["modelo", "CRPS_log", "KS_PIT", "p_KS"]
    c2 = ["modelo", "cob50", "anch50_x", "cob80", "anch80_x", "cob90", "anch90_x"]
    for b in bl:
        v = t[t.bloque == b]
        print(f"\n{S}\n{b}  (n={v.n.iloc[0]:,})\n{S}")
        print("puntuacion propia y calibracion  (CRPS menor es mejor · KS pequeno = PIT uniforme = bien calibrado)")
        print(v[c1].round(4).to_string(index=False))
        print("\ncobertura de los intervalos y su anchura, en factor multiplicativo")
        print(v[c2].round(3).to_string(index=False))
        print("  cobXX deberia salir XX/100 · anchXX_x es cuantas veces mas grande es el extremo alto que el bajo")
    t.to_csv(OUT / "resumen.csv", index=False)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
