"""E41 - Caracterizar la analogia, que es lo unico que gana la tasa de acierto limpiamente.

E39 y E40 dejan un resultado que no esperaba: la estimacion por analogia -- la
mediana de las k tareas de entrenamiento con el titulo mas parecido -- tiene el
mejor PRED(25) de los once canales en validacion y en los tres bloques de prueba
(0,212 · 0,178 · 0,183 · 0,173), con un R2 de 0,038. No gana por poco y no es
sobreajuste de validacion.

El mecanismo esta claro: la mediana de k vecinos SIEMPRE cae sobre un valor que
de verdad se observa -- 1 h, 2 h, 4 h -- mientras que una media condicional cae
en 1,83 h, que no le acierta a nada. Con un objetivo tan cuantizado como este
(los veinte valores mas frecuentes son el 60,7 % de las tareas) eso es
justamente lo que mide PRED(25).

Antes de darle peso en la tesis hay que comprobar tres cosas:

  1  CUANTO DEPENDE DE k Y DE LA SIMILITUD. Si el resultado solo sale con k=15 y
     con e5-large, es una casualidad. Si aguanta en toda la rejilla, es el
     mecanismo.
  2  SI EL AGREGADOR IMPORTA. Mediana contra moda sobre la rejilla contra media.
     Si la moda gana a la mediana, confirma que lo que manda es caer sobre un
     atomo, no el estadistico concreto.
  3  SI SE PUEDE TENER LAS DOS COSAS. La dispersion de los vecinos es una medida
     de confianza gratis. Encaminar cada tarea al brazo que le convenga -- la
     analogia cuando los vecinos concuerdan, el apilado cuando no -- podria ganar
     en tasa de acierto sin tirar el R2, que es lo que no ha conseguido ningun
     brazo hasta ahora.
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binomtest

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/analogia")
S = "=" * 126


def metricas(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    q = np.log(p / y)
    r = np.abs(p - y) / y
    m = float(np.median(q))
    return dict(MAE_h=float(np.abs(p - y).mean()),
                MdAElog=float(np.median(np.abs(q))),
                zeta_pc=float((np.exp(np.median(np.abs(q))) - 1) * 100),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100),
                R2log=float(1 - ((np.log(y) - np.log(p)) ** 2).mean() / np.log(y).var()),
                PRED25=float(np.mean(r <= .25)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                rho=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def mcnemar(a, b):
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return "empate", 1.0, 0
    return ("mejor" if n01 > n10 else "peor" if n10 > n01 else "empate"), \
        float(binomtest(n01, n01 + n10, .5).pvalue), n01 - n10


def vecinos(V, tr, kmax):
    """Indices y similitudes de los kmax vecinos de entrenamiento de cada tarea."""
    N = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    A = N[tr]
    idx_tr = np.flatnonzero(tr)
    J = np.zeros((len(V), kmax), int)
    Sm = np.zeros((len(V), kmax))
    for i in range(0, len(V), 2048):
        s = N[i:i + 2048] @ A.T
        j = np.argpartition(-s, kmax, axis=1)[:, :kmax]
        sj = np.take_along_axis(s, j, 1)
        o = np.argsort(-sj, 1)
        J[i:i + 2048] = idx_tr[np.take_along_axis(j, o, 1)]
        Sm[i:i + 2048] = np.take_along_axis(sj, o, 1)
    return J, Sm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = joblib.load("models/estimador_pbe_v3.joblib")
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
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
    lp = f["apilado"].predict(np.column_stack([
        np.mean([m.predict(np.hstack([E5, SV])) for m in f["bolsa"]], 0),
        f["ridge"].predict(Xs), f["svr"].predict(Xs), f["hgb_ajustado"].predict(AJ)]))
    rej = np.array(sorted(pd.Series(y[tr]).value_counts().pipe(
        lambda c: c[c >= 60]).index.to_numpy()))
    print(f"datos listos · rejilla de {len(rej)} atomos · {time.time() - t0:.0f} s",
          flush=True)

    # ---------------------------------------------------- 1 y 2 · k, similitud, agregador
    KMAX = 50
    vistas = {"e5-large": E5, "e5 ajustado": AJ, "TF-IDF (SVD)": SV,
              "e5-large + ajustado": np.hstack([E5, AJ])}
    filas, guardado = [], {}
    for nv, V in vistas.items():
        J, Sm = vecinos(V, tr, KMAX)
        for k in (5, 10, 15, 25, 50):
            yv = ly[J[:, :k]]
            med = np.exp(np.median(yv, 1))
            for ag, p in (("mediana", med),
                          ("media", np.exp(yv.mean(1))),
                          ("moda en la rejilla",
                           rej[np.abs(np.log(rej)[None, :] -
                                      np.log(med)[:, None]).argmin(1)])):
                d = dict(vista=nv, k=k, agregador=ag,
                         val_PRED25=metricas(y[va], p[va])["PRED25"])
                for b, m in bl.items():
                    d[f"{b}_PRED25"] = metricas(y[m], p[m])["PRED25"]
                    d[f"{b}_R2"] = metricas(y[m], p[m])["R2log"]
                filas.append(d)
                guardado[(nv, k, ag)] = p
        print(f"  {nv} listo · {time.time() - t0:.0f} s", flush=True)
    t1 = pd.DataFrame(filas)
    print(f"\n{S}\n1 y 2 - CUANTO DEPENDE DE k, DE LA SIMILITUD Y DEL AGREGADOR\n{S}")
    print("  (PRED25 por bloque · la columna de validacion es la que se puede usar para elegir)")
    piv = t1.pivot_table(index=["vista", "agregador"], columns="k",
                         values="prueba fija_PRED25")
    print(piv.round(4).to_string())
    print(f"\n  mejor combinacion en VALIDACION: ", end="")
    mej = t1.loc[t1.val_PRED25.idxmax()]
    print(f"{mej.vista} · k={mej.k} · {mej.agregador}  "
          f"(val {mej.val_PRED25:.4f} -> fija {mej['prueba fija_PRED25']:.4f})")

    # ---------------------------------------------------- 3 · encaminar por confianza
    nv, k, ag = mej.vista, int(mej.k), mej.agregador
    J, Sm = vecinos(vistas[nv], tr, KMAX)
    ana = guardado[(nv, k, ag)]
    disp = ly[J[:, :k]].std(1)                       # desacuerdo entre los vecinos
    print(f"\n{S}\n3 - ENCAMINAR CADA TAREA AL BRAZO QUE LE CONVIENE\n{S}")
    print(f"  dispersion de los vecinos: mediana {np.median(disp[va]):.3f} · "
          f"correlacion con |error| de la analogia "
          f"{np.corrcoef(disp[va], np.abs(np.log(ana[va]) - ly[va]))[0, 1]:+.3f}")
    mejor_u, mejor_v = None, -1
    for u in np.quantile(disp[va], np.arange(.05, 1.0, .05)):
        p = np.where(disp <= u, ana, np.exp(lp))
        v = metricas(y[va], p[va])["PRED25"]
        if v > mejor_v:
            mejor_u, mejor_v = float(u), v
    frac = float((disp <= mejor_u).mean())
    print(f"  umbral elegido en validacion: dispersion <= {mejor_u:.3f}, que encamina "
          f"el {frac:.1%} de las tareas a la analogia")
    brazos = {"0 · el apilado actual": np.exp(lp),
              f"1 · analogia ({nv}, k={k}, {ag})": ana,
              "2 · encaminado por dispersion de los vecinos":
                  np.where(disp <= mejor_u, ana, np.exp(lp)),
              "3 · media geometrica de los dos": np.exp((np.log(ana) + lp) / 2)}
    filas, contr = [], []
    for n, p in brazos.items():
        filas.append(dict(bloque="validacion", brazo=n, n=int(va.sum()),
                          **metricas(y[va], p[va])))
    for b, m in bl.items():
        b25 = np.abs(np.exp(lp[m]) - y[m]) / y[m] <= .25
        bf2 = np.maximum(np.exp(lp[m]) / y[m], y[m] / np.exp(lp[m])) <= 2
        for n, p in brazos.items():
            filas.append(dict(bloque=b, brazo=n, n=int(m.sum()), **metricas(y[m], p[m])))
            v1, p1, d1 = mcnemar(np.abs(p[m] - y[m]) / y[m] <= .25, b25)
            v2, p2, d2 = mcnemar(np.maximum(p[m] / y[m], y[m] / p[m]) <= 2, bf2)
            contr.append(dict(bloque=b, brazo=n,
                              pred25_vs_apilado=f"{v1} (p={p1:.3g}, {d1:+d})",
                              factor2_vs_apilado=f"{v2} (p={p2:.3g}, {d2:+d})"))
    t = pd.DataFrame(filas)
    c = pd.DataFrame(contr)
    cols = ["brazo", "MAE_h", "MdAElog", "zeta_pc", "R2log", "PRED25", "factor2", "rho"]
    for b in ["validacion"] + list(bl):
        print(f"\n{S}\n{b}\n{S}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
        if b in bl:
            print(c[c.bloque == b].drop(columns=["bloque"]).to_string(index=False))
    t1.to_csv(OUT / "rejilla.csv", index=False)
    t.to_csv(OUT / "resumen.csv", index=False)
    c.to_csv(OUT / "contrastes.csv", index=False)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
