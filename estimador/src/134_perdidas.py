"""E34 - Entrenamos con una perdida y reportamos otra. Eso cuesta metricas gratis.

Todo el modelo esta entrenado con error cuadratico sobre log-horas. Pero de las
metricas que reportamos, esa perdida solo es la correcta para una: el R2 en log.
Las demas quieren otro funcional de la misma distribucion condicional.

    R2 en log                       media condicional de log y   -> error cuadratico
    MdAElog y zeta de Morley        MEDIANA condicional de log y -> error ABSOLUTO
    MAE en horas                    mediana condicional en horas -> error absoluto en horas
    sesgo SSPB                      mediana, otra vez

O sea que llevamos toda la tesis optimizando la media cuando la metrica que mas
citamos, la exactitud simetrica mediana, pide la mediana. Con una distribucion
tan asimetrica como esta -- log-horas con cola larga a la derecha -- media y
mediana no son lo mismo ni de lejos, y el hueco se paga entero en el MdAElog.

Se prueban seis perdidas sobre los MISMOS rasgos y las MISMAS particiones, y se
reporta el panel entero para cada una. Si la teoria se cumple, cada perdida debe
ganar su metrica y perder las demas; y eso, ademas de subir cifras, es una
demostracion empirica limpia del argumento de Gneiting sobre nuestros datos.

La quinta perdida viene de la literatura de regresion desequilibrada (Yang et al.
2021, "Delving into Deep Imbalanced Regression"): pesar cada tarea por el inverso
de la densidad suavizada de su objetivo, para que la cola deje de ser invisible.
Merece la pena probarla porque E30 muestra justo ese fallo -- el decil de arriba
tiene MdAElog 1,95 -- aunque hay una razon teorica para desconfiar, y es que el
remedio esta pensado para cuando la distribucion de prueba es mas uniforme que la
de entrenamiento, y aqui son la misma.
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/perdidas")
S = "=" * 128


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
                PRED25=float(np.mean(r <= .25)), factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                rho=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def pesos_lds(ly, sigma=.5, n=50):
    """Pesos de Yang et al. 2021: inverso de la densidad suavizada del objetivo."""
    b = np.linspace(ly.min(), ly.max(), n + 1)
    c = (b[:-1] + b[1:]) / 2
    h, _ = np.histogram(ly, bins=b)
    k = np.exp(-.5 * ((c[:, None] - c[None, :]) / sigma) ** 2)
    k /= k.sum(1, keepdims=True)
    dens = k @ h.astype(float)
    w = 1 / np.maximum(dens, 1e-9)
    j = np.clip(np.digitize(ly, b) - 1, 0, n - 1)
    w = w[j]
    return w / w.mean()


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
    bl = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
          "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
          "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    X = np.hstack([np.load(CACHE / "e5-large.npy")] +
                  [sv.transform(tf.transform(tit)).astype(np.float32)
                   for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    print(f"rasgos {X.shape} · {time.time() - t0:.0f} s", flush=True)

    def hgb(**kw):
        return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                             l2_regularization=1., random_state=0, **kw)

    w = pesos_lds(ly[tr])
    print(f"pesos LDS: min {w.min():.2f} · max {w.max():.2f} · "
          f"la tarea mas rara pesa {w.max() / w.min():.0f} veces mas que la mas comun")

    brazos = {}
    defs = [
        ("A · cuadratica en log (lo actual)", dict(), ly, None, np.exp),
        ("B · ABSOLUTA en log", dict(loss="absolute_error"), ly, None, np.exp),
        ("C · cuantil 0,5 en log", dict(loss="quantile", quantile=.5), ly, None, np.exp),
        ("D · absoluta en HORAS", dict(loss="absolute_error"), y, None, lambda v: v),
        ("E · cuadratica en log con pesos LDS", dict(), ly, w, np.exp),
        ("F · devianza gamma en horas", dict(loss="gamma"), y, None, lambda v: v),
    ]
    for etq, kw, obj, pes, inv in defs:
        m = hgb(**kw)
        m.fit(X[tr], obj[tr], sample_weight=pes)
        brazos[etq] = inv(m.predict(X))
        print(f"  {etq:38s} listo · {time.time() - t0:.0f} s", flush=True)

    # cuantiles, para ver si desplazar el funcional ayuda a las tolerancias
    for q in (.35, .40, .45):
        m = hgb(loss="quantile", quantile=q).fit(X[tr], ly[tr])
        brazos[f"G · cuantil {q:.2f} en log"] = np.exp(m.predict(X))
        print(f"  cuantil {q:.2f} listo · {time.time() - t0:.0f} s", flush=True)

    filas = []
    for b, k in bl.items():
        for n, p in brazos.items():
            filas.append(dict(bloque=b, brazo=n, n=int(k.sum()), **metricas(y[k], p[k])))
    t = pd.DataFrame(filas)
    cols = ["brazo", "MAE_h", "MdAElog", "zeta_pc", "SSPB_pc", "R2log", "PRED25", "factor2", "rho"]
    for b in bl:
        v = t[t.bloque == b]
        print(f"\n{S}\n{b}\n{S}")
        print(v[cols].round(4).to_string(index=False))
        print("  mejor en cada metrica: " + " · ".join(
            f"{c} -> {v.loc[v[c].idxmax() if c in ('R2log', 'PRED25', 'factor2', 'rho') else v[c].idxmin(), 'brazo'].split(' · ')[0]}"
            for c in ("MAE_h", "MdAElog", "R2log", "PRED25", "factor2", "rho")))
    t.to_csv(OUT / "resumen.csv", index=False)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
