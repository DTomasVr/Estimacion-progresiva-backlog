"""E29 · El panel de metricas que usa el estado del arte, aplicado a nuestro modelo.

Buscando que reporta la literatura para "predecir horas a partir de texto" en
cualquier industria, aparecen tres familias y una metrica que encaja con este
problema mejor que todo lo que veniamos usando.

  1 LO QUE REPORTA TODO EL MUNDO (monitorizacion de procesos, duracion de
    incidentes de trafico, duracion de ensayos clinicos, construccion):
    MAE, RMSE y MAPE. Interpretables y en las unidades del problema.

  2 LO QUE PIDE LA METODOLOGIA DE ESTIMACION DE ESFUERZO desde Shepperd y
    MacDonell (2012) y Foss et al. (2003): MMRE y PRED estan deprecados por
    sesgados -- MMRE premia subestimar -- y se recomiendan el error absoluto
    medio, la exactitud estandarizada SA frente al azar, y un tamaño del efecto.

  3 LO QUE USA LA GEOCIENCIA para datos positivos y sesgados, que es justo el
    caso de las horas: Morley, Brito y Welling (2018) proponen la EXACTITUD
    SIMETRICA MEDIANA, zeta = exp(mediana|log(pred/real)|) - 1, y el SESGO
    PORCENTUAL SIMETRICO SSPB. Es exactamente nuestro MdAElog expresado en
    porcentaje, con la ventaja de que se lee como "el error porcentual tipico"
    y de que penaliza igual pasarse que quedarse corto.

Se calcula todo sobre el modelo final y sobre las dos palancas, y se añade el
tamaño del efecto (delta de Glass) frente al azar, que es lo que Shepperd y
MacDonell piden junto al SA.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)
DATA = "data/processed/agix_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
ART = Path("models/estimador_agix_v3.joblib")
OUT = Path("results/panel")
SEP = "=" * 150


def azar(y, corridas=1000, semilla=0):
    """Adivinar al azar: predecir cada tarea con OTRA tarea real del mismo bloque.

    Devuelve (MAR esperado, desviacion de los errores absolutos). El tamaño del
    efecto de Shepperd y MacDonell es delta = (MAR_azar - MAR_modelo) / s, y esa
    s tiene que ser la dispersion de los ERRORES del azar, no la de las medias de
    cada corrida: la media de 4.000 errores es estabilisima y dividir por su
    desviacion da deltas de 18, cuando en esa literatura 0,5 ya es un efecto
    grande. Con la dispersion de los errores la delta se lee como una d de Cohen.
    """
    rng = np.random.default_rng(semilla)
    n = len(y)
    mars, err = [], []
    for c in range(corridas):
        i = rng.integers(0, n - 1, n)
        i += i >= np.arange(n)
        e = np.abs(y - y[i])
        mars.append(e.mean())
        if c < 50:
            err.append(e)
    return float(np.mean(mars)), float(np.std(np.concatenate(err)))


def panel(y, p, ref=None):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    e = np.abs(y - p)
    q = np.log(p / y)
    mar = float(e.mean())
    d = dict()
    # 1 · lo que reporta todo el mundo
    d["MAE_h"] = mar
    d["RMSE_h"] = float(np.sqrt(np.mean((y - p) ** 2)))
    d["MdAE_h"] = float(np.median(e))
    d["MAPE_%"] = float(np.mean(e / y) * 100)
    d["MdAPE_%"] = float(np.median(e / y) * 100)      # = MdMRE en porcentaje
    # 2 · metodologia de estimacion de esfuerzo
    if ref is not None:
        m0, s0 = ref
        d["SA"] = float(1 - mar / m0)
        d["delta_Glass"] = float((m0 - mar) / s0)
    d["PRED25_%"] = float(np.mean(e / y <= .25) * 100)
    d["PRED50_%"] = float(np.mean(e / y <= .50) * 100)
    d["LSD"] = float(np.sqrt(np.mean((q + np.var(q) / 2) ** 2)))   # desv. log estandar
    # 3 · geociencia, datos positivos y sesgados (Morley et al. 2018)
    d["zeta_%"] = float((np.exp(np.median(np.abs(q))) - 1) * 100)
    m = float(np.median(q))
    d["SSPB_%"] = float(np.sign(m) * (np.exp(abs(m)) - 1) * 100)
    # 4 · capacidad explicativa y de orden
    ly = np.log(y)
    d["R2_log"] = float(1 - ((ly - np.log(p)) ** 2).mean() / ly.var())
    d["spearman"] = float(pd.Series(y).corr(pd.Series(p), method="spearman"))
    d["factor2_%"] = float(np.mean(np.maximum(p / y, y / p) <= 2) * 100)
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = joblib.load(ART)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    E5 = np.load(CACHE / "e5-large.npy")
    AJ = np.load(CACHE / "e5-ajustado.npy")
    Ms = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    Xd = np.hstack([E5] + [sv.transform(tf.transform(tit)).astype(np.float32)
                           for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    Z = np.column_stack([np.mean([m.predict(Xd) for m in f["bolsa"]], 0),
                         f["ridge"].predict(Ms), f["svr"].predict(Ms),
                         f["hgb_ajustado"].predict(AJ)])
    lp = f["apilado"].predict(Z)
    p = np.exp(lp)
    sw["est"] = sw.HoursEstimate.where(sw.HoursEstimate > 0)
    pt = sw[(sw.source == "TAWOS") & (sw.story_point > 0)]
    pe, co = np.polyfit(np.log(pt.story_point.to_numpy(float)),
                        np.log(pt.effort_hours.to_numpy(float)), 1)
    sw.loc[pt.index, "est"] = np.exp(co) * pt.story_point.to_numpy(float) ** pe
    est = sw.est.to_numpy(float)
    hay = ~np.isnan(est)
    Zc = np.hstack([E5, np.log(np.where(hay, est, 1))[:, None], lp[:, None]])
    pc = np.where(hay, np.where(hay, est, 1) * np.exp(f["corrector"].predict(Zc)), p)
    bayes = joblib.load("models/bayes_tasa.joblib") if Path("models/bayes_tasa.joblib").exists() else None
    print(f"modelo aplicado · {time.time() - t0:.0f} s")

    bl = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
          "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
          "externo no Apache": (sw.split_t3 == "externo").to_numpy()}
    filas = []
    for b, k in bl.items():
        ma = azar(y[k])
        for etq, pred in (("modelo en frío", p), ("con estimación del usuario", pc)):
            filas.append(dict(bloque=b, brazo=etq, n=int(k.sum()),
                              **panel(y[k], pred[k], ma)))
        filas.append(dict(bloque=b, brazo="adivinar al azar (referencia)", n=int(k.sum()),
                          **panel(y[k], np.full(k.sum(), float(np.median(y[k]))), ma)))
    t = pd.DataFrame(filas)
    g1 = ["MAE_h", "RMSE_h", "MdAE_h", "MAPE_%", "MdAPE_%"]
    g2 = ["SA", "delta_Glass", "PRED25_%", "PRED50_%", "LSD"]
    g3 = ["zeta_%", "SSPB_%", "R2_log", "spearman", "factor2_%"]
    for b in bl:
        v = t[t.bloque == b]
        print(f"\n{SEP}\n{b}  (n={v.n.iloc[0]:,})\n{SEP}")
        print("1 · lo que reporta todo el mundo (MAE, RMSE, MAPE)")
        print(v[["brazo"] + g1].round(3).to_string(index=False))
        print("\n2 · lo que pide la metodología de estimación de esfuerzo")
        print(v[["brazo"] + g2].round(3).to_string(index=False))
        print("\n3 · exactitud simétrica (Morley 2018) y capacidad explicativa")
        print(v[["brazo"] + g3].round(3).to_string(index=False))
    t.to_csv(OUT / "panel.csv", index=False)
    json.dump({"nota": "zeta = exp(mediana|log(pred/real)|)-1 · SSPB = sesgo porcentual simétrico"},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/panel.csv")


if __name__ == "__main__":
    main()
