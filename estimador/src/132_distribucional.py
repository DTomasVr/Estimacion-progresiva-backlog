"""E32 - Un solo modelo de la distribucion, y de ahi el estimador de cada metrica.

E30 y E31 juntos dicen algo preciso. El error no esta repartido: el modelo es
razonable en el centro (MdAElog 0,35 entre 1,5 y 3 h) y se rompe en los extremos
(1,85 abajo, 1,95 arriba). Y E31 muestra que esos dos extremos NO son iguales:

    microtarea   <= 0,25 h    AUC 0,617   practicamente ruido
    tarea grande >= 16 h      AUC 0,740   hay senal de sobra
    tarea grande >= 40 h      AUC 0,759

O sea: las tareas grandes SE VEN en el texto, y la regresion las esta aplastando
hasta 4 h porque bajo error cuadratico en log lo optimo es tirar hacia el centro.
Esa senal se esta tirando por culpa de la funcion de perdida, no por falta de
informacion.

La solucion no es otra regresion. Es dejar de estimar UN numero y estimar la
DISTRIBUCION condicional P(y | texto), que es lo que Gneiting (2011) viene
diciendo: un pronostico puntual solo tiene sentido respecto de una perdida, y de
una misma distribucion salen estimadores distintos y todos optimos.

    R2 y error cuadratico en log  -> la media de log y
    MdAElog, zeta de Morley       -> la mediana de log y
    MAE en horas                  -> la mediana en horas
    PRED(25) y factor 2           -> el centro de la ventana modal

Se modela con una red pequena y una softmax sobre la rejilla de valores, que es
la forma estandar de hacer regresion distribucional cuando el objetivo esta
cuantizado, y encima la GPU lo entrena en segundos con 3.000 rasgos, cosa que un
arbol multiclase de 44 clases no aguanta.

Rasgos: el titulo (e5-large), el e5 ajustado, TF-IDF reducido, la descripcion
(que E31 deja en +0,015 de R2 donde existe) y la prediccion de un regresor
auxiliar calculada FUERA DE MUESTRA con 5 pliegues. Eso ultimo arregla un fallo
de E28: alli el clasificador veia la prediccion en muestra, y por eso se fiaba de
ella mucho mas de lo que podria fiarse al desplegar.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/distribucional")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
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
                PRED25=float(np.mean(r <= .25)), PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                rho=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


class Red(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(.3),
                               nn.Linear(512, 256), nn.GELU(), nn.Dropout(.2),
                               nn.Linear(256, k))

    def forward(self, x):
        return self.f(x)


def entrenar(Xtr, ytr, Xva, yva, k, semilla, epocas=60):
    torch.manual_seed(semilla)
    m = Red(Xtr.shape[1], k).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epocas)
    lf = nn.CrossEntropyLoss(label_smoothing=.05)
    Xt = torch.tensor(Xtr, device=DEV)
    yt = torch.tensor(ytr, device=DEV)
    Xv = torch.tensor(Xva, device=DEV)
    yv = torch.tensor(yva, device=DEV)
    n, bs = len(Xt), 256
    mejor, mejor_est, paciencia = 1e9, None, 0
    for _ in range(epocas):
        m.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            lf(m(Xt[j]), yt[j]).backward()
            opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            v = float(nn.functional.cross_entropy(m(Xv), yv))
        if v < mejor - 1e-4:
            mejor = v
            mejor_est = {a: b.clone() for a, b in m.state_dict().items()}
            paciencia = 0
        else:
            paciencia += 1
            if paciencia >= 12:
                break
    m.load_state_dict(mejor_est)
    return m, mejor


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
    D = np.load(CACHE / "e5-large-desc.npy")
    tiene_d = sw.description.fillna("").astype(str).str.split().str.len().gt(0).to_numpy()
    D = np.where(tiene_d[:, None], D, 0).astype(np.float32)
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    Xd = np.hstack([E5, SV])
    Ms = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    Z = np.column_stack([np.mean([m.predict(Xd) for m in f["bolsa"]], 0),
                         f["ridge"].predict(Ms), f["svr"].predict(Ms),
                         f["hgb_ajustado"].predict(AJ)])
    lp = f["apilado"].predict(Z)
    print(f"datos listos · {time.time() - t0:.0f} s", flush=True)

    # ---- rasgo auxiliar FUERA DE MUESTRA (lo que le faltaba a E28) -----------
    aux = np.zeros(len(sw))
    Xa = np.hstack([E5, SV, AJ])
    idx = np.flatnonzero(tr)
    for a, b in KFold(5, shuffle=True, random_state=0).split(idx):
        r = HistGradientBoostingRegressor(max_iter=250, max_depth=6, learning_rate=.06,
                                          l2_regularization=1., random_state=0)
        r.fit(Xa[idx[a]], ly[idx[a]])
        aux[idx[b]] = r.predict(Xa[idx[b]])
    r = HistGradientBoostingRegressor(max_iter=250, max_depth=6, learning_rate=.06,
                                      l2_regularization=1., random_state=0)
    r.fit(Xa[tr], ly[tr])
    aux[~tr] = r.predict(Xa[~tr])
    print(f"rasgo auxiliar fuera de muestra listo · {time.time() - t0:.0f} s", flush=True)

    # ---- rejilla y clases ---------------------------------------------------
    rej = np.array(json.load(open("results/tasa_acierto/meta.json", encoding="utf-8"))["rejilla"])
    cl = np.abs(np.log(rej)[None, :] - ly[:, None]).argmin(1)
    usadas = np.array(sorted(np.unique(cl[tr])))
    mapa = {c: i for i, c in enumerate(usadas)}
    val = rej[usadas]
    lval = np.log(val)
    clm = np.array([mapa.get(c, -1) for c in cl])
    ok = clm >= 0
    print(f"rejilla {len(rej)} · clases usadas {len(usadas)} · "
          f"tareas fuera de rejilla {int((~ok).sum())}")

    X = np.hstack([E5, AJ, SV, D, aux[:, None]]).astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    print(f"matriz de rasgos {X.shape}")

    ktr = tr & ok
    kva = va & ok
    P = np.zeros((len(sw), len(usadas)))
    for s in range(5):
        m, v = entrenar(X[ktr], clm[ktr].astype(np.int64),
                        X[kva], clm[kva].astype(np.int64), len(usadas), s)
        m.eval()
        with torch.no_grad():
            P += torch.softmax(m(torch.tensor(X, device=DEV)), 1).cpu().numpy()
        print(f"  red {s} · NLL val {v:.4f} · {time.time() - t0:.0f} s", flush=True)
    P /= 5

    # ---- estimadores derivados de la MISMA distribucion ---------------------
    C = P.cumsum(1)
    med_log = lval[(C >= .5).argmax(1)]
    media_log = P @ lval
    med_h = val[(C >= .5).argmax(1)]
    media_h = P @ val

    def bayes(lo, hi):
        M = np.zeros((len(val), len(val)), bool)
        for i, c in enumerate(val):
            M[i] = (val >= c * lo) & (val <= c * hi)
        return val[(P @ M.T).argmax(1)]

    b25, b2 = bayes(.8, 1 / .75), bayes(.5, 2.)

    mejor_w, mejor_v = 0., 1e9
    for w in np.arange(0, 1.01, .05):
        v = float(np.mean((ly[va] - (w * media_log[va] + (1 - w) * lp[va])) ** 2))
        if v < mejor_v:
            mejor_w, mejor_v = float(w), v
    mez = mejor_w * media_log + (1 - mejor_w) * lp
    print(f"\nmezcla elegida en validacion: {mejor_w:.2f} distribucional + "
          f"{1 - mejor_w:.2f} modelo actual")

    brazos = {
        "0 · el modelo actual (apilado)": np.exp(lp),
        "1 · distribucional, media de log": np.exp(media_log),
        "2 · distribucional, mediana de log": np.exp(med_log),
        "3 · distribucional, mediana en horas": med_h,
        "4 · distribucional, media en horas": media_h,
        "5 · Bayes +-25 %": b25,
        "6 · Bayes factor 2": b2,
        f"7 · mezcla {mejor_w:.2f} con el actual": np.exp(mez),
    }
    filas = []
    for b, k in bl.items():
        for n, p in brazos.items():
            filas.append(dict(bloque=b, brazo=n, n=int(k.sum()), **metricas(y[k], p[k])))
    t = pd.DataFrame(filas)
    cols = ["brazo", "MAE_h", "MdAElog", "zeta_pc", "SSPB_pc", "R2log",
            "PRED25", "factor2", "rho"]
    for b in bl:
        print(f"\n{S}\n{b}\n{S}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    np.save(OUT / "P.npy", P.astype(np.float32))
    np.save(OUT / "val.npy", val)
    json.dump({"peso_mezcla": mejor_w, "clases": int(len(usadas))},
              open(OUT / "meta.json", "w", encoding="utf-8"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
