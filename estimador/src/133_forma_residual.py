"""E33 - Separar la localizacion de la forma: apilado para el donde, distribucion para el cuanto de incertidumbre.

E32 deja un resultado util y a medias. El modelo distribucional gana en las
metricas de tolerancia (PRED25 0,167 -> 0,186, factor 2 0,446 -> 0,467 en la
prueba fija) y pierde en las de error (R2log 0,199 -> 0,093). La lectura es que
la distribucion sabe mejor QUE FORMA tiene la incertidumbre y peor DONDE esta
centrada: una red de 3.585 rasgos sobre 16.600 filas no le gana al apilado
estimando la posicion, pero si le aporta algo sobre la dispersion y la asimetria.

Asi que no hay que elegir. Se descompone en localizacion y forma:

    log y = lp(x)   +   r
            ^^^^^       ^
            el apilado  el residuo, cuya DISTRIBUCION se modela condicionada a x

Con eso, la distribucion de log y es la del residuo desplazada hasta lp(x), y de
ahi sale cada estimador sin renunciar a la buena localizacion del apilado. Es la
descomposicion de localizacion y escala de toda la vida, solo que la forma no se
supone constante: se aprende, que es justo lo que E31 dice que se puede hacer
(las tareas grandes se ven, AUC 0,740).

Un detalle que no es opcional: el residuo del entrenamiento tiene que calcularse
FUERA DE MUESTRA. Si se usa el residuo en muestra, es artificialmente pequeno y
el modelo de forma aprende una dispersion que no existira al desplegar. Aqui se
reconstruye el apilado entero con 5 pliegues -- bolsa de siete, los dos lineales
dispersos y el codificador ajustado -- lo que cuesta unos quince minutos y es la
diferencia entre medir la incertidumbre y inventarsela.
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
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/forma_residual")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
S = "=" * 128
N_BOLSA = 7
BORDES = np.linspace(-4.0, 4.0, 42)          # 41 tramos para el residuo en log
CENTROS = (BORDES[:-1] + BORDES[1:]) / 2


def hgb(semilla=0, **kw):
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=semilla, **kw)


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


def mcnemar(a, b):
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return "empate", 1.0, 0
    p = binomtest(n01, n01 + n10, .5).pvalue
    return ("mejor" if n01 > n10 else "peor" if n10 > n01 else "empate"), float(p), n01 - n10


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
    mejor, mejor_est, pac = 1e9, None, 0
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
            mejor, pac = v, 0
            mejor_est = {a: b.clone() for a, b in m.state_dict().items()}
        else:
            pac += 1
            if pac >= 12:
                break
    m.load_state_dict(mejor_est)
    return m, mejor


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
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
    D = np.load(CACHE / "e5-large-desc.npy")
    tiene_d = sw.description.fillna("").astype(str).str.split().str.len().gt(0).to_numpy()
    D = np.where(tiene_d[:, None], D, 0).astype(np.float32)
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    Xd = np.hstack([E5, SV])
    Xs = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    Z = np.column_stack([np.mean([m.predict(Xd) for m in f["bolsa"]], 0),
                         f["ridge"].predict(Xs), f["svr"].predict(Xs),
                         f["hgb_ajustado"].predict(AJ)])
    lp = f["apilado"].predict(Z)
    print(f"datos y modelo actual listos · {time.time() - t0:.0f} s", flush=True)

    # ---------------------------------------------------------------- OOF del apilado
    cache = OUT / "lp_honesto.npy"
    if cache.exists():
        lp_h = np.load(cache)
        print("  lp fuera de muestra en cache")
    else:
        lp_h = lp.copy()
        idx = np.flatnonzero(tr)
        for nf, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(idx)):
            ia, ib = idx[a], idx[b]
            rng = np.random.default_rng(nf)
            bolsa = []
            for s in range(N_BOLSA):
                sub = rng.choice(ia, int(.85 * len(ia)), replace=False)
                bolsa.append(hgb(s, max_features=.8).fit(Xd[sub], ly[sub]))
            lr = Ridge(alpha=2.0).fit(Xs[ia], ly[ia])
            ls = LinearSVR(C=.3, max_iter=4000, random_state=0).fit(Xs[ia], ly[ia])
            ha = hgb().fit(AJ[ia], ly[ia])

            def base(sel):
                return np.column_stack([np.mean([m.predict(Xd[sel]) for m in bolsa], 0),
                                        lr.predict(Xs[sel]), ls.predict(Xs[sel]),
                                        ha.predict(AJ[sel])])
            ap = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(base(va), ly[va])
            lp_h[ib] = ap.predict(base(ib))
            print(f"  pliegue {nf + 1}/5 · {time.time() - t0:.0f} s", flush=True)
        np.save(cache, lp_h)
    r_tr = ly[tr] - lp_h[tr]
    print(f"residuo fuera de muestra en entrenamiento: sd {r_tr.std():.3f} · "
          f"en muestra {float((ly[tr] - lp[tr]).std()):.3f}  "
          f"(si el de en muestra fuese mucho menor, medirlo asi era imprescindible)")

    # ---------------------------------------------------------------- forma del residuo
    res = ly - lp_h
    cl = np.clip(np.digitize(res, BORDES) - 1, 0, len(CENTROS) - 1)
    X = np.hstack([E5, AJ, SV, D, lp_h[:, None]]).astype(np.float32)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    print(f"matriz {X.shape} · {len(CENTROS)} tramos de residuo", flush=True)

    P = np.zeros((len(sw), len(CENTROS)))
    for s in range(5):
        m, v = entrenar(X[tr], cl[tr].astype(np.int64), X[va], cl[va].astype(np.int64),
                        len(CENTROS), s)
        m.eval()
        with torch.no_grad():
            P += torch.softmax(m(torch.tensor(X, device=DEV)), 1).cpu().numpy()
        print(f"  red {s} · NLL val {v:.4f} · {time.time() - t0:.0f} s", flush=True)
    P /= 5

    # distribucion absoluta: soporte lp_i + CENTROS, con la misma masa P
    C = P.cumsum(1)
    media_r = P @ CENTROS
    med_r = CENTROS[(C >= .5).argmax(1)]

    rej = np.array(json.load(open("results/tasa_acierto/meta.json", encoding="utf-8"))["rejilla"])
    lrej = np.log(rej)

    def bayes(lo, hi):
        """Para cada candidato de la rejilla, masa de la distribucion absoluta
        que cae dentro de su ventana. Vectorizado sobre tareas."""
        mejor = np.zeros(len(sw))
        mejor_m = np.full(len(sw), -1.0)
        for j, la in enumerate(lrej):
            dentro = ((lp_h[:, None] + CENTROS[None, :] >= la + np.log(lo)) &
                      (lp_h[:, None] + CENTROS[None, :] <= la + np.log(hi)))
            masa = (P * dentro).sum(1)
            k = masa > mejor_m
            mejor_m[k], mejor[k] = masa[k], rej[j]
        return mejor

    b25 = bayes(.8, 1 / .75)
    b2 = bayes(.5, 2.)
    print(f"Bayes sobre la distribucion desplazada · {time.time() - t0:.0f} s", flush=True)

    cte25 = float(json.load(open("results/tasa_acierto/meta.json", encoding="utf-8"))["constante_25"])
    cte2 = float(json.load(open("results/tasa_acierto/meta.json", encoding="utf-8"))["constante_f2"])
    brazos = {
        "0 · el modelo actual (apilado)": np.exp(lp),
        "1 · apilado + media del residuo": np.exp(lp + media_r),
        "2 · apilado + mediana del residuo": np.exp(lp + med_r),
        "3 · Bayes +-25 % sobre la desplazada": b25,
        "4 · Bayes factor 2 sobre la desplazada": b2,
        f"control · constante {cte25:.2f} h": np.full(len(sw), cte25),
        f"control · constante {cte2:.2f} h": np.full(len(sw), cte2),
    }
    filas, contr = [], []
    for b, k in bl.items():
        ref25 = np.abs(cte25 - y[k]) / y[k] <= .25
        ref2 = np.maximum(cte2 / y[k], y[k] / cte2) <= 2
        base25 = np.abs(np.exp(lp[k]) - y[k]) / y[k] <= .25
        base2 = np.maximum(np.exp(lp[k]) / y[k], y[k] / np.exp(lp[k])) <= 2
        for n, p in brazos.items():
            filas.append(dict(bloque=b, brazo=n, n=int(k.sum()), **metricas(y[k], p[k])))
            h25 = np.abs(p[k] - y[k]) / y[k] <= .25
            h2 = np.maximum(p[k] / y[k], y[k] / p[k]) <= 2
            v1, p1, d1 = mcnemar(h25, ref25)
            v2, p2, d2 = mcnemar(h2, ref2)
            v3, p3, d3 = mcnemar(h25, base25)
            v4, p4, d4 = mcnemar(h2, base2)
            contr.append(dict(bloque=b, brazo=n,
                              pred25_vs_cte=f"{v1} (p={p1:.3g}, {d1:+d})",
                              factor2_vs_cte=f"{v2} (p={p2:.3g}, {d2:+d})",
                              pred25_vs_base=f"{v3} (p={p3:.3g}, {d3:+d})",
                              factor2_vs_base=f"{v4} (p={p4:.3g}, {d4:+d})"))
    t = pd.DataFrame(filas)
    c = pd.DataFrame(contr)
    cols = ["brazo", "MAE_h", "MdAElog", "zeta_pc", "SSPB_pc", "R2log", "PRED25", "factor2", "rho"]
    for b in bl:
        print(f"\n{S}\n{b}\n{S}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
        print(c[c.bloque == b].drop(columns=["bloque"]).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    c.to_csv(OUT / "contrastes.csv", index=False)
    np.save(OUT / "P_residuo.npy", P.astype(np.float32))
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
