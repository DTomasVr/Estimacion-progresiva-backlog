"""E42 - La analogia en proyectos nuevos, pero con potencia estadistica.

E41 mide la analogia en el bloque externo y sale mejor, pero sin significancia:
PRED25 0,1705 -> 0,1780 con McNemar p = 0,69 sobre 792 tareas. Ese es justo el
problema que ya aparecio en E18 y que E22 tuvo que corregir: 792 tareas no dan
para detectar una diferencia de siete milesimas, asi que el "no significativo"
no dice que no haya efecto, dice que no se puede ver.

La version con potencia es validacion cruzada por PROYECTO ENTERO sobre las
24.848 tareas: cada pliegue deja fuera proyectos completos, se entrena con el
resto y se predice sobre los que faltan. Asi cada tarea se evalua como si su
proyecto fuese nuevo, y la muestra pasa de 792 a 24.848.

Los vecinos de cada tarea se buscan SOLO entre los proyectos de entrenamiento del
pliegue, que es lo que corresponde: si se buscaran en todo el corpus, la analogia
podria copiarse de tareas del mismo proyecto que se supone que no se ha visto.

Se anade el intervalo de confianza por remuestreo, porque una diferencia de
medias sin su incertidumbre no se puede llevar a una tesis.
"""
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVR

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/analogia_cv")
S = "=" * 116
K_VEC, N_BOLSA, N_PLIEGUES, N_REMUESTRAS = 25, 7, 5, 2000


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


def ic_diferencia(a, b, grupos, semilla=0):
    """IC al 95 % de la diferencia de tasas, remuestreando PROYECTOS enteros.

    Remuestrear tareas sueltas subestima el error cuando las tareas del mismo
    proyecto se parecen entre si, que es exactamente el caso aqui."""
    rng = np.random.default_rng(semilla)
    u = np.unique(grupos)
    idx = {g: np.flatnonzero(grupos == g) for g in u}
    d = []
    for _ in range(N_REMUESTRAS):
        s = np.concatenate([idx[g] for g in rng.choice(u, len(u), replace=True)])
        d.append(a[s].mean() - b[s].mean())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


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
    proy = sw.project.to_numpy()
    E5 = np.load(CACHE / "e5-large.npy")
    Vn = (E5 / np.maximum(np.linalg.norm(E5, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    X = np.hstack([E5, SV])
    AJ = np.load(CACHE / "e5-ajustado.npy")
    Xs = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    print(f"{len(sw):,} tareas · {len(np.unique(proy)):,} proyectos · "
          f"{N_PLIEGUES} pliegues por proyecto entero · {time.time() - t0:.0f} s", flush=True)

    p_mod = np.zeros(len(sw))
    p_ana = np.zeros(len(sw))
    disp = np.zeros(len(sw))
    for nf, (a, b) in enumerate(GroupKFold(N_PLIEGUES).split(X, ly, proy)):
        rng = np.random.default_rng(nf)
        # la referencia tiene que ser el apilado ENTERO, no solo la bolsa: con una
        # referencia mas debil la ganancia de la analogia sale inflada. Los pesos
        # del apilado se ajustan en una particion interna por proyecto, para que
        # ninguna tarea de los proyectos de prueba toque nada.
        pr = np.unique(proy[a])
        pin = set(rng.choice(pr, max(2, int(.2 * len(pr))), replace=False))
        ain = a[np.array([p in pin for p in proy[a]])]
        afu = a[np.array([p not in pin for p in proy[a]])]
        bolsa = []
        for s in range(N_BOLSA):
            sub = rng.choice(afu, int(.85 * len(afu)), replace=False)
            bolsa.append(HistGradientBoostingRegressor(
                max_iter=300, max_depth=6, learning_rate=.06, l2_regularization=1.,
                random_state=s, max_features=.8).fit(X[sub], ly[sub]))
        lr = Ridge(alpha=2.).fit(Xs[afu], ly[afu])
        ls = LinearSVR(C=.3, max_iter=4000, random_state=0).fit(Xs[afu], ly[afu])
        # OJO: aqui NO entra el canal del codificador ajustado. Ese codificador se
        # ajusto sobre split_t3 == "train" de TODO el corpus, asi que ya vio las
        # etiquetas de las tareas de los proyectos que este pliegue deja fuera.
        # Incluirlo da R2 0,398 y Spearman 0,621 en proyectos "nuevos", mejor que
        # en la prueba fija, que es imposible: es fuga, no habilidad. En la prueba
        # fija no ocurre porque esas filas nunca estuvieron en el ajuste.

        def canales(sel):
            return np.column_stack([np.mean([m.predict(X[sel]) for m in bolsa], 0),
                                    lr.predict(Xs[sel]), ls.predict(Xs[sel])])
        ap = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(canales(ain), ly[ain])
        p_mod[b] = ap.predict(canales(b))
        c = pd.Series(y[a]).value_counts()
        rej = np.sort(c[c >= 60].index.to_numpy())
        s = Vn[b] @ Vn[a].T
        j = np.argpartition(-s, K_VEC, axis=1)[:, :K_VEC]
        v = ly[a][j]
        p_ana[b] = np.log(rej[np.abs(np.log(rej)[None, :] -
                                     np.median(v, 1)[:, None]).argmin(1)])
        disp[b] = v.std(1)
        print(f"  pliegue {nf + 1}/{N_PLIEGUES} · {len(b):,} tareas · "
              f"{time.time() - t0:.0f} s", flush=True)

    u = float(np.quantile(disp, .85))          # el mismo 85 % que eligio E41
    brazos = {"modelo (apilado, sin el canal con fuga)": np.exp(p_mod),
              "analogia k=25, moda en la rejilla": np.exp(p_ana),
              "analogia encaminada por acuerdo": np.where(disp <= u, np.exp(p_ana),
                                                          np.exp(p_mod))}
    filas = []
    for n, q in brazos.items():
        filas.append(dict(brazo=n, n=len(sw), **metricas(y, q)))
    t = pd.DataFrame(filas)
    print(f"\n{S}\nTODA LA MUESTRA EVALUADA COMO SI SU PROYECTO FUESE NUEVO\n{S}")
    print(t.round(4).to_string(index=False))

    print(f"\n{S}\nCONTRASTES CONTRA EL MODELO, PAREADOS\n{S}")
    ref25 = np.abs(brazos["modelo (apilado, sin el canal con fuga)"] - y) / y <= .25
    ref2 = np.maximum(brazos["modelo (apilado, sin el canal con fuga)"] / y, y / brazos["modelo (apilado, sin el canal con fuga)"]) <= 2
    for n, q in brazos.items():
        if n.startswith("modelo"):
            continue
        for etq, h, r in (("PRED25", np.abs(q - y) / y <= .25, ref25),
                          ("factor2", np.maximum(q / y, y / q) <= 2, ref2)):
            n01 = int(np.sum(h & ~r))
            n10 = int(np.sum(~h & r))
            pv = binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.
            lo, hi = ic_diferencia(h, r, proy)
            print(f"  {n:34s} {etq:8s} {h.mean() - r.mean():+.4f} "
                  f"[{lo:+.4f}, {hi:+.4f}] · McNemar p = {pv:.3g} ({n01 - n10:+d})")
    print("\n  el IC se calcula remuestreando PROYECTOS enteros, no tareas sueltas:")
    print("  con tareas sueltas saldria mucho mas estrecho de lo que corresponde")
    t.to_csv(OUT / "resumen.csv", index=False)
    np.save(OUT / "pred.npy", np.column_stack([p_mod, p_ana, disp]))
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
