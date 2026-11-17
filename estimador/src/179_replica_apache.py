"""E71 - Replicar el experimento entero en Apache: 186 proyectos REALES e independientes.

El problema es de potencia: con 114 proyectos, demostrar la mejora que da el
modelo de escala necesitaria unos 500 (la potencia con 114 es del 20 %).

Generar proyectos sinteticos NO lo resuelve. Un proyecto sintetico construido a
partir de estos 114 hereda exactamente las relaciones que el modelo ya aprendio,
asi que un contraste sobre el no mide nada del mundo real: mide las propiedades
del generador. La literatura lo dice sin rodeos -- una replica sintetica arrastra
las limitaciones del original del que se derivo -- y aqui el riesgo es peor que
inutil, porque daria p-valores buenos y falsos.

Lo que si aporta potencia es una replica INDEPENDIENTE, y la tenemos sin gastar
nada: los 186 proyectos de Apache con al menos 8 tareas, que se descartaron para
entrenar pero son proyectos reales con horas reales. Si el mismo pipeline, montado
de cero dentro de Apache, da el mismo patron, la evidencia conjunta vale mucho
mas que un p-valor sacado de datos inventados.

Se repite todo dentro de Apache, sin tocar nada de fuera:

  1 predicciones por tarea con validacion cruzada por proyecto entero
  2 el total del proyecto en frio
  3 el modelo de escala (E66) dejando fuera un proyecto cada vez
  4 los mismos contrastes pareados

Y al final se combinan las dos corridas independientes con el metodo de Fisher,
que es lo que corresponde cuando se juntan dos replicas de la misma hipotesis.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import binomtest, wilcoxon  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import PoissonRegressor, RidgeCV  # noqa: E402
from sklearn.model_selection import GroupKFold, LeaveOneOut  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/replica_apache")
S = "=" * 116
MIN_TAREAS, PLIEGUES, N_BOLSA, K_VEC, COMP = 8, 5, 3, 25, 24
MANTEN = ("typo", "readme", "doc", "docs", "update", "upgrade", "bump", "rename",
          "cleanup", "clean up", "remove", "log", "comment")


def poi(semilla=0, **kw):
    p = dict(max_iter=200, max_depth=6, learning_rate=.06, l2_regularization=1.,
             loss="poisson", random_state=semilla)
    p.update(kw)
    return HistGradientBoostingRegressor(**p)


def mt(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    q = np.log(pred / real)
    m = float(np.median(q))
    lr = np.log(real)
    return dict(MAE_h=float(np.abs(pred - real).mean()),
                MdMRE=float(np.median(r)),
                R2_log=float(1 - ((lr - np.log(pred)) ** 2).mean() / lr.var()),
                PRED25=float(np.mean(r <= .25)), PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = joblib.load("models/estimador_pbe_v3.joblib")
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    ap = todo.fuente_jira.eq("Apache").to_numpy()
    sw = todo[ap].reset_index(drop=True)
    Vall = V[ap]
    tam = sw.groupby("project").size()
    k = sw.project.isin(tam[tam >= MIN_TAREAS].index).to_numpy()
    sw, Vall = sw[k].reset_index(drop=True), Vall[k]
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    proy = sw.project.to_numpy()
    Vn = Vall / np.maximum(np.linalg.norm(Vall, axis=1, keepdims=True), 1e-9)
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    X = np.hstack([Vall, SV])
    Xs = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    print(f"APACHE · {len(sw):,} tareas · {sw.project.nunique()} proyectos · "
          f"{time.time() - t0:.0f} s", flush=True)

    # 1 - predicciones por tarea, validacion cruzada por proyecto entero
    pred = np.zeros(len(sw))
    for nf, (a, b) in enumerate(GroupKFold(PLIEGUES).split(X, y, proy)):
        rng = np.random.default_rng(nf)
        pr = np.unique(proy[a])
        pin = set(rng.choice(pr, max(2, int(.2 * len(pr))), replace=False))
        ain = a[np.array([p in pin for p in proy[a]])]
        afu = a[np.array([p not in pin for p in proy[a]])]
        bolsa = []
        for s in range(N_BOLSA):
            sub = rng.choice(afu, int(.85 * len(afu)), replace=False)
            bolsa.append(poi(s, max_features=.8).fit(X[sub], y[sub]))
        m_sv = poi().fit(SV[afu], y[afu])
        A2, yv = Vn[afu], y[afu]

        def vecino(sel):
            o = np.zeros(len(sel))
            for i in range(0, len(sel), 2048):
                s = Vn[sel[i:i + 2048]] @ A2.T
                j = np.argpartition(-s, K_VEC, axis=1)[:, :K_VEC]
                o[i:i + 2048] = np.median(yv[j], axis=1)
            return o
        an = {t: vecino(v) for t, v in (("fu", afu), ("in", ain), ("b", b))}
        m_an = poi().fit(an["fu"][:, None], y[afu])
        m_li = PoissonRegressor(alpha=1e-3, max_iter=300).fit(Xs[afu], y[afu])

        def cana(sel, a_):
            return np.column_stack([np.mean([m.predict(X[sel]) for m in bolsa], 0),
                                    m_sv.predict(SV[sel]), m_an.predict(a_[:, None]),
                                    m_li.predict(Xs[sel])])
        apl = poi(max_iter=120, max_depth=3).fit(cana(ain, an["in"]), y[ain])
        pred[b] = apl.predict(cana(b, an["b"]))
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)
    pred = np.clip(pred, 1e-6, None)
    np.save(OUT / "pred_tarea.npy", pred)

    # 2 - totales
    d = pd.DataFrame({"p": proy, "real": y, "est": pred,
                      "nw": tit.str.split().str.len().to_numpy(),
                      "nc": tit.str.len().to_numpy(),
                      "desc": sw.description.fillna("").str.len().gt(0).to_numpy(),
                      "mant": tit.str.lower().apply(
                          lambda s: any(w in s for w in MANTEN)).to_numpy()})
    nom = list(pd.unique(d.p))
    g = d.groupby("p")
    real_tot = g.real.sum().reindex(nom).to_numpy()
    frio = g.est.sum().reindex(nom).to_numpy()
    n_t = g.size().reindex(nom).to_numpy().astype(float)
    esc_r = np.log(g.real.mean().reindex(nom).to_numpy())
    A = np.log(frio / n_t)
    med = float(np.median(y))
    print(f"\n{S}\nTOTAL DEL PROYECTO EN APACHE ({len(nom)} proyectos)\n{S}")
    tt = pd.DataFrame([dict(brazo="constante", **mt(real_tot, n_t * med)),
                       dict(brazo="el modelo, en frio", **mt(real_tot, frio))])
    print(tt.round(4).to_string(index=False))

    # 3 - modelo de escala, dejando fuera un proyecto
    Mu = pd.DataFrame(Vn).groupby(proy).mean().reindex(nom).to_numpy()
    Mu = Mu / np.maximum(np.linalg.norm(Mu, axis=1, keepdims=True), 1e-9)
    agr = np.column_stack([np.log(n_t), g.nw.mean().reindex(nom).to_numpy(),
                           g.nw.std().reindex(nom).to_numpy(),
                           g.nc.mean().reindex(nom).to_numpy(),
                           g.desc.mean().reindex(nom).to_numpy(),
                           g.mant.mean().reindex(nom).to_numpy()])
    esc_m = np.zeros(len(nom))
    for a, b in LeaveOneOut().split(Mu):
        sv = TruncatedSVD(COMP, random_state=0).fit(Mu[a])
        Xa = np.hstack([sv.transform(Mu[a]), agr[a], A[a][:, None]])
        Xb = np.hstack([sv.transform(Mu[b]), agr[b], A[b][:, None]])
        mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
        m = RidgeCV(alphas=np.logspace(-1, 5, 40)).fit((Xa - mu) / sd, esc_r[a])
        esc_m[b] = m.predict((Xb - mu) / sd)
    print(f"\n{S}\nLA ESCALA EN APACHE\n{S}")
    esc = {}
    for etq, cl, e in (("la media de las predicciones", "A", A),
                       ("modelo de proyecto", "modelo", esc_m)):
        esc[f"sd_{cl}"] = float(e.std())
        print(f"  {etq:30s} sd {e.std():.3f} · err_log "
              f"{np.median(np.abs(e - esc_r)):.4f} · Spearman "
              f"{pd.Series(e).corr(pd.Series(esc_r), method='spearman'):.4f} · "
              f"pendiente {np.polyfit(e, esc_r, 1)[0]:.4f}")
    esc["sd_real"] = float(esc_r.std())
    print(f"  {'REAL':30s} sd {esc_r.std():.3f}")

    # 4 - el mismo contraste
    # LOS DOS BRAZOS SIN SESGO. La primera version solo le quitaba el sesgo al
    # brazo corregido, y en Apache el brazo en frio llega a +41 %: la mejora que
    # salia era en su mayor parte quitar ese sesgo, no ordenar mejor. Comparar un
    # estimador sesgado contra uno centrado no dice nada sobre cual ordena mejor.
    def sin_sesgo(p):
        q = np.log(p / real_tot)
        c = np.array([-np.median(np.delete(q, i)) for i in range(len(nom))])
        return p * np.exp(c)
    frio_c = sin_sesgo(frio)
    tot_m = sin_sesgo(n_t * np.exp(esc_m))
    ea, eb = np.abs(np.log(frio_c / real_tot)), np.abs(np.log(tot_m / real_tot))
    ha = np.abs(frio_c - real_tot) / real_tot <= .25
    hb = np.abs(tot_m - real_tot) / real_tot <= .25
    n01, n10 = int((hb & ~ha).sum()), int((~hb & ha).sum())
    pw = float(wilcoxon(eb, ea).pvalue)
    pm = float(binomtest(n01, n01 + n10, .5).pvalue) if n01 + n10 else 1.
    print(f"\n{S}\nCONTRASTE EN APACHE · el modelo de escala frente al frio\n{S}")
    print(f"  +-25 %: {ha.mean():.4f} -> {hb.mean():.4f} · McNemar p = {pm:.4g} "
          f"({n01}-{n10})")
    print(f"  error en log: {np.median(ea):.4f} -> {np.median(eb):.4f} · "
          f"Wilcoxon p = {pw:.4g} · mejora en {int((eb < ea).sum())} de {len(nom)}")

    # La combinacion de las dos replicas con el metodo de Fisher se hace en E72
    # (180_contraste_corpus.py), que es quien mide el corpus propio con este mismo
    # procedimiento. Aqui estaba escrita a mano con dos constantes que no salian de
    # ningun fichero y que resultaron ser ademas equivocadas: el Wilcoxon de los 114
    # es 0,0833 y no 0,1862. Calcular lo mismo en dos sitios es lo que lo permitio.
    print("\n  el Fisher de las dos replicas lo calcula src/180_contraste_corpus.py")

    tt.to_csv(OUT / "totales.csv", index=False)
    json.dump({"proyectos": len(nom), "tareas": int(len(sw)),
               "p_wilcoxon": pw, "p_mcnemar": pm,
               "pred25_frio": float(ha.mean()), "pred25_corregido": float(hb.mean()),
               "err_log_frio": float(np.median(ea)),
               "err_log_corregido": float(np.median(eb)),
               "mejora_en": int((eb < ea).sum()),
               "mcnemar_01": n01, "mcnemar_10": n10, **esc,
               "spearman_A": float(pd.Series(A).corr(pd.Series(esc_r), method="spearman")),
               "spearman_modelo": float(pd.Series(esc_m).corr(pd.Series(esc_r),
                                                              method="spearman"))},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/ · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
