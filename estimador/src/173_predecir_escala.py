"""E66 - Predecir la escala del proyecto directamente, en vez de corregir la comprimida.

E65 deja el techo claro: las tres formas de descomprimir la escala (mapeo de
cuantiles, calibracion lineal, cambio de etiqueta) mejoran algo, pero ninguna
puede superar la calidad del ORDEN, que es Spearman 0,44. Corregir el recorrido
de una prediccion no mejora su ordenacion; para eso hay que predecir mejor.

Aqui se prueba a estimar log(horas medias por tarea) del proyecto como un
problema propio, con rasgos de PROYECTO, en vez de sacarlo promediando
predicciones por tarea que ya vienen aplastadas.

  A  lo que hay              la media de las predicciones por tarea
  B  vector medio            el proyecto como la media de los vectores de sus
                             tareas, reducido, mas ridge
  C  agregados               numero de tareas, dispersion de lo predicho, longitud
                             de los titulos, fraccion con descripcion, fraccion de
                             titulos que parecen de mantenimiento
  D  B + C
  E  D + lo que hay como rasgo

Con 114 proyectos esto es una muestra minuscula, asi que: ridge con la penalizacion
elegida dentro de cada pliegue, validacion dejando fuera un proyecto cada vez, y
el brazo A como control. Si B, C o D no le ganan a A por bastante, no hay nada.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "8"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/predecir_escala")
S = "=" * 112
MIN_TAREAS = 8
MANTEN = ("typo", "readme", "doc", "docs", "update", "upgrade", "bump", "rename",
          "cleanup", "clean up", "remove", "log", "comment")


def mide(real, pred, etq):
    e = np.abs(pred - real)
    return dict(brazo=etq, err_log=float(np.median(e)),
                sd=float(pred.std()),
                spearman=float(pd.Series(pred).corr(pd.Series(real), method="spearman")),
                R2=float(1 - ((real - pred) ** 2).sum() / ((real - real.mean()) ** 2).sum()),
                pendiente=float(np.polyfit(pred, real, 1)[0]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    tit = sw.title.fillna("").astype(str)
    E5 = np.load(CACHE / "e5-large.npy")
    Vn = E5 / np.maximum(np.linalg.norm(E5, axis=1, keepdims=True), 1e-9)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": est,
                      "nw": tit.str.split().str.len().to_numpy(),
                      "nc": tit.str.len().to_numpy(),
                      "desc": sw.description.fillna("").str.len().gt(0).to_numpy(),
                      "mant": tit.str.lower().apply(
                          lambda s: any(w in s for w in MANTEN)).to_numpy()})
    tam = d.groupby("p").size()
    ok = tam[tam >= MIN_TAREAS].index
    d = d[d.p.isin(ok)].reset_index(drop=True)
    sel = sw.project.isin(ok).to_numpy()
    nom = list(pd.unique(d.p))
    g = d.groupby("p")
    real = np.log(g.real.mean().reindex(nom).to_numpy())
    A = np.log(g.est.mean().reindex(nom).to_numpy())
    Vp = pd.DataFrame(Vn[sel]).groupby(sw.project[sel].to_numpy()).mean().reindex(
        nom).to_numpy()
    agr = np.column_stack([
        np.log(g.size().reindex(nom).to_numpy()),
        g.est.apply(lambda q: float(np.std(np.log(q)))).reindex(nom).to_numpy(),
        g.nw.mean().reindex(nom).to_numpy(),
        g.nc.mean().reindex(nom).to_numpy(),
        g.desc.mean().reindex(nom).to_numpy(),
        g.mant.mean().reindex(nom).to_numpy(),
        g.est.apply(lambda q: float((q < 2).mean())).reindex(nom).to_numpy(),
    ])
    print(f"{len(nom)} proyectos · vector de proyecto {Vp.shape} · "
          f"{agr.shape[1]} agregados · {time.time() - t0:.0f} s", flush=True)

    def loo(usar_vector, usar_agr, usar_A):
        """Ridge dejando fuera un proyecto, con TODO ajustado dentro del pliegue.

        Incluido el SVD del vector de proyecto: ajustarlo con los 114 proyectos y
        luego dejar uno fuera seria una fuga, pequena pero fuga, y con 114
        muestras conviene no regalar nada."""
        p = np.zeros(len(real))
        for a, b in LeaveOneOut().split(Vp):
            partes_a, partes_b = [], []
            if usar_vector:
                sv = TruncatedSVD(24, random_state=0).fit(Vp[a])
                partes_a.append(sv.transform(Vp[a]))
                partes_b.append(sv.transform(Vp[b]))
            if usar_agr:
                partes_a.append(agr[a])
                partes_b.append(agr[b])
            if usar_A:
                partes_a.append(A[a][:, None])
                partes_b.append(A[b][:, None])
            Xa = np.hstack(partes_a)
            Xb = np.hstack(partes_b)
            mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
            m = RidgeCV(alphas=np.logspace(-1, 5, 40)).fit((Xa - mu) / sd, real[a])
            p[b] = m.predict((Xb - mu) / sd)
        return p

    juegos = {"A · la media de las predicciones (lo que hay)": None,
              "B · vector medio del proyecto": (1, 0, 0),
              "C · agregados": (0, 1, 0),
              "D · B + C": (1, 1, 0),
              "E · D + lo que hay": (1, 1, 1)}
    filas = [mide(real, A, "A · la media de las predicciones (lo que hay)")]
    preds = {"A": A}
    for etq, X in juegos.items():
        if X is None:
            continue
        p = loo(*X)
        filas.append(mide(real, p, etq))
        preds[etq[0]] = p
        print(f"  {etq} listo · {time.time() - t0:.0f} s", flush=True)
    filas.append(dict(brazo="REAL (referencia)", err_log=0., sd=float(real.std()),
                      spearman=1., R2=1., pendiente=1.))
    t = pd.DataFrame(filas)
    print(f"\n{S}\nPREDECIR log(horas medias por tarea) DEL PROYECTO\n{S}")
    print(t.round(4).to_string(index=False))
    print("  R2 negativo = peor que decir siempre la media. err_log es el error")
    print("  absoluto mediano en logaritmos; spearman es la calidad del ORDEN")

    # si alguno mejora el orden, se lleva al total
    mejor = t[t.brazo.str.startswith(("B", "C", "D", "E"))].sort_values(
        "spearman", ascending=False)
    if len(mejor):
        b = mejor.iloc[0]
        print(f"\n  el mejor por orden es {b.brazo}: Spearman {b.spearman:.4f} "
              f"frente a {t.iloc[0].spearman:.4f} de lo que hay")
        if b.spearman > t.iloc[0].spearman + .03:
            print("  -> mejora el orden, merece la pena llevarlo al total")
        else:
            print("  -> NO mejora el orden lo suficiente; el techo de E65 sigue en pie")
    t.to_csv(OUT / "resumen.csv", index=False)
    np.save(OUT / "predicciones.npy",
            np.column_stack([preds[k] for k in sorted(preds)]))
    json.dump({"proyectos": len(nom), "componentes": 24},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()


def al_total():
    """Llevar la escala mejor predicha al total: total = n x escala.

    Un detalle que hay que cuidar: se regresa log(media por tarea) con error
    cuadratico, asi que exp(prediccion) es una MEDIANA y para multiplicar por n y
    obtener una suma hace falta la MEDIA. Se corrige con smearing de Duan,
    calculado dejando fuera el proyecto que se evalua. Es la quinta vez que
    aparece este mismo desajuste.
    """
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": est})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)]
    nom = list(pd.unique(d.p))
    g = d.groupby("p")
    real_tot = g.real.sum().reindex(nom).to_numpy()
    pred_tot = g.est.sum().reindex(nom).to_numpy()
    n_t = g.size().reindex(nom).to_numpy().astype(float)
    real = np.log(g.real.mean().reindex(nom).to_numpy())
    P = np.load(OUT / "predicciones.npy")          # columnas A B C D E, en orden
    cols = ["A", "B", "C", "D", "E"]

    def mt2(r, p):
        r = np.asarray(r, float)
        p = np.clip(np.asarray(p, float), 1e-6, None)
        rr = np.abs(p - r) / r
        q = np.log(p / r)
        m = float(np.median(q))
        lr = np.log(r)
        return dict(MAE_h=float(np.abs(p - r).mean()),
                    RMSE_h=float(np.sqrt(np.mean((p - r) ** 2))),
                    MdMRE=float(np.median(rr)), MMRE=float(rr.mean()),
                    R2_log=float(1 - ((lr - np.log(p)) ** 2).mean() / lr.var()),
                    PRED25=float(np.mean(rr <= .25)), PRED50=float(np.mean(rr <= .50)),
                    factor2=float(np.mean(np.maximum(p / r, r / p) <= 2)),
                    SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))

    filas = [dict(brazo="0 · sin tocar (el total de siempre)",
                  **mt2(real_tot, pred_tot))]
    for j, c in enumerate(cols):
        e = P[:, j]
        res = np.exp(real - e)                      # factor de smearing por proyecto
        sm = np.array([np.mean(np.delete(res, i)) for i in range(len(e))])
        for etq, f in ((f"1 · escala {c}", 1.0), (f"2 · escala {c} con smearing", sm)):
            filas.append(dict(brazo=etq, **mt2(real_tot, n_t * np.exp(e) * f)))
        # media geometrica con el total de siempre, por si se complementan
        filas.append(dict(brazo=f"3 · escala {c} mezclada con el total de siempre",
                          **mt2(real_tot, np.sqrt(pred_tot * n_t * np.exp(e) * sm))))
    t = pd.DataFrame(filas).sort_values("MdMRE")
    print(f"\n{S}\nTOTAL DEL PROYECTO CON LA ESCALA PREDICHA APARTE\n{S}")
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "MMRE", "R2_log", "PRED25",
             "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    t.to_csv(OUT / "al_total.csv", index=False)
    print(f"\nescrito {OUT}/al_total.csv")


if __name__ == "__main__":
    al_total()


def significancia():
    """Con 114 proyectos, cualquier mejora hay que contrastarla antes de creersela."""
    from scipy.stats import wilcoxon, binomtest
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": est})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)]
    nom = list(pd.unique(d.p))
    g = d.groupby("p")
    real = g.real.sum().reindex(nom).to_numpy()
    base = g.est.sum().reindex(nom).to_numpy()
    n = g.size().reindex(nom).to_numpy().astype(float)
    P = np.load(OUT / "predicciones.npy")
    nuevo = n * np.exp(P[:, 3])
    eb, en = np.abs(np.log(base / real)), np.abs(np.log(nuevo / real))
    hb = np.abs(base - real) / real <= .25
    hn = np.abs(nuevo - real) / real <= .25
    n01, n10 = int((hn & ~hb).sum()), int((~hn & hb).sum())
    rng = np.random.default_rng(0)
    dif = [float(np.median(en[s]) - np.median(eb[s]))
           for s in (rng.integers(0, len(real), len(real)) for _ in range(4000))]
    print(f"\n{S}\nCONTRASTE: LA MEJORA NO ES DISTINGUIBLE DEL RUIDO\n{S}")
    print(f"  error absoluto en log, mediana: base {np.median(eb):.4f} · "
          f"nuevo {np.median(en):.4f}")
    print(f"  Wilcoxon pareado p = {float(wilcoxon(en, eb).pvalue):.4g} · "
          f"mejora en {int((en < eb).sum())} de {len(real)} proyectos")
    print(f"  +-25 %: {int(hb.sum())} aciertos -> {int(hn.sum())} · "
          f"McNemar p = {binomtest(n01, n01 + n10, .5).pvalue:.4g} ({n01}-{n10})")
    print(f"  IC95 de la diferencia: "
          f"[{np.percentile(dif, 2.5):+.4f}, {np.percentile(dif, 97.5):+.4f}]")
    print("  el intervalo cruza el cero: con 114 proyectos no hay potencia para ver")
    print("  una mejora de este tamano, asi que NO se puede presentar como mejora")


if __name__ == "__main__":
    significancia()
