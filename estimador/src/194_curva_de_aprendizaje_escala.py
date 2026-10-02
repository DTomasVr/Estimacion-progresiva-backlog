"""E81 - ¿Sirve de algo tener MAS proyectos? La curva de aprendizaje del modelo de escala.

Antes de generar datos sinteticos hay que contestar si el problema es de cantidad.
El modelo que predice la escala del proyecto a partir del texto se entrena con 114
grupos. Si su rendimiento SIGUE SUBIENDO al pasar de 20 a 114, entonces faltan
proyectos y generarlos (o conseguirlos) tiene sentido. Si ya se aplano, ningun
volumen adicional lo va a mover y el techo es otra cosa.

Es la pregunta barata que decide si la cara merece la pena.

Se entrena el mismo modelo de escala de E66 (vector medio del proyecto + agregados +
la media de las predicciones por tarea) con subconjuntos de m proyectos, y se evalua
siempre sobre el proyecto dejado fuera. Varias semillas por tamano.

Se mira el ORDEN (Spearman) y la DISPERSION de lo predicho, que es la que esta
comprimida: 0,27 contra 0,79 real.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/curva_escala")
S = "=" * 96
MIN_TAREAS, COMP, SEMILLAS = 8, 24, 5
TAMANOS = (20, 40, 60, 80, 100, 113)
MANTEN = ("typo", "readme", "doc", "docs", "update", "upgrade", "bump", "rename",
          "cleanup", "clean up", "remove", "log", "comment")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    assert len(pred) == len(sw)
    tam = sw.groupby("project").size()
    k = sw.project.isin(tam[tam >= MIN_TAREAS].index).to_numpy()
    sw, V, pred = sw[k].reset_index(drop=True), V[k], pred[k]
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    tit = sw.title.fillna("").astype(str)

    aux = pd.DataFrame({"p": sw.project.to_numpy(), "y": sw.effort_hours.to_numpy(),
                        "e": pred,
                        "nw": tit.str.split().str.len().to_numpy(),
                        "nc": tit.str.len().to_numpy(),
                        "desc": sw.description.fillna("").str.len().gt(0).to_numpy(),
                        "mant": tit.str.lower().apply(
                            lambda s: any(w in s for w in MANTEN)).to_numpy()})
    nom = list(pd.unique(aux.p))
    g = aux.groupby("p")
    real = np.log(g.y.mean().reindex(nom).to_numpy())          # la escala a predecir
    A = np.log(g.e.mean().reindex(nom).to_numpy())             # lo que hay hoy
    Mu = pd.DataFrame(Vn).groupby(aux.p.to_numpy()).mean().reindex(nom).to_numpy()
    Mu = Mu / np.maximum(np.linalg.norm(Mu, axis=1, keepdims=True), 1e-9)
    agr = np.column_stack([
        np.log(g.size().reindex(nom).to_numpy()),
        g.nw.mean().reindex(nom).to_numpy(), g.nw.std().reindex(nom).to_numpy(),
        g.nc.mean().reindex(nom).to_numpy(),
        g.desc.mean().reindex(nom).to_numpy(), g.mant.mean().reindex(nom).to_numpy(),
    ])
    N = len(nom)
    print(f"{N} proyectos · {len(sw):,} tareas · sd real de la escala "
          f"{real.std():.3f} · {time.time() - t0:.0f} s", flush=True)

    filas = []
    for m in TAMANOS:
        for sem in range(SEMILLAS):
            rng = np.random.default_rng(100 * m + sem)
            p_hat = np.empty(N)
            for i in range(N):
                otros = np.delete(np.arange(N), i)
                tr = rng.choice(otros, min(m, len(otros)), replace=False)
                sv = TruncatedSVD(min(COMP, len(tr) - 1), random_state=0).fit(Mu[tr])
                Xa = np.hstack([sv.transform(Mu[tr]), agr[tr], A[tr][:, None]])
                Xb = np.hstack([sv.transform(Mu[[i]]), agr[[i]], A[[i]][:, None]])
                mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
                mo = RidgeCV(alphas=np.logspace(-1, 5, 30)).fit((Xa - mu) / sd, real[tr])
                p_hat[i] = mo.predict((Xb - mu) / sd)[0]
            # recentrar, como en E66: el nivel global se corrige con un solo numero
            p_hat = p_hat - np.median(p_hat) + np.median(real)
            filas.append(dict(
                m=m, semilla=sem,
                spearman=float(pd.Series(p_hat).corr(pd.Series(real), method="spearman")),
                sd=float(p_hat.std()),
                err_log=float(np.median(np.abs(p_hat - real)))))
        print(f"  m={m:3d} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    r = t.groupby("m").agg(spearman=("spearman", "mean"), sd_sp=("spearman", "std"),
                           sd=("sd", "mean"), err_log=("err_log", "mean")).reset_index()
    base_sp = float(pd.Series(A).corr(pd.Series(real), method="spearman"))
    print(f"\n{S}\n¿SIGUE MEJORANDO AL ANADIR PROYECTOS?\n{S}")
    print(f"{'proyectos':>10}{'Spearman':>11}{'(sd)':>8}{'disp. predicha':>16}"
          f"{'err log':>10}")
    for _, x in r.iterrows():
        print(f"{int(x.m):>10}{x.spearman:>11.4f}{x.sd_sp:>8.4f}{x.sd:>16.3f}"
              f"{x.err_log:>10.4f}")
    print(f"\n  referencia · la media de las predicciones por tarea: Spearman "
          f"{base_sp:.4f}")
    print(f"  referencia · dispersion REAL de la escala: {real.std():.3f}")

    print(f"\n{S}\nLECTURA\n{S}")
    ult, pen = r.iloc[-1], r.iloc[-3]
    pend = (ult.spearman - pen.spearman) / (ult.m - pen.m) * 100
    ruido = float(r.sd_sp.mean())
    print(f"  de {int(pen.m)} a {int(ult.m)} proyectos el Spearman sube "
          f"{ult.spearman - pen.spearman:+.4f} ({pend:+.4f} por cada 100 proyectos)")
    print(f"  el ruido entre semillas es de {ruido:.4f}")
    if ult.spearman - pen.spearman > 2 * ruido:
        print("  -> el ORDEN sigue subiendo: mas proyectos ayudarian a ordenar")
    else:
        print("  -> el ORDEN se esta aplanando: el margen que queda es pequeno")
    # Y lo que de verdad importa aqui no es el orden sino la DISPERSION, que es la
    # que produce la nube horizontal. Si no mejora con el tamano, mas proyectos no
    # arreglan la nube por muchos que sean.
    print()
    r2 = r.sort_values("m")
    pend_sd = float(np.polyfit(r2.m, r2.sd, 1)[0]) * 100
    print(f"  la DISPERSION predicha va de {r2.sd.iloc[0]:.3f} (m={int(r2.m.iloc[0])}) "
          f"a {r2.sd.iloc[-1]:.3f} (m={int(r2.m.iloc[-1])}),")
    print(f"  frente a {real.std():.3f} real: {pend_sd:+.3f} por cada 100 proyectos")
    if pend_sd > .05:
        print("  -> la compresion MEJORA con el tamano: generar proyectos atacaria la nube")
    else:
        print("  -> la compresion NO mejora con el tamano: la nube no se arregla con")
        print("     mas proyectos, porque no es falta de datos sino falta de senal")
    t.to_csv(OUT / "crudo.csv", index=False)
    r.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"proyectos": N, "semillas": SEMILLAS, "tamanos": list(TAMANOS),
               "spearman_base": base_sp, "sd_real": float(real.std())},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
