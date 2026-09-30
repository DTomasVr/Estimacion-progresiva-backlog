"""E69 - Predecir la escala del proyecto entrenando TAMBIEN con Apache.

E38 tiro Apache para la prediccion por tarea y acerto: su mediana es 0,67 h
contra 2,0 h del resto, y mezclarlo desplazaba todo hacia abajo. Pero para
predecir la ESCALA DE UN PROYECTO esa misma diferencia deja de ser un problema y
pasa a ser lo que hace falta:

    resto   114 proyectos ·  24.622 tareas · escala log: sd 0,791 · 0,85 a 47,3 h
    Apache  186 proyectos ·  64.587 tareas · escala log: sd 1,024 · 0,17 a 41,3 h
    juntos  300 proyectos                  · sd 1,198

Triplica la muestra y ensancha el recorrido. Y el problema que tenemos es
justamente que el modelo no cubre recorrido: predice la escala con sd 0,271
frente a 0,791 real. Un modelo que solo ha visto proyectos de 0,85 a 47 h no
tiene de donde aprender que existen equipos que trabajan a 0,2 h por tarea;
Apache esta lleno de ellos.

  A  la media de las predicciones por tarea      lo que hay (Spearman 0,444)
  B  modelo de proyecto, solo sin Apache         E66 (0,481)
  C  modelo de proyecto, con Apache              la propuesta
  D  C mas la media de las predicciones como rasgo

Se evalua SIEMPRE sobre los mismos 114 proyectos sin Apache, dejando fuera el que
se mide, con Apache siempre en el entrenamiento. El SVD se ajusta dentro del
pliegue. No se usa ningun rasgo que diga si el proyecto es de Apache, porque en
un proyecto nuevo de Agix eso no se sabe.
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
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/escala_con_apache")
S = "=" * 114
MIN_TAREAS, COMP = 8, 24
MANTEN = ("typo", "readme", "doc", "docs", "update", "upgrade", "bump", "rename",
          "cleanup", "clean up", "remove", "log", "comment")


def mide(real, pred, etq):
    return dict(brazo=etq, err_log=float(np.median(np.abs(pred - real))),
                sd=float(pred.std()),
                spearman=float(pd.Series(pred).corr(pd.Series(real), method="spearman")),
                R2=float(1 - ((real - pred) ** 2).sum() / ((real - real.mean()) ** 2).sum()),
                pendiente=float(np.polyfit(pred, real, 1)[0]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f = joblib.load("models/estimador_agix_v3.joblib")
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy().reset_index(drop=True)
    sw["ap"] = sw.fuente_jira.eq("Apache")
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    V = np.load(CACHE / "e5-large-todo.npy")
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    tam = sw.groupby("project").size()
    ok = tam[tam >= MIN_TAREAS].index
    k = sw.project.isin(ok).to_numpy()
    sw, tit, y, Vn = sw[k].reset_index(drop=True), tit[k].reset_index(drop=True), y[k], Vn[k]
    print(f"{len(sw):,} tareas · {sw.project.nunique()} proyectos · "
          f"{time.time() - t0:.0f} s", flush=True)

    aux = pd.DataFrame({"p": sw.project.to_numpy(), "y": y, "ap": sw.ap.to_numpy(),
                        "nw": tit.str.split().str.len().to_numpy(),
                        "nc": tit.str.len().to_numpy(),
                        "desc": sw.description.fillna("").str.len().gt(0).to_numpy(),
                        "mant": tit.str.lower().apply(
                            lambda s: any(w in s for w in MANTEN)).to_numpy()})
    nom = list(pd.unique(aux.p))
    g = aux.groupby("p")
    real = np.log(g.y.mean().reindex(nom).to_numpy())
    es_ap = g.ap.first().reindex(nom).to_numpy()
    Mu = pd.DataFrame(Vn).groupby(aux.p.to_numpy()).mean().reindex(nom).to_numpy()
    Sd = pd.DataFrame(Vn).groupby(aux.p.to_numpy()).std().reindex(nom).to_numpy()
    Mu = Mu / np.maximum(np.linalg.norm(Mu, axis=1, keepdims=True), 1e-9)
    agr = np.column_stack([
        np.log(g.size().reindex(nom).to_numpy()),
        g.nw.mean().reindex(nom).to_numpy(), g.nw.std().reindex(nom).to_numpy(),
        g.nc.mean().reindex(nom).to_numpy(),
        g.desc.mean().reindex(nom).to_numpy(), g.mant.mean().reindex(nom).to_numpy(),
    ])
    # La media de las predicciones por tarea, para TODOS los proyectos. En la
    # primera version solo estaba para los que no son de Apache, y como el brazo D
    # exigia ese rasgo acababa excluyendo a Apache del entrenamiento: D salia
    # identico a B y no probaba lo que decia probar.
    sw2 = df[df.is_software_task == 1].reset_index(drop=True)
    pt = np.clip(np.load("results/modelo_total/pred_todo.npy"), 1e-6, None)
    prev = pd.DataFrame({"p": sw2.project.to_numpy(), "e": pt}).groupby("p").e.mean()
    A = np.log(prev.reindex(nom).to_numpy())
    hay_A = ~np.isnan(A)
    # para los 114 evaluables se usa la prediccion FUERA DE MUESTRA, que es la
    # honesta; pred_todo.npy sale de un modelo ajustado con el entrenamiento y
    # para esos proyectos seria en muestra
    sw3 = df[df.is_software_task == 1]
    sw3 = sw3[~sw3.fuente_jira.eq("Apache")].reset_index(drop=True)
    po = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    fuera = pd.DataFrame({"p": sw3.project.to_numpy(), "e": po}).groupby("p").e.mean()
    hay_f = np.array([n in fuera.index for n in nom])
    A[hay_f] = np.log(fuera.reindex(np.array(nom)[hay_f]).to_numpy())
    ev = np.flatnonzero((~es_ap) & hay_A)          # los 114 que se evaluan
    print(f"  evaluables {len(ev)} · Apache en entrenamiento {int(es_ap.sum())} · "
          f"{time.time() - t0:.0f} s", flush=True)

    def loo(usar_apache, usar_A, con_sd=True):
        p = np.full(len(nom), np.nan)
        for i in ev:
            tr = np.ones(len(nom), bool)
            tr[i] = False
            if not usar_apache:
                tr &= ~es_ap
            if usar_A:
                tr &= hay_A
            sv = TruncatedSVD(COMP, random_state=0).fit(Mu[tr])
            piezas_tr = [sv.transform(Mu[tr]), agr[tr]]
            piezas_te = [sv.transform(Mu[[i]]), agr[[i]]]
            if con_sd:
                sv2 = TruncatedSVD(8, random_state=0).fit(Sd[tr])
                piezas_tr.append(sv2.transform(Sd[tr]))
                piezas_te.append(sv2.transform(Sd[[i]]))
            if usar_A:
                piezas_tr.append(A[tr][:, None])
                piezas_te.append(A[[i]][:, None])
            Xa, Xb = np.hstack(piezas_tr), np.hstack(piezas_te)
            mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
            m = RidgeCV(alphas=np.logspace(-1, 5, 40)).fit((Xa - mu) / sd, real[tr])
            p[i] = float(m.predict((Xb - mu) / sd)[0])
        return p

    r_ev = real[ev]
    filas = [mide(r_ev, A[ev], "A · la media de las predicciones (lo que hay)")]
    juegos = {"B · modelo de proyecto, SIN Apache": (0, 1),
              "C · modelo de proyecto, CON Apache": (1, 0),
              "D · C + la media de las predicciones": (1, 1)}
    preds = {"A": A[ev]}
    for etq, (ap, ua) in juegos.items():
        p = loo(ap, ua)
        # recentrar con el desplazamiento de los demas proyectos evaluables, que
        # es lo justo: entrenar con Apache desplaza la escala hacia abajo y eso
        # se corrige con un solo numero, sin mirar el proyecto que se mide
        c = np.array([np.median(r_ev[np.arange(len(ev)) != j] -
                                p[ev][np.arange(len(ev)) != j])
                      for j in range(len(ev))])
        filas.append(mide(r_ev, p[ev] + c, etq))
        preds[etq[0]] = p[ev] + c
        print(f"  {etq} listo · {time.time() - t0:.0f} s", flush=True)
    filas.append(dict(brazo="REAL (referencia)", err_log=0., sd=float(r_ev.std()),
                      spearman=1., R2=1., pendiente=1.))
    t = pd.DataFrame(filas)
    print(f"\n{S}\nESCALA DEL PROYECTO · evaluado en los 114 sin Apache\n{S}")
    print(t.round(4).to_string(index=False))
    b = t.iloc[1:-1].sort_values("spearman", ascending=False).iloc[0]
    print(f"\n  mejor por orden: {b.brazo} · Spearman {b.spearman:.4f} frente a "
          f"{t.iloc[0].spearman:.4f} de lo que hay")
    t.to_csv(OUT / "resumen.csv", index=False)
    np.save(OUT / "predicciones.npy", np.column_stack([preds[k] for k in sorted(preds)]))
    json.dump({"proyectos_entrenamiento": int(len(nom)), "evaluados": int(len(ev)),
               "componentes": COMP}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
