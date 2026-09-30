"""E72 - El contraste del modelo de escala sobre los 114, con el MISMO procedimiento
que la replica de Apache.

Por que hace falta este guion: E71 combina las dos replicas con el metodo de Fisher,
y para eso los dos p-valores tienen que salir del mismo procedimiento. Pero el del
corpus propio estaba escrito a mano dentro de E71 (0,1862 y 0,2632) como "medido
antes", y no quedaba guardado en ningun CSV. Un numero que va a una lamina y no se
puede volver a producir no sirve, y ademas no habia forma de comprobar que se
midiera igual que el de Apache. Asi que se recalcula aqui y E71 lo lee de fichero.

Se copia exactamente lo que hace E71 en sus pasos 3 y 4:

  3  modelo de escala con el SVD dentro del pliegue, dejando fuera un proyecto
  4  el contraste pareado con LOS DOS BRAZOS SIN SESGO

Lo unico que cambia es el corpus y de donde salen las predicciones por tarea: aqui
se reusan las de results/modelo_total/pred_total.npy, que son las de fuera de
pliegue del mismo modelo que produce la metrica publicada. Volver a entrenarlo
daria lo mismo y costaria varios minutos.

Que los dos brazos vayan sin sesgo no es un detalle: la primera version de E71 solo
le quitaba el sesgo al corregido y daba un falso significativo.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest, chi2, wilcoxon  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import LeaveOneOut  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/contraste_corpus")
S = "=" * 116
MIN_TAREAS, COMP = 8, 24
MANTEN = ("typo", "readme", "doc", "docs", "update", "upgrade", "bump", "rename",
          "cleanup", "clean up", "remove", "log", "comment")


def mt(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    m = float(np.median(np.log(pred / real)))
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
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw = todo[noap].reset_index(drop=True)
    Vall = V[noap]
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    assert len(pred) == len(sw), f"pred_total {len(pred)} no cuadra con {len(sw)} tareas"

    tam = sw.groupby("project").size()
    k = sw.project.isin(tam[tam >= MIN_TAREAS].index).to_numpy()
    sw, Vall, pred = sw[k].reset_index(drop=True), Vall[k], pred[k]
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy()
    proy = sw.project.to_numpy()
    Vn = Vall / np.maximum(np.linalg.norm(Vall, axis=1, keepdims=True), 1e-9)
    print(f"CORPUS PROPIO · {len(sw):,} tareas · {sw.project.nunique()} proyectos · "
          f"{time.time() - t0:.0f} s", flush=True)

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
    print(f"\n{S}\nTOTAL DEL PROYECTO EN EL CORPUS PROPIO ({len(nom)} proyectos)\n{S}")
    tt = pd.DataFrame([dict(brazo="el modelo, en frio", **mt(real_tot, frio))])
    print(tt.round(4).to_string(index=False))

    # 3 - modelo de escala, dejando fuera un proyecto, con el SVD dentro
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
    esc = {}
    print(f"\n{S}\nLA ESCALA EN EL CORPUS PROPIO\n{S}")
    for etq, cl, e in (("la media de las predicciones", "A", A),
                       ("modelo de proyecto", "modelo", esc_m)):
        esc[f"sd_{cl}"] = float(e.std())
        print(f"  {etq:30s} sd {e.std():.3f} · err_log "
              f"{np.median(np.abs(e - esc_r)):.4f} · Spearman "
              f"{pd.Series(e).corr(pd.Series(esc_r), method='spearman'):.4f} · "
              f"pendiente {np.polyfit(e, esc_r, 1)[0]:.4f}")
    esc["sd_real"] = float(esc_r.std())
    print(f"  {'REAL':30s} sd {esc_r.std():.3f}")

    # 4 - el contraste, LOS DOS BRAZOS SIN SESGO
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
    print(f"\n{S}\nCONTRASTE · el modelo de escala frente al frio, los dos centrados\n{S}")
    print(f"  +-25 %: {ha.mean():.4f} -> {hb.mean():.4f} · McNemar p = {pm:.4g} "
          f"({n01}-{n10})")
    print(f"  error en log: {np.median(ea):.4f} -> {np.median(eb):.4f} · "
          f"Wilcoxon p = {pw:.4g} · mejora en {int((eb < ea).sum())} de {len(nom)}")
    print("\n  " + ("no hay efecto" if min(pw, pm) >= .05 else
                    "OJO: sale significativo, revisar antes de darlo por bueno"))

    # 5 - combinar las dos replicas independientes con el metodo de Fisher.
    # Se hace aqui y en ningun otro sitio: antes estaba dentro de E71 con los p del
    # corpus propio escritos a mano, y eran equivocados.
    fisher = {}
    ap = Path("results/replica_apache/meta.json")
    if ap.exists():
        a = json.load(open(ap))
        print(f"\n{S}\nLAS DOS REPLICAS JUNTAS (Fisher)\n{S}")
        print(f"  corpus propio {len(nom)} proyectos · Apache {a['proyectos']} "
              f"proyectos · {len(nom) + a['proyectos']} en total")
        for etq, p1, p2 in (("wilcoxon", pw, a["p_wilcoxon"]),
                            ("mcnemar", pm, a["p_mcnemar"])):
            x = -2 * (np.log(max(p1, 1e-12)) + np.log(max(p2, 1e-12)))
            pc_ = float(chi2.sf(x, 4))
            fisher[f"fisher_{etq}"] = pc_
            fisher[f"apache_{etq}"] = float(p2)
            print(f"  {etq:9s} propio p = {p1:.4g} · Apache p = {p2:.4g} "
                  f"-> combinado p = {pc_:.4g}")
        print("\n  " + ("el efecto no aparece en ninguna de las dos, ni juntas"
                        if min(fisher["fisher_wilcoxon"],
                               fisher["fisher_mcnemar"]) >= .05 else
                        "OJO: el combinado sale significativo, revisarlo"))
    else:
        print(f"\n  falta {ap}, no se puede combinar · corre antes "
              f"src/179_replica_apache.py")

    tt.to_csv(OUT / "totales.csv", index=False)
    json.dump({"proyectos": len(nom), "tareas": int(len(sw)),
               "p_wilcoxon": pw, "p_mcnemar": pm,
               "pred25_frio": float(ha.mean()), "pred25_corregido": float(hb.mean()),
               "err_log_frio": float(np.median(ea)),
               "err_log_corregido": float(np.median(eb)),
               "mejora_en": int((eb < ea).sum()),
               "mcnemar_01": n01, "mcnemar_10": n10, **esc, **fisher,
               "spearman_A": float(pd.Series(A).corr(pd.Series(esc_r),
                                                     method="spearman")),
               "spearman_modelo": float(pd.Series(esc_m).corr(pd.Series(esc_r),
                                                              method="spearman"))},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/ · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
