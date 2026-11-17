"""E-270 - «¿Cuanto tardaste la ultima vez?» como tareas cerradas, y el previo del equipo.

PARTE 1. Al arrancar se le ensenan al usuario m tareas del backlog y se le pregunta
cuanto tardo la ultima vez en algo asi. Lo que conteste se trata, sin mas, como si
esas m tareas ya estuvieran cerradas (el anclaje desplegado, lambda 12). La
respuesta se simula de tres maneras:

  propia          las horas reales de esa misma tarea (un usuario que acierta)
  vecina hermana  las horas reales de la tarea MAS PARECIDA, por texto, entre los
                  otros proyectos de su misma organizacion: «la ultima vez»
  vecina ajena    lo mismo, pero buscando en las demas organizaciones (control)

Como la tarea no esta cerrada de verdad, en el total cuenta lo que dijo el usuario
y la verdad contra la que se mide incluye sus horas reales. Se prueba tambien
preguntar por las m tareas que mas se parecen a algo ya hecho, en vez de al azar.

PARTE 2 (la idea de Dycom, Minku y Yao 2014, en su forma mas simple). El factor
del equipo no se aplica en frio: se usa como PREVIO, y las tareas cerradas del
proyecto nuevo lo corrigen:

    desplazamiento = (k * log-cociente propio + lambda * w * previo) / (k + lambda)

previo = media del desplazamiento de los hermanos, dejando fuera el proyecto.
Con w = 0 es lo desplegado. E-269 midio rho = 0,205, asi que se espera w ~ 0,2.

Parecido por texto: mpnet congelado sobre los titulos (no vio las horas).
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/la_ultima_vez")
SORTEOS, LAMBDA = 60, 12
MS = (1, 3, 5, 10, 20)
KS = (0, 3, 5, 10, 20)
WS = (0, .2, .5, 1.0)

s = importlib.util.spec_from_file_location("e267", "src/267_entrada_del_usuario.py")
e267 = importlib.util.module_from_spec(s)
s.loader.exec_module(e267)


def carga():
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    d = pd.DataFrame({"id": sw.task_id.to_numpy(), "p": sw.project.to_numpy(),
                      "real": sw.effort_hours.to_numpy(),
                      "est": np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= 8].index)].reset_index(drop=True)
    ids = np.load("data/processed/embeddings/v4_mpnet_titulos.ids.npy", allow_pickle=True)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    assert d.id.isin(pos.index).all(), "tareas sin embedding"
    E = np.load("data/processed/embeddings/v4_mpnet_titulos.npy", mmap_mode="r")[pos.loc[d.id].to_numpy()]
    E = np.asarray(E, dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return d, E


def met(t, real):
    r = np.abs(t - real) / real
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(t / real, real / t) <= 2)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d, E = carga()
    proy = d.p.to_numpy()
    org_t = np.array([e267.org_de(p) for p in proy])
    nom = list(pd.unique(proy))
    org = np.array([e267.org_de(p) for p in nom])
    cuenta = pd.Series(org).value_counts()
    her = [j for j in range(len(nom)) if cuenta[org[j]] >= 2]
    re_all, es_all = d.real.to_numpy(), d.est.to_numpy()
    idx = {j: np.flatnonzero(proy == nom[j]) for j in her}
    real = np.array([re_all[idx[j]].sum() for j in her])
    frio = np.array([es_all[idx[j]].sum() for j in her])
    H = len(her)
    print(f"{H} proyectos con hermano · embeddings {E.shape} · {time.time() - t0:.0f} s", flush=True)

    # vecina mas parecida de cada tarea, en los hermanos y en las otras organizaciones
    vec = {}
    for a, j in enumerate(her):
        i = idx[j]
        for clave, cand in (("hermana", np.flatnonzero((org_t == org[j]) & (proy != nom[j]))),
                            ("ajena", np.flatnonzero(org_t != org[j]))):
            S = E[i] @ E[cand].T
            mejor = S.argmax(1)
            vec[(j, clave)] = (re_all[cand[mejor]], S.max(1))
        if (a + 1) % 20 == 0 or a + 1 == H:
            print(f"  vecinas {a + 1}/{H} · {time.time() - t0:.0f} s", flush=True)
    cal = []
    for clave in ("hermana", "ajena"):
        h = np.concatenate([vec[(j, clave)][0] for j in her])
        r = np.concatenate([re_all[idx[j]] for j in her])
        m = np.concatenate([es_all[idx[j]] for j in her])
        sim = np.concatenate([vec[(j, clave)][1] for j in her])
        cal.append(dict(respuesta=f"vecina {clave}", parecido_mediano=float(np.median(sim)),
                        error_log_mediano=float(np.median(np.abs(np.log(h / r)))),
                        error_log_mediano_del_modelo=float(np.median(np.abs(np.log(m / r)))),
                        spearman_con_real=float(pd.Series(h).corr(pd.Series(r), method="spearman")),
                        spearman_del_modelo=float(pd.Series(m).corr(pd.Series(r), method="spearman"))))
    cal = pd.DataFrame(cal)
    cal.to_csv(OUT / "calidad_de_la_respuesta.csv", index=False)
    print("\n¿CUANTO SE PARECE «LA ULTIMA VEZ» A LAS HORAS REALES DE LA TAREA? (por tarea)")
    print(cal.round(3).to_string(index=False), flush=True)

    def total(j, sel, horas):
        i = idx[j]
        k = len(sel)
        resto = np.ones(len(i), bool)
        resto[sel] = False
        f = np.exp(np.log(horas.sum() / es_all[i][sel].sum()) * k / (k + LAMBDA))
        return horas.sum() + f * es_all[i][resto].sum()

    filas = []
    for m in MS:
        rng = np.random.default_rng(27000 + m)
        acc = {}
        for _ in range(SORTEOS):
            tot = {}
            for a, j in enumerate(her):
                i = idx[j]
                k = min(m, len(i))
                sel = rng.choice(len(i), k, replace=False)
                top = np.argsort(-vec[(j, "hermana")][1])[:k]
                for nombre, ss, horas in (
                        ("propia (acierta)", sel, re_all[i][sel]),
                        ("vecina hermana · tareas al azar", sel, vec[(j, "hermana")][0][sel]),
                        ("vecina hermana · las mas parecidas", top, vec[(j, "hermana")][0][top]),
                        ("vecina ajena (control)", sel, vec[(j, "ajena")][0][sel])):
                    tot.setdefault(nombre, np.zeros(H))[a] = total(j, ss, horas)
            for nombre, t in tot.items():
                acc.setdefault(nombre, []).append(met(t, real))
        for nombre, v in acc.items():
            filas.append(dict(tareas_preguntadas=m, respuesta=nombre,
                              **{c: float(np.mean([x[c] for x in v])) for c in v[0]}))
        print(f"  parte 1 · {m} tareas listo · {time.time() - t0:.0f} s", flush=True)
    t1 = pd.DataFrame(filas)
    t1.to_csv(OUT / "como_cerradas.csv", index=False)
    f0 = met(frio, real)
    print(f"\n{'=' * 100}\nPARTE 1 · LAS RESPUESTAS COMO TAREAS CERRADAS ({H} proyectos · en frio PRED25 "
          f"{f0['PRED25']:.3f}, MdMRE {f0['MdMRE']:.3f})\n{'=' * 100}")
    for c in ("PRED25", "MdMRE", "factor2"):
        print(f"\n{c}:")
        print(t1.pivot(index="respuesta", columns="tareas_preguntadas", values=c).round(3).to_string())

    # parte 2 · el previo del equipo corregido por las cerradas propias
    b = np.log(real / frio)
    org_h = org[her]
    previo = np.array([b[(org_h == org_h[a]) & (np.arange(H) != a)].mean() for a in range(H)])
    filas = []
    for k in KS:
        rng = np.random.default_rng(27100 + k)
        acc = {w: [] for w in WS}
        for _ in range(SORTEOS if k else 1):
            tot = {w: np.zeros(H) for w in WS}
            for a, j in enumerate(her):
                i = idx[j]
                kk = min(k, len(i) - 1)
                sel = rng.choice(len(i), kk, replace=False)
                resto = np.ones(len(i), bool)
                resto[sel] = False
                propio = np.log(re_all[i][sel].sum() / es_all[i][sel].sum()) if kk else 0.
                for w in WS:
                    off = (kk * propio + LAMBDA * w * previo[a]) / (kk + LAMBDA)
                    tot[w][a] = re_all[i][sel].sum() + np.exp(off) * es_all[i][resto].sum()
            for w in WS:
                acc[w].append(met(tot[w], real))
        for w in WS:
            filas.append(dict(cerradas=k, w=w, **{c: float(np.mean([x[c] for x in acc[w]])) for c in acc[w][0]}))
        print(f"  parte 2 · k={k} listo · {time.time() - t0:.0f} s", flush=True)
    t2 = pd.DataFrame(filas)
    t2.to_csv(OUT / "previo_del_equipo.csv", index=False)
    print(f"\n{'=' * 100}\nPARTE 2 · PREVIO DEL EQUIPO CORREGIDO POR LAS CERRADAS PROPIAS ({H} proyectos)\n{'=' * 100}")
    for c in ("PRED25", "MdMRE", "factor2"):
        print(f"\n{c} (filas: peso w del previo, 0 = lo desplegado · columnas: tareas cerradas):")
        print(t2.pivot(index="w", columns="cerradas", values=c).round(3).to_string())
    json.dump({"proyectos": H, "sorteos": SORTEOS, "segundos": round(time.time() - t0)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
