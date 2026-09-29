"""E2 · La escala del proyecto, esta vez sobre un banco que tiene varianza que explicar.

E1 salio negativo pero estaba mal planteado: TAWOS solo tiene un 5,9 % de
varianza ENTRE proyectos, asi que ni el oraculo que conoce la mediana real podia
ganar gran cosa (+0,037 de R²). No habia nada que encontrar.

Donde si hay:

  TAWOS                     26 proyectos ·  5,9 % entre proyectos
  SiP                       20 proyectos ·  9,3 %
  todo sin Apache          204 proyectos · 12,7 %   <- el corpus que desplegamos
  JiraPublico sin Apache   158 proyectos · 21,6 %
  externo no Apache         49 proyectos · 37,6 %

Fuentes de escala disponibles EN EL MOMENTO DE CREAR el proyecto:

  NOMBRE     el bloque publico solo trae el nombre del proyecto, una palabra.
             Flojo, pero es texto y se prueba.
  BACKLOG    Agix genera todas las tareas de golpe, asi que al estimar una tiene
             delante las demas. El conjunto de titulos del proyecto es una
             descripcion implicita. Se calcula dejando fuera la tarea estimada.
  BACKLOG-k  lo mismo pero con solo k tareas del proyecto, que es lo realista
             cuando el backlog es pequeño.

Validacion cruzada por PROYECTO ENTERO, 5 grupos. El control de permutacion
distingue la señal real de haber metido 768 columnas mas.

    python src/96_escala_banco_real.py [jira|todo|externo]
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
DATA = "data/processed/agix_effort_v4.parquet"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
MODELO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OUT = Path("results/escala_proyecto")
SEP = "=" * 124


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0)


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    lr = np.log(p / y)
    ly = np.log(y)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lr))), PRED25=m["PRED25"],
                MAE_h=m["MAE_h"], sesgo=float(np.exp(np.median(lr))),
                R2=float(1 - ((ly - np.log(p)) ** 2).mean() / ly.var()),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def contexto(T, pi, n_grupos, k=None, semilla=0):
    """Media de los OTROS titulos del mismo proyecto. k limita cuantos se usan."""
    if k is None:
        suma = np.zeros((n_grupos, T.shape[1]), np.float32)
        cuenta = np.zeros(n_grupos, np.float32)
        np.add.at(suma, pi, T)
        np.add.at(cuenta, pi, 1)
        return (suma[pi] - T) / np.maximum(cuenta[pi] - 1, 1)[:, None], cuenta[pi]
    rng = np.random.default_rng(semilla)
    C = np.zeros_like(T)
    cuenta = np.zeros(len(T), np.float32)
    for g in range(n_grupos):
        idx = np.flatnonzero(pi == g)
        for i in idx:
            otros = idx[idx != i]
            if len(otros) == 0:
                continue
            sel = rng.choice(otros, min(k, len(otros)), replace=False)
            C[i] = T[sel].mean(0)
            cuenta[i] = len(sel)
    return C, cuenta


def main():
    banco = sys.argv[1] if len(sys.argv) > 1 else "jira"
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sel = {"jira": (sw.source == "JiraPublico") & (~sw.apache),
           "todo": ~sw.apache,
           "externo": (sw.split_t3 == "externo") & (~sw.apache)}[banco]
    sw = sw[sel].reset_index(drop=True)
    emb = np.load(f"{EMB_TIT}.npy")
    ids = np.load(f"{EMB_TIT}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    T = emb[[pos[t] for t in sw.task_id]].astype(np.float32)

    claves = sorted(sw.project.unique())
    idx_p = {k: i for i, k in enumerate(claves)}
    pi = np.array([idx_p[k] for k in sw.project])
    ly = np.log(sw.effort_hours.to_numpy())
    entre = 1 - ((ly - sw.groupby("project").effort_hours.transform(
        lambda s: np.log(s).mean()).to_numpy()) ** 2).mean() / ly.var()
    print(f"banco «{banco}»: {len(sw):,} tareas · {len(claves)} proyectos · "
          f"varianza entre proyectos {entre:.1%}")

    st = SentenceTransformer(MODELO, device="cuda" if torch.cuda.is_available() else "cpu")
    nombres = [k.split(":")[-1].replace("-", " ") for k in claves]
    N = st.encode(nombres, normalize_embeddings=False, show_progress_bar=False).astype(np.float32)
    Nt = N[pi]
    rng = np.random.default_rng(7)
    Nperm = N[rng.permutation(len(claves))][pi]

    Ctx, cuenta = contexto(T, pi, len(claves))
    Ctx10, _ = contexto(T, pi, len(claves), k=10)
    Cperm = Ctx[rng.permutation(len(Ctx))]
    n_col = np.log1p(cuenta)[:, None]
    med_real = sw.groupby("project").effort_hours.transform("median").to_numpy()

    brazos = {
        "A · solo el título": T,
        "B · + nombre del proyecto": np.hstack([T, Nt]),
        "C · + nombre PERMUTADO (control)": np.hstack([T, Nperm]),
        "D · + el backlog entero (otros títulos)": np.hstack([T, Ctx, n_col]),
        "E · + backlog PERMUTADO (control)": np.hstack([T, Cperm, n_col]),
        "F · + solo 10 tareas del backlog": np.hstack([T, Ctx10]),
        "G · + nombre + backlog": np.hstack([T, Nt, Ctx, n_col]),
        "H · + mediana REAL del proyecto (oráculo)": np.hstack([T, np.log(med_real)[:, None]]),
    }

    gkf = GroupKFold(n_splits=5)
    y = sw.effort_hours.to_numpy()
    filas = {}
    for nombre, X in brazos.items():
        p = np.zeros(len(sw))
        for tr, te in gkf.split(X, ly, sw.project.to_numpy()):
            p[te] = hgb().fit(X[tr], ly[tr]).predict(X[te])
        filas[nombre] = dict(brazo=nombre, n=len(sw), **metricas(y, np.exp(p)))
        print(f"  {nombre:44s} MdMRE {filas[nombre]['MdMRE']:.4f} · "
              f"R² {filas[nombre]['R2']:+.4f}", flush=True)

    t = pd.DataFrame(filas.values())
    print(f"\n{SEP}\nBANCO «{banco}» · validación cruzada por proyecto entero\n{SEP}")
    print(t.round(4).to_string(index=False))
    base = t[t.brazo.str.startswith("A")].iloc[0]
    print(f"\n{SEP}\nFRENTE A SOLO EL TÍTULO\n{SEP}")
    for _, r in t.iterrows():
        if not r.brazo.startswith("A"):
            print(f"  {r.brazo:44s} ΔMdMRE {r.MdMRE - base.MdMRE:+.4f} · "
                  f"ΔR² {r.R2 - base.R2:+.4f} · Δρ {r.spearman - base.spearman:+.4f}")
    for real, ctrl, etq in (("B", "C", "nombre del proyecto"),
                            ("D", "E", "el backlog")):
        a = t[t.brazo.str.startswith(real)].iloc[0]
        b = t[t.brazo.str.startswith(ctrl)].iloc[0]
        print(f"\n  señal atribuible a {etq}: ΔMdMRE {a.MdMRE - b.MdMRE:+.4f} · "
              f"ΔR² {a.R2 - b.R2:+.4f}  (real {a.MdMRE:.4f} vs permutado {b.MdMRE:.4f})")

    t.to_csv(OUT / f"banco_{banco}.csv", index=False)
    json.dump({"banco": banco, "n": len(sw), "proyectos": len(claves),
               "varianza_entre": float(entre)},
              open(OUT / f"banco_{banco}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/banco_{banco}.csv")


if __name__ == "__main__":
    main()
