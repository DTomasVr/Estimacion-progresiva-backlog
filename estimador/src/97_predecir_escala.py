"""E3 · Predecir la ESCALA del proyecto, y solo despues la tarea.

E2 dejo dos hechos:

  el oraculo que conoce la mediana real del proyecto vale ΔR² +0,142
  el nombre del proyecto, una sola palabra, ya captura ΔR² +0,055 de eso
  promediar los vectores de los otros titulos del backlog NO vale (ΔR² -0,017
  contra su propio control permutado)

Si el nombre funciona es porque hay texto de nivel PROYECTO que predice la
escala. Entonces conviene atacarla de frente, en dos etapas:

  1  un modelo con una fila POR PROYECTO que predice log(mediana de horas) a
     partir de texto de proyecto. Son 158-204 ejemplos, no 25.000, asi que el
     modelo tiene que ser pequeño.
  2  el modelo de tarea predice la RAZON contra esa escala estimada.

La ventaja sobre meter el texto de proyecto como 768 columnas mas: la etapa 1
tiene el objetivo correcto y no se ahoga entre los rasgos de la tarea.

Textos de proyecto que se prueban, todos disponibles al crear el proyecto:
  nombre           el nombre, tal cual
  titulos          los primeros 20 titulos del backlog, concatenados
  nombre+titulos   ambos

Validacion cruzada por proyecto. El control de permutacion tambien esta aqui:
si barajar los textos entre proyectos no baja el R², no hay señal.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
DATA = "data/processed/agix_effort_v4.parquet"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
MODELO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OUT = Path("results/escala_proyecto")
SEP = "=" * 118


def main():
    banco = sys.argv[1] if len(sys.argv) > 1 else "todo"
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sel = {"jira": (sw.source == "JiraPublico") & (~sw.apache), "todo": ~sw.apache}[banco]
    sw = sw[sel].reset_index(drop=True)
    emb = np.load(f"{EMB_TIT}.npy")
    ids = np.load(f"{EMB_TIT}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fe"] = [pos[t] for t in sw.task_id]
    T = emb[sw.fe.to_numpy()].astype(np.float32)

    g = sw.groupby("project")
    proy = pd.DataFrame({
        "n": g.size(),
        "mediana": g.effort_hours.median(),
        "log_mediana": g.effort_hours.apply(lambda s: np.log(s).mean()),
        "nombre": [k.split(":")[-1].replace("-", " ") for k in sorted(sw.project.unique())],
    }).reset_index()
    proy["titulos"] = [". ".join(sw.title[sw.project == k].astype(str).head(20))[:900]
                       for k in proy.project]
    print(f"banco «{banco}»: {len(sw):,} tareas · {len(proy)} proyectos")
    print(f"  log-mediana por proyecto: sd {proy.log_mediana.std():.3f} "
          f"(medianas {proy.mediana.min():.2f}–{proy.mediana.max():.2f} h)")

    st = SentenceTransformer(MODELO, device="cuda" if torch.cuda.is_available() else "cpu")
    cod = lambda xs: st.encode(list(xs), normalize_embeddings=False,
                               show_progress_bar=False).astype(np.float32)
    Pn, Pt = cod(proy.nombre), cod(proy.titulos)
    # media de los vectores de titulo del proyecto (lo que fallo como rasgo de tarea,
    # aqui se prueba como rasgo de PROYECTO, que es su nivel natural)
    Pm = np.vstack([T[(sw.project == k).to_numpy()].mean(0) for k in proy.project])
    n_col = np.log1p(proy.n.to_numpy())[:, None]

    rng = np.random.default_rng(3)
    perm = rng.permutation(len(proy))
    vistas = {
        "nombre": np.hstack([Pn, n_col]),
        "20 títulos del backlog": np.hstack([Pt, n_col]),
        "media de los vectores": np.hstack([Pm, n_col]),
        "nombre + 20 títulos": np.hstack([Pn, Pt, n_col]),
        "solo el nº de tareas": n_col,
        "nombre PERMUTADO (control)": np.hstack([Pn[perm], n_col]),
        "20 títulos PERMUTADOS (control)": np.hstack([Pt[perm], n_col]),
    }

    yp = proy.log_mediana.to_numpy()
    filas, pred_escala = [], {}
    kf = KFold(5, shuffle=True, random_state=0)
    for nombre, X in vistas.items():
        p = np.zeros(len(proy))
        for tr, te in kf.split(X):
            m = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(X[tr], yp[tr])
            p[te] = m.predict(X[te])
        pred_escala[nombre] = p
        filas.append(dict(vista=nombre, R2=float(1 - ((yp - p) ** 2).mean() / yp.var()),
                          MAElog=float(np.mean(np.abs(yp - p))),
                          spearman=float(pd.Series(yp).corr(pd.Series(p), method="spearman"))))
        print(f"  escala desde «{nombre}»: R² {filas[-1]['R2']:+.4f} · "
              f"ρ {filas[-1]['spearman']:+.3f}")

    t = pd.DataFrame(filas)
    print(f"\n{SEP}\n1 · ¿SE PUEDE PREDECIR LA ESCALA DEL PROYECTO? (una fila por proyecto)\n{SEP}")
    print(t.round(4).to_string(index=False))

    # ---------------------------------------------- 2 · usarla en el modelo de tarea
    from sklearn.ensemble import HistGradientBoostingRegressor

    def hgb():
        return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                             l2_regularization=1., random_state=0)

    def metricas(y, p):
        p = np.clip(np.asarray(p, float), .01, None)
        m = E.point_metrics(y, p)
        ly = np.log(np.asarray(y, float))
        return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(np.log(p) - ly))),
                    PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                    R2=float(1 - ((ly - np.log(p)) ** 2).mean() / ly.var()),
                    spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))

    mejor = t.sort_values("R2", ascending=False).vista.iloc[0]
    mapa = {k: i for i, k in enumerate(proy.project)}
    esc_pred = pred_escala[mejor][[mapa[k] for k in sw.project]]
    esc_real = proy.log_mediana.to_numpy()[[mapa[k] for k in sw.project]]
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    gkf = GroupKFold(n_splits=5)
    grupos = sw.project.to_numpy()

    arms = {
        "A · solo el título (referencia)": (T, None),
        f"B · título + escala estimada («{mejor}»)": (np.hstack([T, esc_pred[:, None]]), None),
        "C · razón contra la escala estimada": (T, esc_pred),
        "D · razón contra la escala REAL (oráculo)": (T, esc_real),
    }
    filas2 = []
    for nombre, (X, desplaz) in arms.items():
        p = np.zeros(len(sw))
        obj = ly if desplaz is None else ly - desplaz
        for tr, te in gkf.split(X, ly, grupos):
            p[te] = hgb().fit(X[tr], obj[tr]).predict(X[te])
        if desplaz is not None:
            p = p + desplaz
        filas2.append(dict(brazo=nombre, n=len(sw), **metricas(y, np.exp(p))))
        print(f"  {nombre:46s} MdMRE {filas2[-1]['MdMRE']:.4f} · "
              f"R² {filas2[-1]['R2']:+.4f}", flush=True)
    t2 = pd.DataFrame(filas2)
    print(f"\n{SEP}\n2 · LA ESCALA ESTIMADA DENTRO DEL MODELO DE TAREA\n{SEP}")
    print(t2.round(4).to_string(index=False))
    b = t2.iloc[0]
    for _, r in t2.iloc[1:].iterrows():
        print(f"  {r.brazo:46s} ΔMdMRE {r.MdMRE - b.MdMRE:+.4f} · ΔR² {r.R2 - b.R2:+.4f}")

    t.to_csv(OUT / f"escala_{banco}.csv", index=False)
    t2.to_csv(OUT / f"escala_tarea_{banco}.csv", index=False)
    json.dump({"banco": banco, "mejor_vista": mejor, "n_proyectos": len(proy)},
              open(OUT / f"escala_{banco}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/escala_{banco}.csv y escala_tarea_{banco}.csv")


if __name__ == "__main__":
    main()
