"""E1 · Darle al modelo la escala del proyecto sin esperar a tener historial.

El diagnostico: en un proyecto nuevo el modelo en frio explica el 2,6 % de la
varianza de log-horas y conocer la media del proyecto explica el 37,5 %. Todo lo
que el texto de la tarea aporta es marginal al lado de una sola cosa que el
modelo no tiene: a que ritmo trabaja ese proyecto.

Hay dos fuentes de escala disponibles EN EL MOMENTO DE CREAR el proyecto, antes
de que se cierre ninguna tarea, y ninguna se ha probado:

  DESCRIPCION   TAWOS trae descripcion para sus 26 proyectos (19 palabras de
                mediana). Es exactamente lo que Agix recibe como entrada.
  EL PROPIO     Agix genera el backlog entero de golpe. Cuando estima una tarea
  BACKLOG       tiene delante las otras 114. El conjunto de textos del proyecto
                es una descripcion implicita, mucho mas rica que una frase, y
                esta disponible sin coste. Se usa dejando fuera la tarea que se
                estima, para que no haya fuga.

Brazos, todos sobre TAWOS con validacion cruzada por PROYECTO ENTERO (5 grupos),
que es la unica condicion que responde la pregunta:

  A  solo el titulo                                   (lo de hoy)
  B  titulo + descripcion del proyecto
  C  titulo + descripcion PERMUTADA entre proyectos   (control)
  D  titulo + media de los otros titulos del proyecto (el backlog como contexto)
  E  titulo + descripcion + media de los otros titulos
  F  titulo + mediana REAL del proyecto               (oraculo, cota superior)

El control C es obligatorio: si B le gana a A pero C tambien, la ganancia no
viene de la descripcion sino de haber metido 768 columnas mas.
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
SEP = "=" * 120


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    from sentence_transformers import SentenceTransformer

    df = pd.read_parquet(DATA)
    sw = df[(df.is_software_task == 1) & (df.source == "TAWOS")].copy().reset_index(drop=True)
    emb = np.load(f"{EMB_TIT}.npy")
    ids = np.load(f"{EMB_TIT}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    T = emb[[pos[t] for t in sw.task_id]].astype(np.float32)

    proy = pd.read_parquet("data/interim/Project.parquet")
    desc = dict(zip(proy.Project_Key, proy.Description.fillna("")))
    faltan = sorted(set(sw.project) - set(desc))
    print(f"TAWOS: {len(sw):,} tareas · {sw.project.nunique()} proyectos · "
          f"sin descripción {len(faltan)}")

    st = SentenceTransformer(MODELO, device="cuda" if torch.cuda.is_available() else "cpu")
    claves = sorted(sw.project.unique())
    D = st.encode([desc.get(k, "") for k in claves], normalize_embeddings=False,
                  show_progress_bar=False).astype(np.float32)
    idx_p = {k: i for i, k in enumerate(claves)}
    Dt = D[[idx_p[k] for k in sw.project]]

    rng = np.random.default_rng(7)
    perm = rng.permutation(len(claves))
    Dperm = D[perm][[idx_p[k] for k in sw.project]]

    # media de los OTROS titulos del mismo proyecto (sin la tarea que se estima)
    suma = np.zeros((len(claves), T.shape[1]), np.float32)
    cuenta = np.zeros(len(claves), np.float32)
    pi = np.array([idx_p[k] for k in sw.project])
    np.add.at(suma, pi, T)
    np.add.at(cuenta, pi, 1)
    Ctx = (suma[pi] - T) / np.maximum(cuenta[pi] - 1, 1)[:, None]

    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    med_real = sw.groupby("project").effort_hours.transform("median").to_numpy()
    n_proy = cuenta[pi][:, None]

    brazos = {
        "A · solo el título": T,
        "B · título + descripción del proyecto": np.hstack([T, Dt]),
        "C · título + descripción PERMUTADA (control)": np.hstack([T, Dperm]),
        "D · título + media de los otros títulos": np.hstack([T, Ctx, n_proy]),
        "E · título + descripción + otros títulos": np.hstack([T, Dt, Ctx, n_proy]),
        "F · título + mediana REAL del proyecto (oráculo)":
            np.hstack([T, np.log(med_real)[:, None]]),
    }

    gkf = GroupKFold(n_splits=5)
    grupos = sw.project.to_numpy()
    filas, pred = [], {k: np.zeros(len(sw)) for k in brazos}
    for nombre, X in brazos.items():
        for tr, te in gkf.split(X, ly, grupos):
            m = hgb().fit(X[tr], ly[tr])
            pred[nombre][te] = m.predict(X[te])
        filas.append(dict(brazo=nombre, n=len(sw), **metricas(y, np.exp(pred[nombre]))))
        print(f"  {nombre:50s} MdMRE {filas[-1]['MdMRE']:.4f} · R² {filas[-1]['R2']:+.4f}")

    t = pd.DataFrame(filas)
    print(f"\n{SEP}\nVALIDACIÓN CRUZADA POR PROYECTO ENTERO · TAWOS · {len(sw):,} tareas, "
          f"{sw.project.nunique()} proyectos\n{SEP}")
    print(t.round(4).to_string(index=False))

    base = t[t.brazo.str.startswith("A")].iloc[0]
    print(f"\n{SEP}\nLECTURA\n{SEP}")
    for _, r in t.iterrows():
        if r.brazo.startswith("A"):
            continue
        print(f"  {r.brazo:50s} ΔMdMRE {r.MdMRE - base.MdMRE:+.4f} · "
              f"ΔR² {r.R2 - base.R2:+.4f}")
    b = t[t.brazo.str.startswith("B")].iloc[0]
    c = t[t.brazo.str.startswith("C")].iloc[0]
    print(f"\n  control de permutación: la descripción real da {b.MdMRE:.4f} y la permutada "
          f"{c.MdMRE:.4f}")
    print(f"  -> la señal atribuible a la descripción es {c.MdMRE - b.MdMRE:+.4f} de MdMRE")

    t.to_csv(OUT / "resumen.csv", index=False)
    np.save(OUT / "predicciones.npy", np.vstack([pred[k] for k in brazos]))
    json.dump({"brazos": list(brazos), "n": len(sw),
               "proyectos": int(sw.project.nunique())},
              open(OUT / "meta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
