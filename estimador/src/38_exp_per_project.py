"""Within-project models, which is what the literature actually reports.

Deep-SE, GPT2SP and the SBERT+GBT line all train and evaluate inside a single
project. Everything measured so far has been pooled across projects, and
`split_project` is the opposite extreme -- a project never seen in training.
Neither matches the published setting, nor the setting Agix reaches once a team
has its own history.

For each project with enough data: chronological split (train on the earlier
tasks, test on the later ones), which is both the honest evaluation and the way
the product actually works.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import features as F  # noqa: E402

pd.set_option("display.width", 250)
E.DATA = "data/processed/agix_effort_v2.parquet"
MIN_TASKS = 150
TEST_FRAC = 0.25


def main():
    emb, pos = F.load_embeddings_and_pos()
    df = pd.read_parquet(E.DATA)
    sw = df[df.is_software_task == 1].copy()
    # chronological order where dates exist, otherwise task id order
    sw["orden"] = sw.created.fillna(pd.Timestamp("2000-01-01"))
    sw = sw.sort_values(["project", "orden", "task_id"])

    counts = sw.project.value_counts()
    proyectos = counts[counts >= MIN_TASKS].index.tolist()
    print(f"proyectos con >= {MIN_TASKS} tareas: {len(proyectos)} de "
          f"{sw.project.nunique()}")

    filas = []
    for proj in proyectos:
        d = sw[sw.project == proj]
        n_te = int(len(d) * TEST_FRAC)
        tr, te = d.iloc[:-n_te], d.iloc[-n_te:]
        if len(tr) < 80 or len(te) < 30:
            continue
        Xtr = emb[[pos[t] for t in tr.task_id]]
        Xte = emb[[pos[t] for t in te.task_id]]
        y_tr = np.log(tr.effort_hours.to_numpy())
        y_te = te.effort_hours.to_numpy()

        base = float(tr.effort_hours.median())
        m_base = E.point_metrics(y_te, np.full(len(te), base))
        r = Ridge(alpha=1.0).fit(Xtr, y_tr)
        m_r = E.point_metrics(y_te, np.exp(r.predict(Xte)))
        h = HistGradientBoostingRegressor(max_iter=300, max_depth=5,
                                          learning_rate=0.06, random_state=0)
        h.fit(Xtr, y_tr)
        m_h = E.point_metrics(y_te, np.exp(h.predict(Xte)))

        filas.append(dict(
            proyecto=proj, fuente=d.source.iloc[0], n=len(d),
            n_train=len(tr), n_test=len(te),
            base_MdMRE=round(m_base["MdMRE"], 4),
            ridge_MdMRE=round(m_r["MdMRE"], 4),
            hgb_MdMRE=round(m_h["MdMRE"], 4),
            ridge_PRED25=round(m_r["PRED25"], 4),
            mejora_pct=round((m_base["MdMRE"] - min(m_r["MdMRE"], m_h["MdMRE"]))
                             / m_base["MdMRE"] * 100, 1)))

    r = pd.DataFrame(filas).sort_values("n", ascending=False)
    print(f"\n{'='*100}\nmodelos dentro de cada proyecto, corte cronologico "
          f"({int(TEST_FRAC*100)}% final como prueba)\n{'='*100}")
    print(r.to_string(index=False))

    print(f"\n--- agregado sobre {len(r)} proyectos ---")
    for col in ["base_MdMRE", "ridge_MdMRE", "hgb_MdMRE", "ridge_PRED25"]:
        w = np.average(r[col], weights=r.n_test)
        print(f"  {col:16s} mediana {r[col].median():.4f}   "
              f"ponderado por tamano {w:.4f}")
    mejor = np.minimum(r.ridge_MdMRE, r.hgb_MdMRE)
    gana = int((mejor < r.base_MdMRE).sum())
    print(f"\n  proyectos donde el modelo supera a su linea base: {gana}/{len(r)}")
    print(f"  mejora mediana sobre la linea base: {r.mejora_pct.median():.1f}%")

    E.record("por_proyecto", "ridge", "cronologico",
             {"MdMRE": float(np.average(r.ridge_MdMRE, weights=r.n_test)),
              "PRED25": float(np.average(r.ridge_PRED25, weights=r.n_test)),
              "n_proyectos": len(r)})
    E.record("por_proyecto", "baseline", "cronologico",
             {"MdMRE": float(np.average(r.base_MdMRE, weights=r.n_test)),
              "n_proyectos": len(r)})
    r.to_csv("results/por_proyecto.csv", index=False)
    print("\nescrito results/por_proyecto.csv")


if __name__ == "__main__":
    main()
