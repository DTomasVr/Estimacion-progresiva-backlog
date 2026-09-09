"""Evaluate v3 in the setting the literature reports, and against its baseline.

Deep-SE and GPT2SP are scored per project against a median-effort baseline. The
replication by Tawosi et al. found GPT2SP beats that baseline in 6 of 16
projects after correcting a bug in the original MAE computation. That is the
comparison this script reproduces, plus the feature blocks v3 adds.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

pd.set_option("display.width", 250)
DATA = "data/processed/agix_effort_v3.parquet"
NUEVAS = ["n_links", "n_links_estructurales", "en_sprint", "tiene_asignado",
          "Title_Changed_After_Estimation", "Story_Point_Changed_After_Estimation"]


def embeddings(ids):
    sys.path.insert(0, "src")
    enc = __import__("15_encode_sbert")
    emb = enc.load_embeddings(enc.DEFAULT)
    base = pd.read_parquet("data/processed/agix_effort_clean.parquet",
                           columns=["task_id"]).task_id.to_numpy()
    pos = {t: i for i, t in enumerate(base)}
    return emb, pos


def bloques(tr, te, emb, pos, usar_nuevas, usar_dev):
    def rows(d):
        return emb[[pos[t] for t in d.task_id]]
    Xtr, Xte = [rows(tr)], [rows(te)]
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cols = ["task_kind", "source"] + (["desarrollador"] if usar_dev else [])
    a = tr[cols].astype(str)
    Xtr.append(oh.fit_transform(a))
    Xte.append(oh.transform(te[cols].astype(str)))
    if usar_nuevas:
        sc = StandardScaler()
        Xtr.append(sc.fit_transform(np.log1p(tr[NUEVAS].to_numpy(float))))
        Xte.append(sc.transform(np.log1p(te[NUEVAS].to_numpy(float))))
    return np.hstack(Xtr), np.hstack(Xte)


def main():
    df = pd.read_parquet(DATA)
    sw = df[(df.is_software_task == 1) & df.split_temporal.notna()].copy()
    emb, pos = embeddings(sw.task_id)

    configs = [("solo texto", False, False),
               ("texto + nuevas", True, False),
               ("texto + nuevas + desarrollador", True, True)]
    resultados = {n: [] for n, _, _ in configs}
    base_all, n_test = [], []
    proyectos = sorted(sw[sw.split_temporal == "test"].project.unique())

    for proj in proyectos:
        d = sw[sw.project == proj]
        tr, te = d[d.split_temporal == "train"], d[d.split_temporal == "test"]
        if len(tr) < 60 or len(te) < 15:
            continue
        y_tr = np.log(tr.effort_hours.to_numpy())
        y_te = te.effort_hours.to_numpy()
        base = E.point_metrics(y_te, np.full(len(te),
                                             float(tr.effort_hours.median())))
        base_all.append(base)
        n_test.append(len(te))
        for nombre, nuevas, dev in configs:
            Xtr, Xte = bloques(tr, te, emb, pos, nuevas, dev)
            mejor = None
            for mk in (lambda: Ridge(alpha=1.0),
                       lambda: HistGradientBoostingRegressor(
                           max_iter=250, max_depth=5, learning_rate=0.06,
                           random_state=0)):
                m = E.point_metrics(y_te, np.exp(mk().fit(Xtr, y_tr).predict(Xte)))
                if mejor is None or m["MdMRE"] < mejor["MdMRE"]:
                    mejor = m
            resultados[nombre].append(mejor)

    n_test = np.array(n_test, float)
    print(f"proyectos evaluados: {len(n_test)}   tareas de prueba: "
          f"{int(n_test.sum()):,}\n")
    filas = []
    b_md = np.array([m["MdMRE"] for m in base_all])
    filas.append(dict(configuracion="linea base (mediana del proyecto)",
                      MdMRE_mediano=round(float(np.median(b_md)), 4),
                      MdMRE_ponderado=round(float(np.average(b_md, weights=n_test)), 4),
                      PRED25=round(float(np.average(
                          [m["PRED25"] for m in base_all], weights=n_test)), 4),
                      gana_a_la_base="—"))
    for nombre, _, _ in configs:
        md = np.array([m["MdMRE"] for m in resultados[nombre]])
        p25 = np.array([m["PRED25"] for m in resultados[nombre]])
        filas.append(dict(
            configuracion=nombre,
            MdMRE_mediano=round(float(np.median(md)), 4),
            MdMRE_ponderado=round(float(np.average(md, weights=n_test)), 4),
            PRED25=round(float(np.average(p25, weights=n_test)), 4),
            gana_a_la_base=f"{int((md < b_md).sum())}/{len(md)}"))
        E.record("v3_temporal", nombre.replace(" ", "_"), "split_temporal",
                 {"MdMRE": float(np.median(md)),
                  "PRED25": float(np.average(p25, weights=n_test)),
                  "gana_base": int((md < b_md).sum()),
                  "n_proyectos": len(md)})
    r = pd.DataFrame(filas)
    print(r.to_string(index=False))
    print("\n  Referencia: GPT2SP, tras la correccion del estudio de replicacion")
    print("  de Tawosi et al., supera a la linea base de la mediana en 6 de 16.")

    # the estimate-bearing subset, which is the product's steady state
    est = sw[sw.tiene_estimacion == 1]
    tr, te = est[est.split_temporal == "train"], est[est.split_temporal == "test"]
    if len(te) > 100:
        y = te.effort_hours.to_numpy()
        m_h = E.point_metrics(y, te.HoursEstimate.to_numpy())
        b = (pd.DataFrame({"d": tr.desarrollador,
                           "r": np.log(tr.effort_hours) - np.log(tr.HoursEstimate)})
             .groupby("d").r.agg(["median", "size"]))
        g = float(np.median(np.log(tr.effort_hours) - np.log(tr.HoursEstimate)))
        k = 500.0
        corr = (b["size"] * b["median"] + k * g) / (b["size"] + k)
        adj = te.desarrollador.map(corr).fillna(g).to_numpy()
        m_c = E.point_metrics(y, np.exp(np.log(te.HoursEstimate) + adj))
        print(f"\n--- subconjunto con estimacion humana ({len(te):,} de prueba) ---")
        print(f"  estimacion cruda        MdMRE {m_h['MdMRE']:.4f}  "
              f"PRED25 {m_h['PRED25']:.4f}")
        print(f"  calibrada por persona   MdMRE {m_c['MdMRE']:.4f}  "
              f"PRED25 {m_c['PRED25']:.4f}")
        E.record("v3_temporal", "humano_crudo", "split_temporal", m_h)
        E.record("v3_temporal", "humano_calibrado", "split_temporal", m_c)


if __name__ == "__main__":
    main()
