"""Step 5: the final evaluation. The test block is read exactly once.

Everything here is fixed in advance by results/seleccion_val.json, produced by
43_select_on_val.py without ever opening the test block. Nothing is tuned in
this script.

Protocol per project: fit on 'train', calibrate the conformal quantile on
'val', predict 'test'. Two regimes are reported separately -- cold start, where
no human estimate exists, and steady state, where it does.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

pd.set_option("display.width", 250)
DATA = "data/processed/agix_effort_v3.parquet"
NUEVAS = ["n_links", "n_links_estructurales", "n_componentes", "en_sprint",
          "tiene_asignado", "Title_Changed_After_Estimation",
          "Story_Point_Changed_After_Estimation"]
CATS = ["task_kind", "source", "desarrollador"]


def features(tr, otros, emb, pos):
    def rows(d):
        return emb[[pos[t] for t in d.task_id]]
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                       min_frequency=5)
    sc = StandardScaler()
    A = [rows(tr), oh.fit_transform(tr[CATS].astype(str)),
         sc.fit_transform(np.log1p(tr[NUEVAS].to_numpy(float)))]
    out = [np.hstack(A)]
    for d in otros:
        out.append(np.hstack([rows(d), oh.transform(d[CATS].astype(str)),
                              sc.transform(np.log1p(d[NUEVAS].to_numpy(float)))]))
    return out


def main():
    sel = json.load(open("results/seleccion_val.json", encoding="utf8"))
    print("configuracion congelada en validacion:")
    print(json.dumps(sel, indent=2, ensure_ascii=False))
    assert sel["codificador"] == "congelado"
    alpha = sel["ridge_alpha"]

    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    emb = np.load("data/processed/embeddings/"
                  "paraphrase-multilingual-mpnet-base-v2.npy")
    base_ids = pd.read_parquet("data/processed/agix_effort_clean.parquet",
                               columns=["task_id"]).task_id.to_numpy()
    pos = {t: i for i, t in enumerate(base_ids)}

    print(f"\n{'='*92}\nLECTURA UNICA DEL BLOQUE DE PRUEBA\n{'='*92}")
    filas, guardados = [], []
    for proj in sorted(sw[sw.split_t3 == "test"].project.unique()):
        d = sw[sw.project == proj]
        tr, va, te = (d[d.split_t3 == s] for s in ("train", "val", "test"))
        if len(tr) < 50 or len(te) < 12 or len(va) < 8:
            continue
        Xtr, Xva, Xte = features(tr, [va, te], emb, pos)
        y_tr = np.log(tr.effort_hours.to_numpy())
        y_va, y_te = va.effort_hours.to_numpy(), te.effort_hours.to_numpy()

        mdl = Ridge(alpha=alpha).fit(Xtr, y_tr)
        lp_va, lp_te = mdl.predict(Xva), mdl.predict(Xte)
        p = np.exp(lp_te)
        m = E.point_metrics(y_te, p)
        b = E.point_metrics(y_te, np.full(len(te), float(tr.effort_hours.median())))

        q80 = float(np.quantile(np.abs(np.log(y_va) - lp_va), 0.80))
        lo, hi = p / np.exp(q80), p * np.exp(q80)
        cov = float(np.mean((y_te >= lo) & (y_te <= hi)))

        filas.append(dict(proyecto=proj, fuente=d.source.iloc[0], n_test=len(te),
                          base_MdMRE=round(b["MdMRE"], 4),
                          MdMRE=round(m["MdMRE"], 4),
                          PRED25=round(m["PRED25"], 4),
                          PRED50=round(m["PRED50"], 4),
                          cob80=round(cov, 3), factor80=round(float(np.exp(q80)), 2),
                          gana=m["MdMRE"] < b["MdMRE"]))
        guardados.append((y_te, p, lo, hi))

    r = pd.DataFrame(filas)
    w = r.n_test.to_numpy(float)
    print(r.to_string(index=False))

    print(f"\n--- ARRANQUE EN FRIO: sin estimacion humana ---")
    print(f"  proyectos {len(r)}   tareas de prueba {int(w.sum()):,}")
    print(f"  MdMRE   mediano {r.MdMRE.median():.4f}   "
          f"ponderado {np.average(r.MdMRE, weights=w):.4f}")
    print(f"  linea base      {r.base_MdMRE.median():.4f}   "
          f"ponderado {np.average(r.base_MdMRE, weights=w):.4f}")
    print(f"  PRED25          {np.average(r.PRED25, weights=w):.4f}   "
          f"PRED50 {np.average(r.PRED50, weights=w):.4f}")
    print(f"  supera a su linea base en {int(r.gana.sum())}/{len(r)} proyectos")
    y_all = np.concatenate([g[0] for g in guardados])
    lo_all = np.concatenate([g[2] for g in guardados])
    hi_all = np.concatenate([g[3] for g in guardados])
    print(f"  cobertura del intervalo al 80%: "
          f"{float(np.mean((y_all>=lo_all)&(y_all<=hi_all))):.4f}   "
          f"factor mediano {np.average(r.factor80, weights=w):.2f}x")

    E.record("final", "arranque_frio", "split_t3",
             {"MdMRE": float(np.average(r.MdMRE, weights=w)),
              "MdMRE_mediano": float(r.MdMRE.median()),
              "PRED25": float(np.average(r.PRED25, weights=w)),
              "cov80": float(np.mean((y_all >= lo_all) & (y_all <= hi_all))),
              "gana_base": int(r.gana.sum()), "n_proyectos": len(r)})

    # ---------------- risk flags ----------------
    print(f"\n--- MARCADO DE RIESGO ---")
    tr = sw[sw.split_t3 == "train"]
    te = sw[sw.split_t3 == "test"]
    Xtr, Xte = features(tr, [te], emb, pos)
    for thr in (4, 8, 16, 40):
        ytr = (tr.effort_hours > thr).astype(int).to_numpy()
        yte = (te.effort_hours > thr).astype(int).to_numpy()
        if ytr.sum() < 50 or yte.sum() < 20:
            continue
        clf = HistGradientBoostingClassifier(max_iter=250, max_depth=5,
                                             learning_rate=0.06, random_state=0)
        pr = clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, pr)
        print(f"  > {thr:>2} h   AUC {auc:.3f}   tasa base {yte.mean():.3f}")
        E.record("final", f"riesgo_{thr}h", "split_t3",
                 {"AUC": float(auc), "tasa_base": float(yte.mean())})

    # ---------------- steady state ----------------
    est = sw[sw.tiene_estimacion == 1]
    tr_e, va_e, te_e = (est[est.split_t3 == s] for s in ("train", "val", "test"))
    if len(te_e) > 100:
        y = te_e.effort_hours.to_numpy()
        m_h = E.point_metrics(y, te_e.HoursEstimate.to_numpy())
        g = float(np.median(np.log(tr_e.effort_hours) - np.log(tr_e.HoursEstimate)))
        b = (pd.DataFrame({"d": tr_e.desarrollador,
                           "r": np.log(tr_e.effort_hours)
                                - np.log(tr_e.HoursEstimate)})
             .groupby("d").r.agg(["median", "size"]))
        k = 500.0
        corr = (b["size"] * b["median"] + k * g) / (b["size"] + k)
        adj = te_e.desarrollador.map(corr).fillna(g).to_numpy()
        p_cal = np.exp(np.log(te_e.HoursEstimate) + adj)
        m_c = E.point_metrics(y, p_cal)
        res = np.abs(np.log(va_e.effort_hours)
                     - (np.log(va_e.HoursEstimate)
                        + va_e.desarrollador.map(corr).fillna(g)))
        q = float(np.quantile(res, 0.80))
        cov = float(np.mean((y >= p_cal / np.exp(q)) & (y <= p_cal * np.exp(q))))
        print(f"\n--- REGIMEN ESTABLE: con estimacion humana "
              f"({len(te_e):,} de prueba) ---")
        print(f"  estimacion cruda       MdMRE {m_h['MdMRE']:.4f}  "
              f"PRED25 {m_h['PRED25']:.4f}  PRED50 {m_h['PRED50']:.4f}")
        print(f"  calibrada por persona  MdMRE {m_c['MdMRE']:.4f}  "
              f"PRED25 {m_c['PRED25']:.4f}  PRED50 {m_c['PRED50']:.4f}")
        print(f"  intervalo 80%: cobertura {cov:.4f}  factor {np.exp(q):.2f}x")
        E.record("final", "humano_crudo", "split_t3", m_h)
        E.record("final", "humano_calibrado", "split_t3",
                 {**m_c, "cov80": cov, "factor80": float(np.exp(q))})

    r.to_csv("results/final_test.csv", index=False)
    print("\nescrito results/final_test.csv")


if __name__ == "__main__":
    main()
