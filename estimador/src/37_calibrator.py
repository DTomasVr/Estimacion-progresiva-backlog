"""Per-developer calibration of the estimate, in the shape PBE would ship it.

The correction is a per-person offset in log space, shrunk toward the global
correction in proportion to how little history that person has -- a developer
with five tracked tasks should not be trusted as much as one with five hundred.

    correccion_d = (n_d * sesgo_d + k * sesgo_global) / (n_d + k)

k is a pseudo-count tuned on validation. It also defines the cold-start
behaviour for free: with n_d = 0 the correction collapses to the global one, so
a new developer is handled by the same formula rather than a special case.

Also reported: how much history a developer needs before their own correction
beats the global one, and a conformal interval on top of the corrected value.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

pd.set_option("display.width", 250)
E.DATA = "data/processed/pbe_effort_v2.parquet"


class Calibrador:
    """Corrects a developer's estimate using their own tracked history."""

    def __init__(self, k=20.0, pendiente=False):
        self.k = k
        self.pendiente = pendiente

    def fit(self, est, real, dev):
        le, lr = np.log(est), np.log(real)
        self.global_ = float(np.median(lr - le))
        self.slope_ = 1.0
        if self.pendiente:
            # a shared slope captures the size-dependent part of the bias
            A = np.vstack([le, np.ones_like(le)]).T
            self.slope_, self.inter_ = np.linalg.lstsq(A, lr, rcond=None)[0]
            resid = lr - (self.slope_ * le + self.inter_)
            self.global_ = float(np.median(resid))
            base = resid
        else:
            base = lr - le
        df = pd.DataFrame({"d": np.asarray(dev), "b": base})
        g = df.groupby("d").b.agg(["median", "size"])
        self.bias_ = ((g["size"] * g["median"] + self.k * self.global_)
                      / (g["size"] + self.k))
        self.n_ = g["size"]
        return self

    def predict(self, est, dev):
        le = np.log(est)
        adj = pd.Series(np.asarray(dev)).map(self.bias_).fillna(self.global_)
        base = self.slope_ * le + self.inter_ if self.pendiente else le
        return np.exp(base + adj.to_numpy())

    def historia(self, dev):
        return pd.Series(np.asarray(dev)).map(self.n_).fillna(0).to_numpy()


def datos():
    t = pd.read_csv("data/raw/jones/SiP/Sip-task-info.csv", encoding="cp1252")
    t = t.drop_duplicates("TaskNumber")
    t["task_id"] = "SIP-" + t.TaskNumber.astype(str)
    df = pd.read_parquet(E.DATA)
    sw = df[(df.is_software_task == 1) & (df.source == "SiP")]
    sw = sw.merge(t[["task_id", "HoursEstimate", "AssignedToID"]], on="task_id",
                  how="left")
    return sw[sw.HoursEstimate.notna() & (sw.HoursEstimate > 0)].copy()


def resumen(y, p, etiqueta):
    m = E.point_metrics(y, p)
    return dict(modelo=etiqueta, MdMRE=round(m["MdMRE"], 4),
                PRED25=round(m["PRED25"], 4), PRED50=round(m["PRED50"], 4),
                RMSE_log=round(m["RMSE_log"], 4),
                compresion=round(float(np.std(np.log(p)) / np.std(np.log(y))), 3),
                grandes=f"{int(((y>16)&(p>16)).sum())}/{int((y>16).sum())}"), m


def main():
    sw = datos()
    print(f"tareas con estimacion: {len(sw):,}   "
          f"desarrolladores: {sw.AssignedToID.nunique()}")
    print("\nhistorial por desarrollador (top 10):")
    print(sw.AssignedToID.value_counts().head(10).to_string())

    for split_col in ["split", "split_project"]:
        tr = sw[sw[split_col] == "train"]
        va = sw[sw["split"] == "val"] if split_col == "split" else tr.sample(
            frac=.15, random_state=7)
        te = sw[sw[split_col] == "test"]
        if len(te) < 100:
            continue
        print(f"\n{'='*78}\nsplit={split_col}  train {len(tr):,}  test {len(te):,}"
              f"\n{'='*78}")
        y_te = te.effort_hours.to_numpy()
        rows = []

        rows.append(resumen(y_te, te.HoursEstimate.to_numpy(), "humano crudo")[0])

        # tune the shrinkage on validation, never on test
        mejor_k, mejor = None, np.inf
        for k in [0, 2, 5, 10, 20, 50, 100, 200, 500]:
            c = Calibrador(k=k).fit(tr.HoursEstimate, tr.effort_hours,
                                    tr.AssignedToID)
            p = c.predict(va.HoursEstimate, va.AssignedToID)
            s = float(np.median(np.abs(va.effort_hours.to_numpy() - p)
                                / va.effort_hours.to_numpy()))
            if s < mejor:
                mejor, mejor_k = s, k
        print(f"  k elegido en validacion: {mejor_k}  (MdMRE val {mejor:.4f})")

        cg = Calibrador(k=1e9).fit(tr.HoursEstimate, tr.effort_hours,
                                   tr.AssignedToID)
        rows.append(resumen(y_te, cg.predict(te.HoursEstimate, te.AssignedToID),
                            "correccion global")[0])

        cal = Calibrador(k=mejor_k).fit(tr.HoursEstimate, tr.effort_hours,
                                        tr.AssignedToID)
        p_cal = cal.predict(te.HoursEstimate, te.AssignedToID)
        rows.append(resumen(y_te, p_cal, f"por desarrollador (k={mejor_k})")[0])

        c0 = Calibrador(k=0).fit(tr.HoursEstimate, tr.effort_hours,
                                 tr.AssignedToID)
        rows.append(resumen(y_te, c0.predict(te.HoursEstimate, te.AssignedToID),
                            "por desarrollador sin encoger")[0])

        cs = Calibrador(k=mejor_k, pendiente=True).fit(
            tr.HoursEstimate, tr.effort_hours, tr.AssignedToID)
        rows.append(resumen(y_te, cs.predict(te.HoursEstimate, te.AssignedToID),
                            "por desarrollador + pendiente")[0])

        r = pd.DataFrame(rows)
        print(r.to_string(index=False))
        for _, x in r.iterrows():
            E.record("calibrador", x.modelo.replace(" ", "_"), split_col,
                     {"MdMRE": x.MdMRE, "PRED25": x.PRED25, "PRED50": x.PRED50,
                      "RMSE_log": x.RMSE_log, "compresion": x.compresion})

        # how much history does a developer need?
        hist = cal.historia(te.AssignedToID)
        b = pd.cut(hist, [-1, 0, 25, 100, 400, np.inf],
                   labels=["sin historial", "1-25", "26-100", "101-400", ">400"])
        d = pd.DataFrame({
            "real": y_te, "humano": te.HoursEstimate.to_numpy(),
            "calibrado": p_cal, "banda": b})
        g = d.groupby("banda", observed=True).apply(lambda x: pd.Series({
            "n": len(x),
            "MdMRE_humano": np.median(np.abs(x.real - x.humano) / x.real),
            "MdMRE_calibrado": np.median(np.abs(x.real - x.calibrado) / x.real),
        }), include_groups=False)
        g["mejora_%"] = ((g.MdMRE_humano - g.MdMRE_calibrado)
                         / g.MdMRE_humano * 100).round(1)
        print(f"\n  efecto segun el historial del desarrollador:\n{g.round(4).to_string()}")

        # conformal band on the calibrated value
        p_va = cal.predict(va.HoursEstimate, va.AssignedToID)
        res = np.abs(np.log(va.effort_hours.to_numpy()) - np.log(p_va))
        for lv in (0.5, 0.8):
            q = float(np.quantile(res, lv))
            lo, hi = p_cal / np.exp(q), p_cal * np.exp(q)
            cov = float(np.mean((y_te >= lo) & (y_te <= hi)))
            print(f"  intervalo {lv:.0%}: cobertura real {cov:.3f}  "
                  f"factor {np.exp(q):.2f}x  ancho mediano "
                  f"{float(np.median(hi-lo)):.2f} h")

        if split_col == "split":
            out = {"k": mejor_k, "global_log_offset": cal.global_,
                   "por_desarrollador": {str(kk): float(vv)
                                         for kk, vv in cal.bias_.items()},
                   "n_por_desarrollador": {str(kk): int(vv)
                                           for kk, vv in cal.n_.items()}}
            with open("results/calibrador.json", "w", encoding="utf8") as fh:
                json.dump(out, fh, indent=2)
            print("\n  escrito results/calibrador.json")


if __name__ == "__main__":
    main()
