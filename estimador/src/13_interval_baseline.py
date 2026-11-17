"""Predict a RANGE instead of a point, and check the range is honest.

With this much irreducible label noise a point estimate is misleading. A split
conformal interval gives a band with a guaranteed coverage level, calibrated on
the validation split and verified on the untouched test split.

For a sprint planner "between 2 and 9 hours, 80% of the time" is both more
truthful and more actionable than "4.3 hours".
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

pd.set_option("display.width", 240)
LEVELS = [0.50, 0.80, 0.90]


def build(tr, ot, text_col="text"):
    vec = TfidfVectorizer(min_df=2, max_features=50000, ngram_range=(1, 2),
                          sublinear_tf=True, strip_accents="unicode")
    Xtr = vec.fit_transform(tr[text_col])
    enc = OneHotEncoder(handle_unknown="ignore")
    Ktr = enc.fit_transform(tr[["task_kind", "source"]])
    out = [sparse.hstack([Xtr, Ktr]).tocsr()]
    for d in ot:
        out.append(sparse.hstack([vec.transform(d[text_col]),
                                  enc.transform(d[["task_kind", "source"]])]).tocsr())
    return out


def main():
    df = pd.read_parquet("data/processed/pbe_effort_clean.parquet")
    sw = df[df.is_software_task == 1]
    tr, va, te = (sw[sw.split == s] for s in ("train", "val", "test"))
    print(f"train {len(tr):,}  val {len(va):,}  test {len(te):,}\n")

    Xtr, Xva, Xte = build(tr, [va, te])
    model = Ridge(alpha=1.0).fit(Xtr, np.log(tr.effort_hours.to_numpy()))

    # calibration residuals on the validation split (never seen in training)
    res_va = np.abs(np.log(va.effort_hours.to_numpy()) - model.predict(Xva))
    pred_te = model.predict(Xte)
    y_te = te.effort_hours.to_numpy()

    print("--- intervalo conforme (calibrado en val, verificado en test) ---")
    rows = []
    for lv in LEVELS:
        q = float(np.quantile(res_va, lv))
        lo, hi = np.exp(pred_te - q), np.exp(pred_te + q)
        cov = float(((y_te >= lo) & (y_te <= hi)).mean())
        rows.append(dict(nivel=f"{lv:.0%}", cobertura=round(cov, 3),
                         factor=round(float(np.exp(q)), 2),
                         ancho_h=round(float(np.median(hi - lo)), 2),
                         lo_med=round(float(np.median(lo)), 2),
                         hi_med=round(float(np.median(hi)), 2)))
    r = pd.DataFrame(rows)
    print(r.to_string(index=False))
    print("\n  'factor' = cuantas veces se multiplica/divide la prediccion puntual")
    print("  'cobertura' deberia aproximar el nivel nominal")

    # a naive interval that ignores the text entirely, for reference
    print("\n--- referencia: banda fija alrededor de la mediana de entrenamiento ---")
    med = float(tr.effort_hours.median())
    res_base = np.abs(np.log(va.effort_hours.to_numpy()) - np.log(med))
    for lv in LEVELS:
        q = float(np.quantile(res_base, lv))
        lo, hi = med / np.exp(q), med * np.exp(q)
        cov = float(((y_te >= lo) & (y_te <= hi)).mean())
        print(f"  {lv:.0%}: cobertura {cov:.3f}  factor {np.exp(q):5.2f}  "
              f"banda {lo:5.2f}-{hi:6.2f} h  (ancho {hi-lo:6.2f} h)")

    print("\n--- por fuente, nivel 80% ---")
    q80 = float(np.quantile(res_va, 0.80))
    lo, hi = np.exp(pred_te - q80), np.exp(pred_te + q80)
    t = te.assign(lo=lo, hi=hi, ok=(y_te >= lo) & (y_te <= hi))
    print(t.groupby("source").agg(
        n=("ok", "size"), cobertura=("ok", "mean"),
        ancho_mediano=("hi", lambda s: np.nan)).round(3).to_string())
    print(t.groupby("source").apply(
        lambda d: pd.Series({"ancho_mediano_h": np.median(d.hi - d.lo)}),
        include_groups=False).round(2).to_string())

    print("\n--- por categoria, nivel 80% ---")
    print(t.groupby("task_kind").agg(n=("ok", "size"), cobertura=("ok", "mean"))
          .sort_values("n", ascending=False).round(3).to_string())

    print("\n--- generalizacion a proyectos no vistos (split_project) ---")
    trp = sw[sw.split_project == "train"]
    tep = sw[sw.split_project == "test"]
    # hold out a calibration slice from the project-train side
    rng = np.random.default_rng(7)
    m = rng.random(len(trp)) < 0.15
    trp2, vap = trp[~m], trp[m]
    Xtr2, Xva2, Xte2 = build(trp2, [vap, tep])
    mp = Ridge(alpha=1.0).fit(Xtr2, np.log(trp2.effort_hours.to_numpy()))
    rv = np.abs(np.log(vap.effort_hours.to_numpy()) - mp.predict(Xva2))
    pt = mp.predict(Xte2)
    yt = tep.effort_hours.to_numpy()
    mre = np.abs(yt - np.exp(pt)) / yt
    print(f"  MdMRE en proyectos no vistos: {np.median(mre):.3f}  "
          f"PRED25 {float((mre<=.25).mean()):.3f}")
    q = float(np.quantile(rv, 0.80))
    cov = float(((yt >= np.exp(pt - q)) & (yt <= np.exp(pt + q))).mean())
    print(f"  cobertura al 80%: {cov:.3f}  (factor {np.exp(q):.2f})")


if __name__ == "__main__":
    main()
