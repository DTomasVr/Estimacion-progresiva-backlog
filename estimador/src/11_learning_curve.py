"""Is ~17k rows enough? Measure the learning curve instead of guessing.

Trains on growing subsamples of the training split and scores on the FIXED
test split. If the curve has flattened well before the full training set, then
sample size is not the binding constraint -- signal is.

Models:
  median   -- predict the training median, ignores all features (the honest floor)
  knn      -- TF-IDF cosine KNN weighted by similarity (the thesis's approach)
  ridge    -- linear model on TF-IDF, log target (a stronger reference)
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

pd.set_option("display.width", 240)
SEED = 20260830
SIZES = [500, 1000, 2000, 4000, 8000, 13103]


def scores(true_h, pred_h):
    pred_h = np.clip(pred_h, 0.01, None)
    mre = np.abs(true_h - pred_h) / true_h
    return {
        "MAE_h": np.mean(np.abs(true_h - pred_h)),
        "MdAE_h": np.median(np.abs(true_h - pred_h)),
        "MdMRE": np.median(mre),
        "PRED25": float((mre <= .25).mean()),
        "RMSE_log": float(np.sqrt(np.mean((np.log(true_h) - np.log(pred_h)) ** 2))),
    }


def main():
    df = pd.read_parquet("data/processed/agix_effort_dataset.parquet")
    sw = df[df.is_software_task == 1]
    tr_all = sw[sw.split == "train"]
    te = sw[sw.split == "test"]
    print(f"train {len(tr_all):,}   test {len(te):,}\n")

    y_te = te.effort_hours.to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []

    for n in SIZES:
        n = min(n, len(tr_all))
        idx = rng.choice(len(tr_all), size=n, replace=False)
        tr = tr_all.iloc[idx]
        y_tr_log = np.log(tr.effort_hours.to_numpy())

        # --- baseline: ignore every feature ---
        med = float(tr.effort_hours.median())
        rows.append(dict(n=n, model="median", **scores(y_te, np.full(len(te), med))))

        vec = TfidfVectorizer(min_df=2, max_features=50000, ngram_range=(1, 2),
                              sublinear_tf=True, strip_accents="unicode")
        Xtr = vec.fit_transform(tr.text.fillna(""))
        Xte = vec.transform(te.text.fillna(""))

        # --- KNN weighted by TF-IDF similarity (thesis approach) ---
        k = min(10, n - 1)
        knn = KNeighborsRegressor(n_neighbors=k, metric="cosine", weights="distance")
        knn.fit(Xtr, y_tr_log)
        rows.append(dict(n=n, model="knn", **scores(y_te, np.exp(knn.predict(Xte)))))

        # --- ridge on log target ---
        rg = Ridge(alpha=1.0).fit(Xtr, y_tr_log)
        rows.append(dict(n=n, model="ridge", **scores(y_te, np.exp(rg.predict(Xte)))))

    res = pd.DataFrame(rows)
    for m in ["median", "knn", "ridge"]:
        print(f"--- {m} ---")
        print(res[res.model == m].drop(columns="model").round(3).to_string(index=False))
        print()

    print("=" * 70)
    print("MdMRE por tamano de entrenamiento (menor es mejor)")
    print("=" * 70)
    piv = res.pivot(index="n", columns="model", values="MdMRE").round(3)
    print(piv.to_string())
    print("\nPRED(25) por tamano (mayor es mejor)")
    print(res.pivot(index="n", columns="model", values="PRED25").round(3).to_string())

    best = res[res.model != "median"].sort_values("MdMRE").iloc[0]
    base = res[(res.model == "median") & (res.n == SIZES[-1])].iloc[0]
    print(f"\nmejor modelo   : {best.model} con n={int(best.n)}, "
          f"MdMRE {best.MdMRE:.3f}, PRED25 {best.PRED25:.3f}")
    print(f"linea base     : median, MdMRE {base.MdMRE:.3f}, PRED25 {base.PRED25:.3f}")
    gain = (base.MdMRE - best.MdMRE) / base.MdMRE * 100
    print(f"mejora sobre la linea base: {gain:.1f}% en MdMRE")

    # marginal value of the last doubling
    for m in ["knn", "ridge"]:
        s = res[res.model == m].sort_values("n")
        d = s.MdMRE.iloc[-2] - s.MdMRE.iloc[-1]
        print(f"  {m}: al pasar de {int(s.n.iloc[-2]):,} a {int(s.n.iloc[-1]):,} filas, "
              f"MdMRE cambia {d:+.4f}")


if __name__ == "__main__":
    main()
