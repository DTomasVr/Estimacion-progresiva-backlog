"""Experiment C: which feature blocks actually earn their place?

Adds and removes blocks one at a time so the contribution of each is visible,
instead of assuming that more features help. Run with both a linear head and a
tree head, because they react very differently to redundant inputs.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402
import features as F  # noqa: E402

pd.set_option("display.width", 240)

BLOCKS = {
    "emb":            dict(use_emb=True,  use_kind=False, use_priority=False,
                           use_source=False, use_numeric=False, use_project=False),
    "emb+cat":        dict(use_emb=True,  use_kind=True,  use_priority=False,
                           use_source=True,  use_numeric=False, use_project=False),
    "emb+num":        dict(use_emb=True,  use_kind=False, use_priority=False,
                           use_source=False, use_numeric=True,  use_project=False),
    "emb+proj":       dict(use_emb=True,  use_kind=False, use_priority=False,
                           use_source=False, use_numeric=False, use_project=True),
    "emb+cat+num":    dict(use_emb=True,  use_kind=True,  use_priority=True,
                           use_source=True,  use_numeric=True,  use_project=False),
    "todo":           dict(use_emb=True,  use_kind=True,  use_priority=True,
                           use_source=True,  use_numeric=True,  use_project=True),
    "sin_emb":        dict(use_emb=False, use_kind=True,  use_priority=True,
                           use_source=True,  use_numeric=True,  use_project=True),
}


def main():
    emb, pos = F.load_embeddings_and_pos()
    rows = []
    for split_col in ["split", "split_project"]:
        tr, va, te = E.load(split_col)
        y_tr = np.log(tr.effort_hours.to_numpy())
        y_va, y_te = va.effort_hours.to_numpy(), te.effort_hours.to_numpy()
        print(f"\n{'='*78}\nsplit={split_col}\n{'='*78}")
        for name, opts in BLOCKS.items():
            fb = F.FeatureBuilder(**opts)
            Xtr, Xva, Xte = fb.fit_transform(tr, [va, te], emb, pos)
            for head, mk in [("ridge", lambda: Ridge(alpha=1.0)),
                             ("hgb", lambda: HistGradientBoostingRegressor(
                                 max_iter=400, learning_rate=0.06,
                                 max_depth=6, random_state=0))]:
                mdl = mk().fit(Xtr, y_tr)
                lp_te, lp_va = mdl.predict(Xte), mdl.predict(Xva)
                m = E.point_metrics(y_te, np.exp(lp_te))
                m.update(E.interval_metrics(lp_va, y_va, lp_te, y_te))
                E.record("ablation", f"{head}_{name}", split_col, m,
                         config={"bloques": name, "n_features": int(Xtr.shape[1])})
                rows.append(dict(split=split_col, head=head, bloques=name,
                                 n_feat=Xtr.shape[1], MdMRE=round(m["MdMRE"], 4),
                                 PRED25=round(m["PRED25"], 4),
                                 RMSE_log=round(m["RMSE_log"], 4)))
                print(f"  {head:5s} {name:12s} n={Xtr.shape[1]:4d}  "
                      f"MdMRE {m['MdMRE']:.4f}  PRED25 {m['PRED25']:.4f}  "
                      f"RMSElog {m['RMSE_log']:.4f}")

    df = pd.DataFrame(rows)
    print(f"\n{'='*78}\nresumen por bloque\n{'='*78}")
    print(df.pivot_table(index=["bloques", "head"], columns="split",
                         values="MdMRE").round(4).to_string())


if __name__ == "__main__":
    main()
