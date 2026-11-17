"""Step 0: a three-way chronological split inside each project.

    |------- 60% train -------|-- 15% val --|---- 25% test ----|

Validation sits chronologically BEFORE test, never as a random slice, so no
future information reaches the selection stage. Everything -- feature blocks,
hyperparameters, conformal quantiles, early stopping -- is chosen on validation.
Test is read once, at the end.
"""
import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
PATH = "data/processed/pbe_effort_v3.parquet"
TR, VA = 0.60, 0.15
MIN_PROJ = 80


def main():
    df = pd.read_parquet(PATH)
    df["orden"] = df.created.fillna(pd.Timestamp("2000-01-01"))
    df = df.sort_values(["project", "orden", "task_id"]).reset_index(drop=True)
    sw = df.is_software_task == 1

    df["split_t3"] = pd.NA
    usados = []
    for proj, idx in df[sw].groupby("project").groups.items():
        idx = list(idx)
        n = len(idx)
        if n < MIN_PROJ:
            df.loc[idx, "split_t3"] = "train"      # too small to evaluate on
            continue
        a, b = int(n * TR), int(n * (TR + VA))
        df.loc[idx[:a], "split_t3"] = "train"
        df.loc[idx[a:b], "split_t3"] = "val"
        df.loc[idx[b:], "split_t3"] = "test"
        usados.append(dict(proyecto=proj, fuente=df.loc[idx[0], "source"],
                           n=n, train=a, val=b - a, test=n - b))

    r = pd.DataFrame(usados).sort_values("n", ascending=False)
    print(f"proyectos evaluables (>= {MIN_PROJ} tareas): {len(r)} de "
          f"{df[sw].project.nunique()}")
    print(r.to_string(index=False))
    print(f"\ntotales: {df[sw].split_t3.value_counts().to_dict()}")

    # a chronological split can still leak if the same text repeats over time
    s = df[sw]
    tr_h = set(s[s.split_t3 == "train"].text_hash)
    va_h = set(s[s.split_t3 == "val"].text_hash)
    te = s[s.split_t3 == "test"]
    print(f"\ncontrol de fuga por texto repetido:")
    print(f"  prueba que aparece en entrenamiento: "
          f"{int(te.text_hash.isin(tr_h).sum())}")
    print(f"  prueba que aparece en validacion   : "
          f"{int(te.text_hash.isin(va_h).sum())}")
    print(f"  validacion que aparece en entrenamiento: "
          f"{int(s[s.split_t3=='val'].text_hash.isin(tr_h).sum())}")

    # dates must actually be ordered where they exist
    con_fecha = s[s.created.notna()]
    malos = 0
    for proj, d in con_fecha.groupby("project"):
        for a, b in [("train", "val"), ("val", "test")]:
            x, y = d[d.split_t3 == a], d[d.split_t3 == b]
            if len(x) and len(y) and x.created.max() > y.created.min():
                malos += 1
    print(f"  proyectos con solape de fechas entre bloques: {malos}")

    df.to_parquet(PATH, compression="zstd", index=False)
    print(f"\nactualizado {PATH} con la columna split_t3")


if __name__ == "__main__":
    main()
