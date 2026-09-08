"""Render the experiment registry as the comparison tables for the thesis.

Reads results/experiments.json -- every arm scored on the same splits with the
same metrics -- and prints the point-estimate, interval and cross-lingual
comparisons, plus a CSV for the write-up.
"""
import sys

import pandas as pd

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

pd.set_option("display.width", 240)

POINT = ["MdMRE", "PRED25", "PRED50", "MdAE_h", "RMSE_log"]
INTERVAL = ["cov50", "cov80", "cov90", "factor80", "width80_h"]
CROSS = ["MdMRE", "MdMRE_es", "drift_median", "drift_abs_median", "drift_within10"]


def section(title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")


def main():
    t = E.table()
    t["modelo"] = t.arm + "/" + t.variant

    for split in ["split", "split_project"]:
        d = t[t.split == split]
        if d.empty:
            continue
        label = ("particion aleatoria" if split == "split"
                 else "proyectos no vistos")
        section(f"Estimacion puntual - {label}")
        print(d.sort_values("MdMRE")[["modelo"] + POINT].round(4)
              .to_string(index=False))

        base = d[d.arm == "baseline"]
        if not base.empty:
            b = float(base.MdMRE.iloc[0])
            best = d[d.arm != "baseline"].sort_values("MdMRE").iloc[0]
            print(f"\n  mejor: {best.modelo}  MdMRE {best.MdMRE:.4f}  "
                  f"({(b-best.MdMRE)/b*100:.1f}% mejor que la linea base)")

        section(f"Intervalos conformes - {label}")
        print(d.sort_values("factor80")[["modelo"] + INTERVAL].round(4)
              .to_string(index=False))

    cross = t[t.arm == "bilingual"]
    if not cross.empty:
        section("Robustez cross-lingual (entrenado en EN frente a EN+ES)")
        c = cross[["modelo", "split"] + [k for k in CROSS if k in cross.columns]]
        print(c.round(4).to_string(index=False))
        print("\n  drift_median      = razon mediana de horas predichas ES/EN "
              "(1,00 = invariante)")
        print("  drift_within10    = fraccion de tareas cuya prediccion cambia "
              "menos de un 10%")
        print("  MdMRE_es          = exactitud sobre el test traducido al espanol")

    t.drop(columns=["modelo"]).to_csv("results/comparacion_brazos.csv", index=False)
    print(f"\nescrito results/comparacion_brazos.csv ({len(t)} filas)")


if __name__ == "__main__":
    main()
