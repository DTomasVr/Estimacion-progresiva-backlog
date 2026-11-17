"""Complete inventory of every model trained, with its dataset version.

Each registry entry carries the SHA-256 of the data file it was produced from,
so runs can be attributed to a dataset version rather than to memory.
"""
import hashlib
import json
import os
import sys

import pandas as pd

pd.set_option("display.width", 260)
pd.set_option("display.max_rows", 300)

VERSIONS = {
    "data/processed/pbe_effort_dataset.parquet": "v0 (sin limpiar)",
    "data/processed/pbe_effort_clean.parquet": "v1 (limpieza D1-D10)",
    "data/processed/pbe_effort_v2.parquet": "v2 (por desarrollador)",
}


def fingerprints():
    out = {}
    for path, label in VERSIONS.items():
        if os.path.exists(path):
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for c in iter(lambda: fh.read(1 << 20), b""):
                    h.update(c)
            out[h.hexdigest()[:16]] = label
    return out


def main():
    reg = json.load(open("results/experiments.json", encoding="utf8"))
    fp = fingerprints()
    rows = []
    for r in reg:
        m = r["metrics"]
        rows.append({
            "familia": r["arm"],
            "modelo": r["variant"],
            "particion": "aleatoria" if r["split"] == "split" else "proy. no vistos",
            "datos": fp.get(r["data_sha256"], f"desconocido:{r['data_sha256'][:8]}"),
            "MdMRE": m.get("MdMRE"),
            "PRED25": m.get("PRED25"),
            "RMSE_log": m.get("RMSE_log"),
            "compresion": m.get("compresion"),
            "spearman": m.get("spearman"),
            "cov80": m.get("cov80"),
            "AUC": m.get("AUC"),
            "accuracy": m.get("accuracy"),
        })
    df = pd.DataFrame(rows)
    df.to_csv("results/inventario_modelos.csv", index=False)

    print(f"corridas registradas: {len(df)}")
    print("\n=== por version de datos ===")
    print(df.datos.value_counts().to_string())

    reg_df = df[df.MdMRE.notna()].copy()
    for part in ["aleatoria", "proy. no vistos"]:
        d = reg_df[reg_df.particion == part].sort_values("MdMRE")
        print(f"\n{'='*100}\nREGRESION - particion {part} (ordenado por MdMRE)"
              f"\n{'='*100}")
        print(d[["familia", "modelo", "datos", "MdMRE", "PRED25", "RMSE_log",
                 "compresion", "spearman", "cov80"]].to_string(index=False))

    clf = df[df.AUC.notna()]
    if len(clf):
        print(f"\n{'='*100}\nCLASIFICADORES DE RIESGO (AUC)\n{'='*100}")
        print(clf.sort_values(["particion", "AUC"], ascending=[True, False])[
            ["modelo", "particion", "datos", "AUC"]].to_string(index=False))

    buk = df[df.accuracy.notna()]
    if len(buk):
        print(f"\n{'='*100}\nCLASIFICACION POR RANGOS (exactitud)\n{'='*100}")
        print(buk.sort_values(["particion", "accuracy"], ascending=[True, False])[
            ["modelo", "particion", "datos", "accuracy"]].to_string(index=False))

    print("\nescrito results/inventario_modelos.csv")


if __name__ == "__main__":
    main()
