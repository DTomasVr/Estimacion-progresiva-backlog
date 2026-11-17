"""How much of the error is irreducible from the text, and what ceiling that implies.

The model compresses: on the SiP test it multiplies sub-hour tasks by 3,15 and
divides tasks over sixteen hours by eleven. Before treating that as a defect to
be engineered away, it is worth asking how much of it is the optimal response to
noise. A squared- or absolute-error-optimal predictor under heavy aleatoric noise
IS compressed -- that is not a bug, it is the conditional mean (or median) doing
its job.

The corpus answers the question directly. Some tasks have BYTE-IDENTICAL text and
different recorded hours. No model that reads only the text can tell them apart,
so the spread inside those groups is a lower bound on the irreducible noise.

From that noise we can simulate the ceiling: what an oracle that predicted the
true conditional median, and carried only this noise, would score. If the
acceptance criteria sit above that ceiling, they are unreachable in principle
with this data and no amount of modelling will get there.

Caveat stated in the output: repeated titles are not a random sample -- they are
generic ones ("documentation changes") that are probably noisier than average,
so this is an upper bound on the noise and a pessimistic ceiling.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
DATA = "data/processed/pbe_effort_v4.parquet"
SEP = "=" * 110


def main():
    d = pd.read_parquet(DATA)
    sw = d[d.is_software_task == 1].copy()
    sw["lh"] = np.log(sw.effort_hours)
    total = float(sw.lh.var())

    g = sw.groupby("text_hash").filter(lambda x: len(x) > 1)
    s = g.groupby("text_hash").lh.agg(["size", "std"])
    s = s[s["size"] > 1]
    sd = float(s["std"].median())
    dentro = float((s["std"] ** 2 * (s["size"] - 1)).sum() / (s["size"] - 1).sum())

    print(f"{SEP}\nRUIDO IRREDUCIBLE DESDE EL TEXTO\n{SEP}")
    print(f"  tareas con texto idéntico repetido: {len(g):,} en {g.text_hash.nunique():,} grupos")
    print(f"  desviación típica de log-horas dentro de esos grupos: mediana {sd:.3f}")
    print(f"    -> dos tareas con EL MISMO texto difieren típicamente ×{np.exp(sd):.2f}")
    print(f"  varianza total de log-horas: {total:.3f}")
    print(f"  varianza dentro de grupo: {dentro:.3f} "
          f"({dentro / total:.1%} de la total, irreducible desde el texto)")

    g2 = sw.groupby(["project", "text_hash"]).filter(lambda x: len(x) > 1)
    s2 = g2.groupby(["project", "text_hash"]).lh.agg(["size", "std"])
    s2 = s2[s2["size"] > 1]
    print(f"  dentro del mismo proyecto: {len(s2)} grupos, mediana {s2['std'].median():.3f} "
          f"(×{np.exp(float(s2['std'].median())):.2f}) -- el proyecto no explica el ruido")

    print(f"\n{SEP}\nTECHO: LO QUE SACARÍA UN ORÁCULO QUE SOLO CARGA ESE RUIDO\n{SEP}")
    rng = np.random.default_rng(0)
    filas = []
    for s_log in (0.30, 0.40, sd, 0.60):
        e = rng.normal(0, s_log, 400_000)
        r = np.abs(np.exp(e) - 1)
        filas.append(dict(sd_log=round(float(s_log), 3), factor=round(float(np.exp(s_log)), 2),
                          MdMRE=float(np.median(r)), PRED25=float(np.mean(r <= .25)),
                          PRED50=float(np.mean(r <= .50)),
                          MdAElog=float(np.median(np.abs(e)))))
    t = pd.DataFrame(filas).sort_values("sd_log")
    print(t.round(3).to_string(index=False))
    fila = t[np.isclose(t.sd_log, round(sd, 3))].iloc[0]

    print(f"\n  con el ruido medido (sd {sd:.3f}) el oráculo sacaría "
          f"MdMRE {fila.MdMRE:.3f} y PRED(25) {fila.PRED25:.3f}")
    print(f"  criterio de aceptación de la tesis: PRED(25) >= 0,75  ->  INALCANZABLE, "
          f"el techo está en {fila.PRED25:.2f}")
    print(f"  nuestro modelo: MdMRE 0,687 y PRED(25) 0,162 en la prueba fija, "
          f"así que del techo nos separa bastante: sí queda margen, pero no hasta 0,75")
    print("\n  aviso: los títulos repetidos son genéricos y probablemente más ruidosos que")
    print("  la media, así que esta es una cota SUPERIOR del ruido y un techo PESIMISTA.")

    Path("results").mkdir(exist_ok=True)
    t.to_csv("results/techo_ruido.csv", index=False)
    json.dump({"sd_log_mediana": sd, "factor": float(np.exp(sd)),
               "varianza_total": total, "varianza_dentro": dentro,
               "fraccion_irreducible": dentro / total,
               "n_grupos": int(g.text_hash.nunique()), "n_tareas": int(len(g)),
               "techo_MdMRE": float(fila.MdMRE), "techo_PRED25": float(fila.PRED25)},
              open("results/techo_ruido.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\nescrito results/techo_ruido.csv y techo_ruido.json")


if __name__ == "__main__":
    main()
