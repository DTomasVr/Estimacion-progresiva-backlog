"""Dos conjuntos para la mitad de abajo que reducen el peso del corpus de Jira,
que es de donde salen las fugas ("inquilino" de tenant, "constructor",
"Anadir documentacion"). Hoy Jira es el 79 % de las filas.

  v7  SOLO sintetico            6.106 filas. Maxima adaptacion al dominio de
                                producto, riesgo de olvido catastrofico.
  v8  50/50                     sintetico completo + Jira submuestreado a la
                                misma cantidad. Conserva capacidad general y
                                reduce el peso de la fuente de las fugas.

Los dos se entrenan DESDE el checkpoint ya entrenado con Jira, asi que "quitar
Jira" significa dejar de reforzarlo, no no haberlo visto nunca.

out: data/processed/s2s_v7_*.parquet y s2s_v8_*.parquet
"""
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260927

for split in ["train", "val", "test"]:
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_v5_{split}.parquet")
    sint = d[d.estilo == "sintetico"]
    jira = d[d.estilo != "sintetico"]
    sint.reset_index(drop=True).to_parquet(
        f"{BASE}/data/processed/s2s_v7_{split}.parquet", index=False)
    n = min(len(jira), len(sint))
    mezcla = pd.concat([sint, jira.sample(n=n, random_state=SEED)],
                       ignore_index=True).sample(frac=1, random_state=SEED)
    mezcla.reset_index(drop=True).to_parquet(
        f"{BASE}/data/processed/s2s_v8_{split}.parquet", index=False)
    print(f"{split}: v5 {len(d)} ({len(jira)} jira + {len(sint)} sint) -> "
          f"v7 {len(sint)} | v8 {len(mezcla)} ({n} jira + {len(sint)} sint)")
