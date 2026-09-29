"""Evaluar un juego de vectores ya cacheado, sin esperar a que E35 termine.

E35 entrena los brazos en serie y solo evalua al final, pero deja los vectores de
cada brazo en la cache en cuanto lo termina. Esto los lee y los pasa por EL MISMO
HGB que usa el canal congelado, que es la comparacion que aisla al codificador:
lo unico que cambia entre lecturas es quien produce los vectores.

    python src/143_eval_codificador.py [B] [C] ...
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
CACHE = Path("data/processed/embeddings/codificadores")
S = "=" * 116


def metricas(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    q = np.log(p / y)
    r = np.abs(p - y) / y
    m = float(np.median(q))
    return dict(MAE_h=float(np.abs(p - y).mean()),
                MdAElog=float(np.median(np.abs(q))),
                zeta_pc=float((np.exp(np.median(np.abs(q))) - 1) * 100),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100),
                R2log=float(1 - ((np.log(y) - np.log(p)) ** 2).mean() / np.log(y).var()),
                PRED25=float(np.mean(r <= .25)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                rho=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def main():
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    tr = (sw.split_t3 == "train").to_numpy()
    bl = {"validacion": (sw.split_t3 == "val").to_numpy(),
          "prueba fija": (sw.split_t3 == "test").to_numpy(),
          "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
          "externo no Apache": (sw.split_t3 == "externo").to_numpy()}

    juegos = {"e5-large congelado": "e5-large.npy",
              "A · ajustado escalar cuadratico (el actual)": "e5-ajustado.npy"}
    for a in sys.argv[1:]:
        f = CACHE / f"e5-ajustado-{a}.npy"
        if f.exists():
            juegos[f"{a} · ajustado con el objetivo nuevo"] = f.name
        else:
            print(f"  aviso: {f.name} todavia no esta en la cache")
    filas = []
    for etq, f in juegos.items():
        V = np.load(CACHE / f)
        m = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                          l2_regularization=1., random_state=0)
        m.fit(V[tr], ly[tr])
        p = np.exp(m.predict(V))
        for b, k in bl.items():
            filas.append(dict(bloque=b, codificador=etq, n=int(k.sum()),
                              **metricas(y[k], p[k])))
        print(f"  {etq} evaluado", flush=True)
    t = pd.DataFrame(filas)
    cols = ["codificador", "MAE_h", "MdAElog", "zeta_pc", "R2log", "PRED25", "factor2", "rho"]
    for b in bl:
        print(f"\n{S}\n{b}  (el MISMO HGB sobre cada juego de vectores)\n{S}")
        print(t[t.bloque == b][cols].round(4).to_string(index=False))
    t.to_csv("results/finetune_objetivos/parcial.csv", index=False)


if __name__ == "__main__":
    main()
