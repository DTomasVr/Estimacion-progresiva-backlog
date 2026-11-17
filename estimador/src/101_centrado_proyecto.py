"""E7 · Centrar por proyecto: usar TODO el corpus sin heredar su escala.

Hasta ahora la eleccion era binaria y mala en los dos extremos. Entrenar con
Apache mete 47.278 filas cuya escala (0,83 h de mediana) no es la del trabajo de
producto (2,00 h), y el modelo aprende a predecirla desde el texto: el sesgo en
proyectos externos no-Apache se va a 0,44. Quitar Apache arregla el sesgo pero
tira el 74 % de los datos.

Hay una tercera opcion que es la herramienta estandar para esto y que no se ha
probado: **modelo de efectos mixtos**, o en su version simple, centrar el
objetivo por proyecto. Se entrena sobre

    log(horas) - media de log(horas) DE ESE PROYECTO

calculada solo con filas de entrenamiento. Asi el modelo no puede ganar nada
prediciendo la escala del proyecto -- se la han quitado -- y tiene que aprender
lo unico que transfiere: si esta tarea es grande O PEQUEÑA PARA SU PROYECTO. Con
eso, las 47.278 filas de Apache dejan de contaminar la escala y pasan a aportar
señal de dificultad relativa.

Al predecir hace falta devolver una escala. Se leen las tres que existen:

    global      la mediana del entrenamiento            (arranque en frio puro)
    proyecto    la mediana real del proyecto            (oraculo / con historial)
    fuente      la mediana de la organizacion

Y se mide aparte lo que de verdad dice si el texto mejoro: el R² DENTRO del
proyecto y el Spearman, que no dependen de acertar la escala.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 235)
DATA = "data/processed/pbe_effort_v4.parquet"
CACHE = Path("data/processed/embeddings/codificadores")
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
OUT = Path("results/centrado")
SEP = "=" * 132


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0)


def metricas(y, p, proyecto=None):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    ly, lp = np.log(y), np.log(p)
    d = dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lp - ly))),
             PRED25=m["PRED25"], MAE_h=m["MAE_h"],
             R2=float(1 - ((ly - lp) ** 2).mean() / ly.var()),
             sesgo=float(np.exp(np.median(lp - ly))),
             spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))
    if proyecto is not None:
        g = pd.DataFrame({"p": proyecto, "ly": ly, "lp": lp})
        dy = ly - g.groupby("p").ly.transform("mean").to_numpy()
        dp = lp - g.groupby("p").lp.transform("mean").to_numpy()
        d["R2_intra"] = float(1 - ((dy - dp) ** 2).mean() / dy.var())
        d["rho_intra"] = float(pd.Series(dy).corr(pd.Series(dp), method="spearman"))
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy().reset_index(drop=True)
    sw["apache"] = sw.fuente_jira.eq("Apache")
    e5 = CACHE / "e5-large.npy"
    if e5.exists() and len(np.load(e5)) == int((~sw.apache).sum()):
        print("aviso: e5-large solo está codificado para el corpus sin Apache; "
              "se usa mpnet de títulos, que cubre todo")
    emb = np.load(f"{EMB_TIT}.npy")
    ids = np.load(f"{EMB_TIT}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    X = emb[[pos[t] for t in sw.task_id]].astype(np.float32)

    tr_todo = (sw.split_t3 == "train").to_numpy()
    tr_sin = tr_todo & (~sw.apache).to_numpy()
    bloques = {"prueba fija": (sw.split_t3 == "test").to_numpy(),
               "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
               "prueba TAWOS": ((sw.split_t3 == "test") & (sw.source == "TAWOS")).to_numpy(),
               "externo no Apache": ((sw.split_t3 == "externo") & (~sw.apache)).to_numpy(),
               "externo (94 % Apache)": (sw.split_t3 == "externo").to_numpy()}
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    print(f"corpus completo {len(sw):,} · entrenamiento con Apache {tr_todo.sum():,} · "
          f"sin Apache {tr_sin.sum():,}")

    # medias de proyecto calculadas SOLO con filas de entrenamiento
    def medias(mask):
        g = pd.DataFrame({"p": sw.project, "ly": ly})[mask].groupby("p").ly.mean()
        return g, float(np.median(ly[mask]))

    resultados = []
    for etq_tr, mtr in (("todo (con Apache)", tr_todo), ("sin Apache", tr_sin)):
        g_proy, med_global = medias(mtr)
        centro = sw.project.map(g_proy)
        cubierto = centro.notna().to_numpy()
        centro = centro.fillna(med_global).to_numpy()

        crudo = hgb().fit(X[mtr], ly[mtr])
        cent = hgb().fit(X[mtr], (ly - centro)[mtr])
        for b, k in bloques.items():
            proy = sw.project.to_numpy()[k]
            filas = {
                f"crudo · {etq_tr}": np.exp(crudo.predict(X[k])),
                f"centrado + mediana global · {etq_tr}":
                    np.exp(cent.predict(X[k]) + med_global),
                f"centrado + media REAL del proyecto · {etq_tr}":
                    np.exp(cent.predict(X[k]) + centro[k]),
            }
            for nombre, p in filas.items():
                resultados.append(dict(entrenamiento=etq_tr, bloque=b, brazo=nombre,
                                       n=int(k.sum()),
                                       cobertura_centro=float(cubierto[k].mean()),
                                       **metricas(y[k], p, proy)))
        print(f"  ajustado con «{etq_tr}»", flush=True)

    t = pd.DataFrame(resultados)
    t.to_csv(OUT / "resumen.csv", index=False)
    cols = ["brazo", "n", "MdMRE", "MdAElog", "PRED25", "R2", "sesgo", "spearman",
            "R2_intra", "rho_intra"]
    for b in bloques:
        print(f"\n{SEP}\n{b}\n{SEP}")
        v = t[t.bloque == b].sort_values("MdAElog")
        print(v[cols].round(4).to_string(index=False))

    print(f"\n{SEP}\nLA PREGUNTA DE FONDO: ¿MEJORA EL TEXTO AL CENTRAR?\n{SEP}")
    print("  (R²_intra y ρ_intra no dependen de acertar la escala)")
    for b in ("prueba fija", "externo no Apache"):
        v = t[t.bloque == b]
        print(f"\n  {b}")
        for _, r in v.iterrows():
            print(f"    {r.brazo:52s} R²_intra {r.R2_intra:+.4f} · "
                  f"ρ_intra {r.rho_intra:+.4f}")
    json.dump({"bloques": list(bloques)}, open(OUT / "meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
