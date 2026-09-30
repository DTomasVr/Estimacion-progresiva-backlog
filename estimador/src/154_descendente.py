"""E48 - Estimacion descendente del proyecto, y combinarla con la ascendente.

Jorgensen (2004, "Top-Down and Bottom-Up Expert Estimation of Software
Development Effort") midio las dos estrategias con equipos de profesionales.
Dos resultados suyos importan aqui:

  1 La ascendente -- descomponer en tareas y sumar -- fue casi el doble de
    precisa: MRE medio 23 % y 31 % frente a 45 % y 55 % de la descendente. Eso
    valida la arquitectura que ya tenemos.
  2 Pero la condicion para que la descendente funcione es "recordar proyectos
    muy parecidos ya terminados". Cuando el estimador no tenia un proyecto
    parecido en la cabeza, la descendente fallaba.

La segunda es una oportunidad, no una advertencia, porque recordar proyectos
parecidos es justo lo que un vector sabe hacer. Si se representa el proyecto por
la media de los vectores de sus tareas, se pueden buscar los proyectos mas
parecidos del historial y estimar el total a partir de lo que costaron ellos.

Y hay una razon para esperar que la combinacion gane a las dos por separado: la
ascendente y la descendente fallan por motivos distintos. La ascendente arrastra
el desplazamiento comun del proyecto (E44); la descendente no lo arrastra, porque
no pasa por las tareas, pero es mas burda.

  0  ascendente            sumar las tareas, con smearing. Lo que ya hay
  1  descendente k vecinos horas por tarea de los k proyectos mas parecidos,
                           multiplicado por el numero de tareas
  2  descendente ajustada  igual, pero corrigiendo por el tamano tipico de tarea
                           que predice el modelo, para no suponer que todas las
                           tareas cuestan lo mismo
  3  combinacion           media geometrica, con el peso elegido dejando fuera
                           un proyecto cada vez

Supuesto que hay que declarar: la descendente usa los totales REALES de otros
proyectos. Eso es realista para una organizacion que ya ha entregado cosas, pero
el corpus mezcla organizaciones distintas, asi que aqui es una cota optimista de
lo que daria dentro de una sola empresa.
"""
import json
import os
import sys
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "4"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/descendente")
S = "=" * 120
MIN_TAREAS = 8


def mt(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    q = np.log(pred / real)
    m = float(np.median(q))
    lr = np.log(real)
    return dict(MAE_h=float(np.abs(pred - real).mean()),
                RMSE_h=float(np.sqrt(np.mean((pred - real) ** 2))),
                MdMRE=float(np.median(r)), MMRE=float(r.mean()),
                R2_log=float(1 - ((lr - np.log(pred)) ** 2).mean() / lr.var()),
                PRED25=float(np.mean(r <= .25)), PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    proy = sw.project.to_numpy()
    lp = np.load("results/analogia_cv/pred.npy")[:, 0]
    E5 = np.load(CACHE / "e5-large.npy")
    Vn = E5 / np.maximum(np.linalg.norm(E5, axis=1, keepdims=True), 1e-9)

    e = np.exp(ly - lp)
    tramo = pd.qcut(lp, 6, labels=False, duplicates="drop")
    ft = pd.DataFrame({"t": tramo, "e": e}).groupby("t").e.mean()
    est = np.exp(lp) * ft.reindex(tramo).to_numpy()

    d = pd.DataFrame({"p": proy, "real": y, "est": est, "lp": lp})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    orden = d.p.to_numpy()
    idx = {p: np.flatnonzero(orden == p) for p in pd.unique(orden)}
    nombres = list(idx)

    # el proyecto como la media de los vectores de sus tareas
    pos = {t: k for k, t in enumerate(sw.index)}
    fila = {p: d.index[orden == p] for p in nombres}
    mapa = d.index.to_numpy()
    del pos, fila, mapa
    orig = sw.index.to_numpy()
    sel = sw.project.isin(nombres).to_numpy()
    Vp = pd.DataFrame(Vn[sel]).groupby(sw.project[sel].to_numpy()).mean()
    Vp = Vp.reindex(nombres).to_numpy()
    Vp = Vp / np.maximum(np.linalg.norm(Vp, axis=1, keepdims=True), 1e-9)

    g = d.groupby("p")
    real = g.real.sum().reindex(nombres).to_numpy()
    asc = g.est.sum().reindex(nombres).to_numpy()
    n_t = g.size().reindex(nombres).to_numpy()
    tam_tip = np.exp(g.lp.mean().reindex(nombres).to_numpy())     # tamano tipico predicho
    print(f"{len(nombres)} proyectos · vectores de proyecto {Vp.shape}")

    Sim = Vp @ Vp.T
    np.fill_diagonal(Sim, -2)          # nunca mirarse a si mismo
    hpt = real / n_t                   # horas por tarea reales de cada proyecto
    filas = []
    mejores = {}
    for k in (3, 5, 10, 20, 40):
        v = np.argsort(-Sim, axis=1)[:, :k]
        desc = n_t * np.exp(np.median(np.log(hpt)[v], axis=1))
        # version ajustada: escalar por lo grande que el modelo cree que son las
        # tareas de ESTE proyecto frente a las de los vecinos
        rel = tam_tip / np.exp(np.median(np.log(tam_tip)[v], axis=1))
        desc_aj = desc * rel
        for etq, p in ((f"1 · descendente, k={k}", desc),
                       (f"2 · descendente ajustada, k={k}", desc_aj)):
            filas.append(dict(brazo=etq, **mt(real, p)))
            mejores[etq] = p
    filas.append(dict(brazo="0 · ascendente (sumar tareas)", **mt(real, asc)))
    t = pd.DataFrame(filas).sort_values("MdMRE")
    print(f"\n{S}\nDESCENDENTE FRENTE A ASCENDENTE\n{S}")
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "MMRE", "R2_log", "PRED25",
             "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))

    # combinacion, con el peso elegido dejando fuera un proyecto cada vez
    mejor_desc = t[t.brazo.str.startswith(("1 ·", "2 ·"))].iloc[0].brazo
    dsc = mejores[mejor_desc]
    la, ld = np.log(asc), np.log(dsc)
    pesos = np.zeros(len(real))
    for i in range(len(real)):
        k = np.ones(len(real), bool)
        k[i] = False
        mejor_w, mejor_v = 0., 1e9
        for w in np.arange(0, 1.01, .05):
            v = float(np.median(np.abs(w * ld[k] + (1 - w) * la[k] - np.log(real[k]))))
            if v < mejor_v:
                mejor_w, mejor_v = float(w), v
        pesos[i] = mejor_w
    comb = np.exp(pesos * ld + (1 - pesos) * la)
    print(f"\n{S}\nCOMBINACION  (mejor descendente: {mejor_desc})\n{S}")
    c = pd.DataFrame([dict(brazo="0 · solo ascendente", **mt(real, asc)),
                      dict(brazo=f"1 · solo {mejor_desc}", **mt(real, dsc)),
                      dict(brazo="2 · media geometrica 50/50",
                           **mt(real, np.exp(.5 * ld + .5 * la))),
                      dict(brazo=f"3 · peso elegido fuera de muestra "
                                 f"(mediana {np.median(pesos):.2f})", **mt(real, comb))])
    print(c[["brazo", "MAE_h", "RMSE_h", "MdMRE", "MMRE", "R2_log", "PRED25",
             "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))
    t.to_csv(OUT / "barrido.csv", index=False)
    c.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"mejor_descendente": mejor_desc, "peso_mediano": float(np.median(pesos))},
              open(OUT / "meta.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
