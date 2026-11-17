"""E108 - Estrechar la banda POR TAREA sin perder la cobertura.

La banda por tarea es simetrica en log y la misma para todas: [h/4,32 , h*4,32],
o sea x18,6 de anchura. Cubre el 77,6 % (E99). Es lo que dice el ruido medio,
pero es la misma anchura para "corregir una errata" que para "migrar el
esquema", y eso es lo que se puede mejorar: no bajar la incertidumbre media,
sino REPARTIRLA, dando banda estrecha donde el modelo sabe mas.

Cuatro formas de que la anchura dependa de la tarea, todas con garantia
conforme (el cuantil se calibra en pliegues que no contienen la tarea):

  A  la de hoy: una sola anchura
  B  por ACUERDO DE LOS VECINOS. El artefacto ya calcula la dispersion de las
     horas de las 25 tareas mas parecidas. Si concuerdan, la tarea es facil.
     Banda conforme por quintil de esa dispersion
  C  CUANTILES APRENDIDOS (CQR, Romano et al. 2019): dos HGB con perdida
     cuantil al 10 % y al 90 %, corregidos con el residuo conforme. El modelo
     aprende directamente que tareas son mas inciertas
  D  por el VALOR PREDICHO. La incertidumbre relativa no es igual en 0,5 h que
     en 20 h; banda conforme por quintil de la prediccion
  E  con la ESTIMACION DEL USUARIO, donde la hay: el residuo del corrector es
     mucho menor (MdMRE 0,32 contra 0,74), asi que su banda debe serlo

Se mide la cobertura al 80 % global y por tramo de horas reales, y la anchura
mediana. Una banda mas estrecha que cubre lo mismo es una mejora sin trampa.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "10"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/banda_por_tarea")
S = "=" * 110
PLIEGUES, NIVEL, K_VEC, LOTE, N_GRUPOS = 3, .8, 25, 2048, 5
PAR = dict(max_iter=250, max_depth=6, learning_rate=.06, l2_regularization=1.,
           random_state=0)
CORTES_Y = [1, 2, 4, 8, 16]
ETIQ_Y = ["<=1", "1-2", "2-4", "4-8", "8-16", ">16"]


def qconf(x, p, arriba):
    s = np.sort(np.asarray(x, float))
    m = len(s)
    j = (int(np.ceil((m + 1) * p)) if arriba else int(np.floor((m + 1) * p))) - 1
    return float(s[int(np.clip(j, 0, m - 1))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy").astype(np.float32)
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    y = sw.effort_hours.to_numpy(float)
    ly = np.log(y)
    proy = sw.project.to_numpy()
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    est = sw.HoursEstimate.where(sw.HoursEstimate > 0).to_numpy(float)
    hay = ~np.isnan(est)
    part = list(GroupKFold(PLIEGUES).split(V, y, proy))
    a_ = (1 - NIVEL) / 2
    print(f"{len(y):,} tareas · con estimacion {hay.mean():.1%} · "
          f"{time.time() - t0:.0f} s", flush=True)

    # prediccion central, dispersion de vecinos y cuantiles aprendidos, fuera de pliegue
    p_med = np.zeros(len(y))
    q_lo = np.zeros(len(y))
    q_hi = np.zeros(len(y))
    disp = np.zeros(len(y))
    p_est = np.full(len(y), np.nan)
    Xe = np.hstack([V, np.log(np.where(hay, est, 1))[:, None]]).astype(np.float32)
    for nf, (a, b) in enumerate(part):
        p_med[b] = HistGradientBoostingRegressor(**PAR).fit(V[a], ly[a]).predict(V[b])
        q_lo[b] = HistGradientBoostingRegressor(
            **dict(PAR, loss="quantile", quantile=.1)).fit(V[a], ly[a]).predict(V[b])
        q_hi[b] = HistGradientBoostingRegressor(
            **dict(PAR, loss="quantile", quantile=.9)).fit(V[a], ly[a]).predict(V[b])
        for i in range(0, len(b), LOTE):
            bb = b[i:i + LOTE]
            sim = Vn[bb] @ Vn[a].T
            vec = np.argpartition(-sim, K_VEC, axis=1)[:, :K_VEC]
            disp[bb] = np.std(ly[a][vec], axis=1)
        ah, bh = a[hay[a]], b[hay[b]]
        p_est[bh] = HistGradientBoostingRegressor(**PAR).fit(
            Xe[ah], ly[ah]).predict(Xe[bh])
        print(f"  pliegue {nf + 1}/{PLIEGUES} · {time.time() - t0:.0f} s", flush=True)

    r = ly - p_med                                  # residuo en log
    fold = np.zeros(len(y), int)
    for nf, (_, b) in enumerate(part):
        fold[b] = nf

    def conforme_por_grupo(res, grupo):
        """Cuantiles conformes del residuo, por grupo, calibrados en OTROS pliegues."""
        lo, hi = np.zeros(len(y)), np.zeros(len(y))
        for nf in range(PLIEGUES):
            cal, ev = fold != nf, fold == nf
            for g in np.unique(grupo[ev]):
                cg = cal & (grupo == g)
                eg = ev & (grupo == g)
                lo[eg] = qconf(res[cg], a_, False)
                hi[eg] = qconf(res[cg], 1 - a_, True)
        return lo, hi

    def quintil(x):
        """Grupos por cuantil, con cortes calculados en TODO x (solo ordena)."""
        c = np.quantile(x, np.linspace(0, 1, N_GRUPOS + 1)[1:-1])
        return np.digitize(x, c)

    bandas = {}
    lo, hi = conforme_por_grupo(r, np.zeros(len(y), int))
    bandas["A · una sola anchura"] = (p_med + lo, p_med + hi)
    lo, hi = conforme_por_grupo(r, quintil(disp))
    bandas["B · por acuerdo de los vecinos"] = (p_med + lo, p_med + hi)
    # CQR: la puntuacion es cuanto se sale el real de [q_lo, q_hi]
    sc = np.maximum(q_lo - ly, ly - q_hi)
    cqr = np.zeros(len(y))
    for nf in range(PLIEGUES):
        cal, ev = fold != nf, fold == nf
        cqr[ev] = qconf(sc[cal], NIVEL, True)
    bandas["C · cuantiles aprendidos (CQR)"] = (q_lo - cqr, q_hi + cqr)
    lo, hi = conforme_por_grupo(r, quintil(p_med))
    bandas["D · por el valor predicho"] = (p_med + lo, p_med + hi)

    tr_y = np.digitize(y, CORTES_Y)
    filas = []
    for etq, (L, H) in bandas.items():
        dentro = (ly >= L) & (ly <= H)
        w = np.exp(H - L)
        d = dict(banda=etq, ambito="todas", cobertura=float(dentro.mean()),
                 anchura_mediana=float(np.median(w)),
                 anchura_p10=float(np.quantile(w, .1)),
                 anchura_p90=float(np.quantile(w, .9)))
        for g, e in enumerate(ETIQ_Y):
            d[f"cob_{e}"] = float(dentro[tr_y == g].mean())
        filas.append(d)

    # E: donde hay estimacion del usuario
    re_ = np.where(hay, ly - p_est, np.nan)
    loE, hiE = np.zeros(len(y)), np.zeros(len(y))
    for nf in range(PLIEGUES):
        cal, ev = (fold != nf) & hay, (fold == nf) & hay
        loE[ev] = qconf(re_[cal], a_, False)
        hiE[ev] = qconf(re_[cal], 1 - a_, True)
    # control del artefacto de E93: el 34,7 % de las filas con estimacion tienen
    # el real COPIADO de la estimacion. Se recalibra y se evalua SIN ellas.
    limpio = hay & (np.abs(np.where(hay, est, 1) - y) / y > .01)
    loL, hiL = np.zeros(len(y)), np.zeros(len(y))
    for nf in range(PLIEGUES):
        cal, ev = (fold != nf) & limpio, (fold == nf) & limpio
        loL[ev] = qconf(re_[cal], a_, False)
        hiL[ev] = qconf(re_[cal], 1 - a_, True)
    print(f"  filas con estimacion {int(hay.sum()):,} · limpias "
          f"{int(limpio.sum()):,}", flush=True)
    for etq, (L, H), s in (("A' · una sola anchura, mismas tareas",
                            bandas["A · una sola anchura"], hay),
                           ("E · con la estimacion del usuario",
                            (p_est + loE, p_est + hiE), hay),
                           ("A'' · una sola anchura, SOLO limpias",
                            bandas["A · una sola anchura"], limpio),
                           ("E' · con la estimacion, SOLO limpias",
                            (p_est + loL, p_est + hiL), limpio)):
        dentro = (ly[s] >= L[s]) & (ly[s] <= H[s])
        w = np.exp(H[s] - L[s])
        d = dict(banda=etq, ambito="limpias" if s is limpio else "con estimacion",
                 cobertura=float(dentro.mean()),
                 anchura_mediana=float(np.median(w)),
                 anchura_p10=float(np.quantile(w, .1)),
                 anchura_p90=float(np.quantile(w, .9)))
        for g, e in enumerate(ETIQ_Y):
            m_ = tr_y[s] == g
            d[f"cob_{e}"] = float(dentro[m_].mean()) if m_.any() else np.nan
        filas.append(d)

    t = pd.DataFrame(filas)
    print(f"\n{S}\nLA BANDA POR TAREA AL 80 %\n{S}")
    print(t[["banda", "ambito", "cobertura", "anchura_mediana", "anchura_p10",
             "anchura_p90"]].round(3).to_string(index=False))
    print(f"\n{S}\nCOBERTURA POR TRAMO DE HORAS REALES\n{S}")
    print(t[["banda"] + [f"cob_{e}" for e in ETIQ_Y]].round(3).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    a0 = t.iloc[0]
    for _, x in t.iloc[1:].iterrows():
        ref = (a0 if x.ambito == "todas" else t[t.banda.str.startswith("A''")].iloc[0] if x.ambito == "limpias" else t[t.banda.str.startswith("A' ")].iloc[0])
        print(f"  {x.banda:42s} x{x.anchura_mediana:6.2f} (cubre {x.cobertura:.1%})"
              f"  frente a x{ref.anchura_mediana:.2f}")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n": int(len(y)), "nivel": NIVEL, "k_vec": K_VEC},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
