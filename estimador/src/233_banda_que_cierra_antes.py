"""E114 - Que la banda del total se cierre antes: mirar COMO son las cerradas.

E107 dejo la banda del total calibrada por numero de tareas cerradas y tramo de
tamano, y baja bien: x10,61 en frio, x4,96 con 3, x3,75 con 5, x2,37 con 10 y
x2,05 con 20. Pero solo mira CUANTAS hay cerradas, no COMO son.

Y ahi hay senal que no se usa. Despues de cerrar k tareas se observa algo que
el frio no tiene: si esas k concuerdan con lo predicho -poca dispersion de
log(real/predicho)- el desplazamiento del proyecto esta bien determinado y lo
que queda abierto se puede acotar mas; si estan desperdigadas, no.

Es prediccion conforme NORMALIZADA, pero a nivel de proyecto, donde si hay un
observable que anticipa la incertidumbre. En la banda POR TAREA fracaso seis
veces porque el texto no sabe que tareas son inciertas; aqui el dato lo da el
propio avance del proyecto.

Rasgos disponibles en el momento de estimar, todos del proyecto en curso:

  k              tareas cerradas
  sd_cerradas    dispersion de log(real/predicho) en esas k
  |off|          cuanto corrige el anclaje
  n_abiertas     cuantas quedan por estimar
  frac_horas     que parte de las horas predichas ya esta cerrada
  tramo          tamano del proyecto

  A  la banda de E107: por k y tramo
  B  normalizada: el residuo se divide por sigma(rasgos), ajustado fuera de
     muestra, y el cuantil conforme se toma sobre el residuo normalizado
  C  B pero ademas por tramo

Lo que importa no es la anchura media sino cuanto se adelanta: a que k se llega
a la anchura que hoy exige 10 o 20 cerradas.
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

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
OUT = Path("results/banda_que_cierra_antes")
S = "=" * 112
MIN_TAREAS, LAMBDA, SORTEOS, NIVEL = 8, 12, 30, .8
KS = (1, 2, 3, 5, 10, 20)
CORTES = [20, 60, 200]
PAR = dict(max_iter=200, max_depth=4, learning_rate=.06, l2_regularization=1.,
           random_state=0)


def qconf(x, p, arriba):
    s = np.sort(np.asarray(x, float))
    m = len(s)
    j = (int(np.ceil((m + 1) * p)) if arriba else int(np.floor((m + 1) * p))) - 1
    return float(s[int(np.clip(j, 0, m - 1))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    tam = sw.groupby("project").size()
    m = sw.project.isin(tam[tam >= MIN_TAREAS].index).to_numpy()
    sw, pred = sw[m].reset_index(drop=True), pred[m]
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    nom = list(pd.unique(proy))
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    P = len(nom)
    n_t = np.array([len(idx[p]) for p in nom])
    tr = np.digitize(n_t, CORTES)
    print(f"{P} proyectos · {len(sw):,} tareas · {time.time() - t0:.0f} s",
          flush=True)

    # un registro por (k, sorteo, proyecto)
    reg = []
    for k in KS:
        for s in range(SORTEOS):
            rng = np.random.default_rng(1200 + s)
            for j, p in enumerate(nom):
                i = idx[p]
                kk = min(k, len(i) - 1)
                sel = rng.choice(i, kk, replace=False)
                rest = np.setdiff1d(i, sel)
                lr = np.log(y[sel] / pred[sel])
                off = float(np.log(y[sel].sum() / pred[sel].sum())
                            * kk / (kk + LAMBDA))
                ab = float((pred[rest] * np.exp(off)).sum())
                reg.append((k, s, j, tr[j], kk, len(rest),
                            float(np.std(lr)) if kk > 1 else -1.0,
                            abs(off), float(y[sel].sum()),
                            float(np.log(max(ab, 1e-6))),
                            float(pred[sel].sum() / pred[i].sum()),
                            float(np.log(y[rest].sum() / max(ab, 1e-9)))))
    R = pd.DataFrame(reg, columns=["k", "s", "j", "tramo", "kk", "n_ab", "sd_cer",
                                   "abs_off", "h_cer", "log_ab", "frac_h", "rho"])
    print(f"  {len(R):,} registros · {time.time() - t0:.0f} s", flush=True)

    RASGOS = ["kk", "n_ab", "sd_cer", "abs_off", "log_ab", "frac_h", "tramo"]
    X = R[RASGOS].to_numpy(np.float32)
    rho = R.rho.to_numpy()
    jj = R.j.to_numpy()

    # sigma(rasgos) fuera de muestra, dejando fuera PROYECTOS enteros
    grupo = jj % 4
    sig = np.zeros(len(R))
    for g in range(4):
        a, b = grupo != g, grupo == g
        mo = HistGradientBoostingRegressor(**PAR).fit(X[a], np.log(np.abs(rho[a]) + 1e-3))
        sig[b] = np.exp(mo.predict(X[b]))
    sig = np.clip(sig, 1e-3, None)
    print(f"  sigma ajustado · corr(|rho|, sigma) = "
          f"{np.corrcoef(np.abs(rho), sig)[0, 1]:.4f} · "
          f"{time.time() - t0:.0f} s", flush=True)

    filas, por_tramo = [], []

    def construye(etq, sk, nivel_nom):
        """Banda de un metodo a un nivel NOMINAL dado. Devuelve (dentro, anchura)."""
        aa = (1 - nivel_nom) / 2
        Rk = R[sk]
        jk, sgk, rk, trk = (Rk.j.to_numpy(), sig[sk], rho[sk], Rk.tramo.to_numpy())
        lo, hi = np.zeros(sk.sum()), np.zeros(sk.sum())
        for g in range(4):
            cal, ev = (jk % 4) != g, (jk % 4) == g
            if etq == "A":
                lo[ev], hi[ev] = qconf(rk[cal], aa, False), qconf(rk[cal], 1 - aa, True)
            elif etq == "A2":
                for t_ in np.unique(trk):
                    c2, e2 = cal & (trk == t_), ev & (trk == t_)
                    if c2.sum() < 60 or not e2.any():
                        c2 = cal
                    lo[e2] = qconf(rk[c2], aa, False)
                    hi[e2] = qconf(rk[c2], 1 - aa, True)
            elif etq == "B":
                z = rk[cal] / sgk[cal]
                lo[ev] = qconf(z, aa, False) * sgk[ev]
                hi[ev] = qconf(z, 1 - aa, True) * sgk[ev]
            elif etq == "C":
                for t_ in np.unique(trk):
                    c2, e2 = cal & (trk == t_), ev & (trk == t_)
                    if c2.sum() < 60 or not e2.any():
                        c2 = cal
                    z = rk[c2] / sgk[c2]
                    lo[e2] = qconf(z, aa, False) * sgk[e2]
                    hi[e2] = qconf(z, 1 - aa, True) * sgk[e2]
            else:
                # D: el cuantil del tramo CONTRAIDO hacia el global, con peso
                # n/(n+M). Conserva el caracter condicional de E89 sin pagar la
                # anchura que cuesta calibrar cada tramo por separado.
                M = float(etq[1:])
                lg, hg = qconf(rk[cal], aa, False), qconf(rk[cal], 1 - aa, True)
                for t_ in np.unique(trk):
                    c2, e2 = cal & (trk == t_), ev & (trk == t_)
                    if not e2.any():
                        continue
                    n2 = int(c2.sum())
                    if n2 < 10:
                        lo[e2], hi[e2] = lg, hg
                        continue
                    w_ = n2 / (n2 + M)
                    lo[e2] = w_ * qconf(rk[c2], aa, False) + (1 - w_) * lg
                    hi[e2] = w_ * qconf(rk[c2], 1 - aa, True) + (1 - w_) * hg
        ab, cer = np.exp(Rk.log_ab.to_numpy()), Rk.h_cer.to_numpy()
        w = (cer + ab * np.exp(hi)) / np.maximum(cer + ab * np.exp(lo), 1e-9)
        return (rk >= lo) & (rk <= hi), w

    NOMBRES = {"A": "A · solo por k", "A2": "A2 · por k y tramo (hoy)",
               "B": "B · normalizada", "C": "C · normalizada + tramo",
               "D200": "D · tramo contraido M=200", "D600": "D · tramo contraido M=600",
               "D1500": "D · tramo contraido M=1500"}
    for k in KS:
        sk = R.k.to_numpy() == k
        for etq in ("A", "A2", "B", "C", "D200", "D600", "D1500"):
            # se ajusta el nivel NOMINAL hasta que la cobertura REAL sea NIVEL,
            # para que la comparacion de anchuras sea a cobertura igual
            lo_n, hi_n = .60, .999
            for _ in range(12):
                md = (lo_n + hi_n) / 2
                d_, _ = construye(etq, sk, md)
                if d_.mean() < NIVEL:
                    lo_n = md
                else:
                    hi_n = md
            dentro, w = construye(etq, sk, hi_n)
            trk_ = R[sk].tramo.to_numpy()
            for t_ in np.unique(trk_):
                por_tramo.append(dict(k=k, banda=NOMBRES[etq], tramo=int(t_),
                                      proyectos=int(len(np.unique(R[sk].j.to_numpy()[trk_ == t_]))),
                                      cobertura=float(dentro[trk_ == t_].mean())))
            filas.append(dict(k=k, banda=NOMBRES[etq], nivel_nominal=float(hi_n),
                              cobertura=float(dentro.mean()),
                              anchura_mediana=float(np.median(w)),
                              anchura_p25=float(np.quantile(w, .25)),
                              anchura_p75=float(np.quantile(w, .75))))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)

    t = pd.DataFrame(filas)
    print(f"\n{S}\nANCHURA MEDIANA DE LA BANDA DEL TOTAL AL {NIVEL:.0%}\n{S}")
    print(t.pivot(index="k", columns="banda",
                  values="anchura_mediana").round(2).to_string())
    print(f"\n{S}\nCOBERTURA\n{S}")
    print(t.pivot(index="k", columns="banda", values="cobertura").round(3).to_string())
    print(f"\n{S}\nEL CUARTIL BUENO: PROYECTOS CUYAS CERRADAS CONCUERDAN\n{S}")
    print(t.pivot(index="k", columns="banda",
                  values="anchura_p25").round(2).to_string())

    print(f"\n{S}\nLECTURA\n{S}")
    A = t[t.banda.str.startswith("A2")].set_index("k")
    B = t[t.banda.str.startswith("B")].set_index("k")
    C = t[t.banda.str.startswith("C")].set_index("k")
    for k in KS:
        print(f"  k={k:2d}: hoy x{A.loc[k, 'anchura_mediana']:5.2f} · "
              f"B x{B.loc[k, 'anchura_mediana']:5.2f} · "
              f"C x{C.loc[k, 'anchura_mediana']:5.2f}  "
              f"(p25: {A.loc[k, 'anchura_p25']:5.2f} / "
              f"{B.loc[k, 'anchura_p25']:5.2f} / {C.loc[k, 'anchura_p25']:5.2f})")
    obj = float(A.loc[10, "anchura_mediana"])
    for etq, D in (("B", B), ("C", C)):
        alc = [k for k in KS if D.loc[k, "anchura_mediana"] <= obj]
        if alc:
            print(f"\n  {etq} alcanza con k={min(alc)} la anchura que A exige con k=10")
    t.to_csv(OUT / "resumen.csv", index=False)
    # cobertura de cada banda por tramo de tamano (guia V2: «cobertura en cada grupo»)
    pd.DataFrame(por_tramo).to_csv(OUT / "cobertura_por_tramo.csv", index=False)
    json.dump({"proyectos": P, "sorteos": SORTEOS, "nivel": NIVEL,
               "corr_sigma": float(np.corrcoef(np.abs(rho), sig)[0, 1])},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
