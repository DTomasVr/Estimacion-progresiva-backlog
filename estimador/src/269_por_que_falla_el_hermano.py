"""E-269 - ¿Por que empeora el total al usar el factor de otro proyecto del mismo equipo?

E-267: aplicar al proyecto B el factor aprendido en un hermano A baja el +-25 %
de 34,4 % a 26-29 %. Aqui se mide la causa.

b_p = log(total real / total en frio) es el desplazamiento de escala del proyecto.
Predecir b_B con w * b_A deja un error de varianza  s2 * (1 - 2 w rho + w^2),
donde rho es la correlacion de b entre hermanos. Solo mejora si w < 2 rho, y lo
optimo es w = rho. Se mide:

  1 rho: correlacion intraclase de b_p por organizacion (ANOVA de un factor);
  2 la curva real de +-25 % segun w, usando el hermano entero y solo k tareas;
  3 la version jerarquica: la media de TODOS los hermanos, contraida con el peso
    de credibilidad  Z = n rho / (1 + (n - 1) rho)  (Buhlmann-Straub / Jewell),
    dejando fuera el proyecto que se predice.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/por_que_falla_el_hermano")
WS = (0, .05, .1, .15, .2, .3, .5, .75, 1.0)
SORTEOS = 200

s = importlib.util.spec_from_file_location("e267", "src/267_entrada_del_usuario.py")
e267 = importlib.util.module_from_spec(s)
s.loader.exec_module(e267)


def icc(b, org):
    t = pd.DataFrame({"b": b, "o": org})
    g = t.groupby("o").b
    k, N = g.ngroups, len(t)
    n0 = (N - (g.size() ** 2).sum() / N) / (k - 1)
    msb = (g.size() * (g.mean() - t.b.mean()) ** 2).sum() / (k - 1)
    msw = ((t.b - g.transform("mean")) ** 2).sum() / (N - k)
    return float((msb - msw) / (msb + (n0 - 1) * msw)), float(msb), float(msw)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = e267.carga()
    G = [g for _, g in d.groupby("p", sort=False)]
    nom = [g.p.iloc[0] for g in G]
    re_ = [g.real.to_numpy() for g in G]
    es_ = [g.est.to_numpy() for g in G]
    real = np.array([x.sum() for x in re_])
    frio = np.array([x.sum() for x in es_])
    b = np.log(real / frio)
    org = np.array([e267.org_de(p) for p in nom])
    cuenta = pd.Series(org).value_counts()
    her = np.flatnonzero(pd.Series(org).map(cuenta).to_numpy() >= 2)
    P = len(nom)
    pred25 = lambda t, m: float(np.mean(np.abs(t[m] - real[m]) / real[m] <= .25))

    rho, msb, msw = icc(b[her], org[her])
    print(f"{P} proyectos · {len(her)} con hermano · b: media {b.mean():+.3f}, sd {b.std():.3f}")
    print(f"\n1 · CORRELACION ENTRE HERMANOS\n  intraclase rho = {rho:.3f}  (entre organizaciones {msb:.3f}, dentro {msw:.3f})")
    por_org = pd.DataFrame({"org": org[her], "b": b[her]}).groupby("org").b.agg(["size", "mean", "std"])
    print(por_org.round(3).to_string())
    print(f"  con rho = {rho:.3f}: usar el factor del hermano solo mejora si w < {2 * rho:.2f}; "
          f"lo optimo es w = {rho:.2f} y quita como mucho el {100 * rho ** 2:.1f} % de la varianza")
    print("  el anclaje desplegado usa w = k/(k+12): 0,29 con 5 tareas, 0,45 con 10, 0,63 con 20")

    rng = np.random.default_rng(269)
    filas = []
    for w in WS:
        acc = {"hermano entero": [], "hermano, 10 tareas": [], "otra organizacion entera": []}
        for _ in range(SORTEOS):
            t1, t2, t3 = frio.copy(), frio.copy(), frio.copy()
            for j in her:
                A = rng.choice(np.flatnonzero((org == org[j]) & (np.arange(P) != j)))
                t1[j] = frio[j] * np.exp(w * b[A])
                sel = rng.choice(len(re_[A]), min(10, len(re_[A])), replace=False)
                t2[j] = frio[j] * np.exp(w * np.log(re_[A][sel].sum() / es_[A][sel].sum()))
                C = rng.choice(np.flatnonzero(org != org[j]))
                t3[j] = frio[j] * np.exp(w * b[C])
            for nombre, t in zip(acc, (t1, t2, t3)):
                acc[nombre].append(pred25(t, her))
        filas.append(dict(w=w, **{k: float(np.mean(v)) for k, v in acc.items()}))
    t = pd.DataFrame(filas)
    t.to_csv(OUT / "curva_w.csv", index=False)
    print(f"\n2 · +-25 % DEL TOTAL DE B SEGUN EL PESO w ({len(her)} proyectos, en frio {pred25(frio, her):.3f})")
    print(t.round(3).to_string(index=False))

    filas = []
    for nombre, peso in (("media de los hermanos, sin contraer", lambda n: 1.0),
                         ("credibilidad Z = n rho / (1 + (n-1) rho)", lambda n: n * max(rho, 0) / (1 + (n - 1) * max(rho, 0))),
                         ("credibilidad con rho = 0,3 (si fuese mayor)", lambda n: n * .3 / (1 + (n - 1) * .3))):
        tt = frio.copy()
        for j in her:
            m = (org == org[j]) & (np.arange(P) != j)
            tt[j] = frio[j] * np.exp(peso(m.sum()) * b[m].mean())
        r = np.abs(tt[her] - real[her]) / real[her]
        filas.append(dict(brazo=nombre, PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                          factor2=float(np.mean(np.maximum(tt[her] / real[her], real[her] / tt[her]) <= 2))))
    r = np.abs(frio[her] - real[her]) / real[her]
    filas.insert(0, dict(brazo="en frio", PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                         factor2=float(np.mean(np.maximum(frio[her] / real[her], real[her] / frio[her]) <= 2))))
    t3 = pd.DataFrame(filas)
    t3.to_csv(OUT / "jerarquico.csv", index=False)
    print("\n3 · VERSION JERARQUICA: todos los hermanos enteros, dejando fuera el proyecto predicho")
    print(t3.round(3).to_string(index=False))
    json.dump({"rho": rho, "proyectos_con_hermano": int(len(her)), "sd_b": float(b.std()),
               "segundos": round(time.time() - t0)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
