"""E-268 - La regla completa: cada tarea esta CERRADA, ESTIMADA por el usuario o SOLA.

E-267 midio por separado dar horas reales y dar estimaciones. En el producto
conviven: unas tareas ya se cerraron (horas reales), otras tienen la estimacion
del usuario y el resto solo tiene el modelo. La regla que se va a implementar:

  base de una tarea   = estimacion del usuario si la hay; si no, modelo x f_u
  f_u                 = (suma usuario / suma modelo, en las estimadas) ^ (m / (m + lambda_u))
  f_c                 = (suma real / suma base, en las cerradas)       ^ (k / (k + lambda_c))
  total               = suma real de las cerradas + f_c x suma base de las abiertas

Con m = 0 y k = 0 es el frio; con una sola entrada solo pesa esa. Se mide en los
proyectos que traen estimacion real del usuario (80 % de sus tareas o mas), para
todas las combinaciones de cuanto estima el usuario y cuantas tareas hay cerradas.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/tres_estados")
SORTEOS = 60
ESTIMA = (0, .25, .50, 1.0)
CERRADAS = (0, 3, 5, 10, 20)
LAM_U = (12, 30)
LAM_C = (12, 30, 96)

s = importlib.util.spec_from_file_location("e267", "src/267_entrada_del_usuario.py")
e267 = importlib.util.module_from_spec(s)
s.loader.exec_module(e267)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    d = e267.carga()
    G = [g for _, g in d.groupby("p", sort=False)]
    G = [g for g in G if g.usu.notna().mean() >= e267.COBERTURA]
    P = len(G)
    re_ = [g.real.to_numpy() for g in G]
    es_ = [g.est.to_numpy() for g in G]
    us_ = [g.usu.to_numpy() for g in G]
    con = [np.flatnonzero(~np.isnan(x)) for x in us_]
    n = np.array([len(x) for x in re_])
    real = np.array([x.sum() for x in re_])
    limpio = np.array([g.copiada.mean() < .25 for g in G])
    print(f"{P} proyectos con estimacion del usuario · {int(limpio.sum())} con pocas filas copiadas · "
          f"tareas por proyecto: mediana {int(np.median(n))}", flush=True)

    def total(j, est_idx, cer_idx, lu, lc):
        base = es_[j].copy()
        m = len(est_idx)
        if m:
            fu = np.exp(np.log(us_[j][est_idx].sum() / es_[j][est_idx].sum()) * m / (m + lu))
            base = base * fu
            base[est_idx] = us_[j][est_idx]
        k = len(cer_idx)
        if not k:
            return base.sum()
        ab = np.ones(n[j], bool)
        ab[cer_idx] = False
        fc = np.exp(np.log(re_[j][cer_idx].sum() / base[cer_idx].sum()) * k / (k + lc))
        return re_[j][cer_idx].sum() + fc * base[ab].sum()

    filas = []
    for fe in ESTIMA:
        for k in CERRADAS:
            rng = np.random.default_rng(26800 + int(fe * 100) * 100 + k)
            acc = {}
            for _ in range(SORTEOS):
                sel = []
                for j in range(P):
                    m = min(int(round(fe * n[j])), len(con[j]))
                    ei = rng.choice(con[j], m, replace=False) if m else np.array([], int)
                    kk = min(k, n[j] - 1)
                    ci = rng.choice(n[j], kk, replace=False) if kk else np.array([], int)
                    sel.append((ei, ci))
                for lu in LAM_U:
                    for lc in LAM_C:
                        if (fe == 0 and lu != LAM_U[0]) or (k == 0 and lc != LAM_C[0]):
                            continue
                        t = np.array([total(j, *sel[j], lu, lc) for j in range(P)])
                        r = np.abs(t - real) / real
                        acc.setdefault((lu, lc), []).append(
                            (np.mean(r <= .25), np.median(r), np.mean(r[limpio] <= .25),
                             np.mean(np.maximum(t / real, real / t) <= 2)))
            for (lu, lc), v in acc.items():
                v = np.mean(v, 0)
                filas.append(dict(usuario_estima=f"{int(fe * 100)} %", cerradas=k, lambda_u=lu, lambda_c=lc,
                                  PRED25=float(v[0]), MdMRE=float(v[1]), PRED25_limpios=float(v[2]),
                                  factor2=float(v[3])))
        print(f"  usuario estima {int(fe * 100)} % listo · {time.time() - t0:.0f} s", flush=True)
    t = pd.DataFrame(filas)
    t.to_csv(OUT / "resumen.csv", index=False)
    mejor = t.sort_values("PRED25", ascending=False).groupby(["usuario_estima", "cerradas"], sort=False).head(1)
    orden = [f"{int(f * 100)} %" for f in ESTIMA]
    for col, titulo in (("PRED25", f"PRED25 del total, los {P} proyectos"),
                        ("PRED25_limpios", f"PRED25 del total, los {int(limpio.sum())} con pocas filas copiadas"),
                        ("MdMRE", "MdMRE del total")):
        print(f"\n{titulo} (mejor lambda de cada celda; filas: cuanto estima el usuario · columnas: tareas cerradas)")
        print(mejor.pivot(index="usuario_estima", columns="cerradas", values=col).reindex(orden).round(3).to_string())
    fijo = t[(t.lambda_u == 12) & (t.lambda_c == 30) | (t.usuario_estima == "0 %") & (t.lambda_c == 30)
             | (t.cerradas == 0) & (t.lambda_u == 12)]
    fijo = fijo.drop_duplicates(["usuario_estima", "cerradas"])
    print("\nPRED25 con UNA sola configuracion (lambda_u = 12, lambda_c = 30), que es lo que se desplegaria:")
    print(fijo.pivot(index="usuario_estima", columns="cerradas", values="PRED25").reindex(orden).round(3).to_string())
    print("\nlambdas que ganan en cada celda:")
    print(mejor.assign(l=mejor.lambda_u.astype(str) + "/" + mejor.lambda_c.astype(str))
          .pivot(index="usuario_estima", columns="cerradas", values="l").reindex(orden).to_string())
    json.dump({"proyectos": P, "limpios": int(limpio.sum()), "sorteos": SORTEOS,
               "segundos": round(time.time() - t0)}, open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
