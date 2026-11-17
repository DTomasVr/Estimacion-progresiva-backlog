"""E100 - El desarrollador: la pieza que faltaba del 89 %.

E80 descompuso el residuo del modelo: 11 % ENTRE proyectos y 89 % DENTRO. El
anclaje por proyecto recoge ese 11 % y por eso funciona. El 89 % restante se
dio por ruido irreducible, y ocho vias distintas se estrellaron contra el.

Pero nunca se miro quien HACE la tarea. Sousa et al. (IEEE Access, 2023), que
es el aval directo de este enfoque, usa 'assigned_to' entre sus variables de
entrada. Y nuestro corpus la trae: el 87,2 % de las tareas tienen desarrollador,
1.156 distintos, y entre los 159 con 20 tareas o mas la mediana de horas va de
1,00 h (p10) a 6,00 h (p90), con una dispersion en log de 0,711.

Eso es del mismo orden que la dispersion de escala ENTRE proyectos (0,791), que
es justo la que el anclaje recupera. Si una parte del 89 % es en realidad ENTRE
DESARROLLADORES dentro del proyecto, hay una segunda ancla que nadie ha usado.

Y es desplegable: en PBE la tarea se asigna, y el panel ya reparte por persona.

  1  DESCOMPOSICION   del residuo en tres niveles: entre proyectos, entre
     desarrolladores dentro del proyecto, y dentro del desarrollador
  2  EL ANCLAJE       igual que el de proyecto pero por persona, con su propia
     contraccion k/(k+lambda), y combinado con el de proyecto
  3  LA COTA          que daria conocer la mediana real del desarrollador

  A  en frio
  B  anclaje por proyecto                       lo actual
  C  anclaje por desarrollador
  D  los dos: proyecto y despues desarrollador
  E  oraculo: la mediana real del desarrollador

Protocolo: las tareas del desarrollador que se usan para anclar son OTRAS, nunca
la que se evalua, y se sortea cuales con la metrica promediada entre sorteos.
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
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
OUT = Path("results/anclaje_desarrollador")
S = "=" * 100
LAM_P, SORTEOS = 12, 40
LAM_D = (4, 8, 12, 24)
KS = (1, 3, 5, 10)


def met(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    mar = np.abs(np.median(y) - y).mean()
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                MAE_h=float(np.abs(p - y).mean()),
                SA=float(1 - np.abs(p - y).mean() / mar),
                spearman=float(spearmanr(p, y).statistic))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    sw = todo[~todo.fuente_jira.eq("Apache")].reset_index(drop=True)
    pred = np.clip(np.load("results/experto_cortas/predicciones.npz")["p_glob"],
                   1e-6, None)
    assert len(pred) == len(sw)
    hay = sw.desarrollador.notna().to_numpy()
    sw, pred = sw[hay].reset_index(drop=True), pred[hay]
    y = sw.effort_hours.to_numpy(float)
    proy = sw.project.to_numpy()
    dev = sw.desarrollador.astype(str).to_numpy()
    # el desarrollador solo sirve si tiene historial: al menos 5 tareas
    cnt = pd.Series(dev).value_counts()
    ok = np.isin(dev, cnt[cnt >= 5].index.to_numpy())
    sw, pred, y, proy, dev = (sw[ok].reset_index(drop=True), pred[ok], y[ok],
                              proy[ok], dev[ok])
    rho = np.log(y / pred)
    print(f"{len(y):,} tareas con desarrollador de >=5 tareas · "
          f"{len(np.unique(dev)):,} desarrolladores · {len(np.unique(proy))} "
          f"proyectos · {time.time() - t0:.0f} s", flush=True)

    # ---------- 1. descomposicion en tres niveles ----------
    g = pd.DataFrame(dict(rho=rho, p=proy, d=dev))
    gran = g.rho.mean()
    mp = g.groupby("p").rho.mean()
    np_ = g.groupby("p").size()
    entre_p = float(np.average((mp - gran) ** 2, weights=np_))
    # dentro del proyecto, cuanto hay entre desarrolladores
    g["res_p"] = g.rho - g.p.map(mp)
    md = g.groupby(["p", "d"]).res_p.mean()
    nd = g.groupby(["p", "d"]).size()
    entre_d = float(np.average(md ** 2, weights=nd))
    dentro = float(g.groupby(["p", "d"]).res_p.var().fillna(0)
                   .mul(nd).sum() / nd.sum())
    tot = entre_p + entre_d + dentro
    print(f"\n{S}\n¿DE QUIEN ES EL RESIDUO?\n{S}")
    print(f"  entre PROYECTOS                       {entre_p:.4f}  "
          f"{entre_p / tot:6.1%}")
    print(f"  entre DESARROLLADORES del mismo proy. {entre_d:.4f}  "
          f"{entre_d / tot:6.1%}")
    print(f"  dentro del desarrollador              {dentro:.4f}  "
          f"{dentro / tot:6.1%}")

    i_p = {p: np.flatnonzero(proy == p) for p in np.unique(proy)}
    i_d = {d: np.flatnonzero(dev == d) for d in np.unique(dev)}

    def ancla(base, idx_map, clave, k, lam, rng):
        """Desplazamiento por grupo, estimado con k tareas OTRAS del grupo."""
        off = np.zeros(len(y))
        for g_, ii in idx_map.items():
            if len(ii) <= k:
                continue
            sel = rng.choice(ii, k, replace=False)
            o = np.log(y[sel].sum() / base[sel].sum()) * k / (k + lam)
            m = np.ones(len(ii), bool)
            m[np.searchsorted(ii, sel)] = False
            off[ii[m]] = o
        return off

    filas = []
    filas.append(dict(brazo="A · en frio", k=0, lam_d=0, **met(y, pred)))
    for k in KS:
        # el ancla de proyecto no depende de lambda_d: se calcula UNA vez por
        # sorteo y se reutiliza en los cuatro lambda, en vez de repetirla 4 veces
        ops = [ancla(pred, i_p, proy, k, LAM_P, np.random.default_rng(3000 + s))
               for s in range(SORTEOS)]
        mb = [met(y, pred * np.exp(o)) for o in ops]
        filas.append(dict(brazo="B · solo proyecto", k=k, lam_d=0,
                          **{c: float(np.mean([x[c] for x in mb]))
                             for c in mb[0]}))
        for lam_d in LAM_D:
            ods = [ancla(pred, i_d, dev, k, lam_d, np.random.default_rng(4000 + s))
                   for s in range(SORTEOS)]
            ms_c = [met(y, pred * np.exp(od)) for od in ods]
            ms_d = [met(y, pred * np.exp(op + od)) for op, od in zip(ops, ods)]
            filas.append(dict(brazo="C · solo desarrollador", k=k, lam_d=lam_d,
                              **{c: float(np.mean([x[c] for x in ms_c]))
                                 for c in ms_c[0]}))
            filas.append(dict(brazo="D · proyecto + desarrollador", k=k,
                              lam_d=lam_d,
                              **{c: float(np.mean([x[c] for x in ms_d]))
                                 for c in ms_d[0]}))
        print(f"  k={k} listo · {time.time() - t0:.0f} s", flush=True)

    # E: oraculo, la mediana real del desarrollador (dejando fuera la tarea)
    off_o = np.zeros(len(y))
    for d_, ii in i_d.items():
        for j in ii:
            otras = ii[ii != j]
            off_o[j] = np.log(y[otras].sum() / pred[otras].sum())
    filas.append(dict(brazo="E · oraculo del desarrollador", k=-1, lam_d=0,
                      **met(y, pred * np.exp(off_o))))

    t = pd.DataFrame(filas)
    cols = ["brazo", "k", "lam_d", "PRED25", "MdMRE", "factor2", "MAE_h", "SA",
            "spearman"]
    print(f"\n{S}\nANCLAR POR PERSONA ({len(y):,} tareas)\n{S}")
    print(t[cols].round(4).to_string(index=False))

    print(f"\n{S}\nLECTURA\n{S}")
    a = float(t[t.brazo.str.startswith("A")].PRED25.iloc[0])
    o = float(t[t.brazo.str.startswith("E")].PRED25.iloc[0])
    b5 = t[(t.brazo.str.startswith("B")) & (t.k == 5)]
    mejor = t[t.k > 0].loc[t[t.k > 0].PRED25.idxmax()]
    print(f"  en frio {a:.1%} · oraculo del desarrollador {o:.1%} "
          f"({(o - a) * 100:+.1f} pts)")
    if len(b5):
        print(f"  solo proyecto, k=5: {float(b5.PRED25.iloc[0]):.1%}")
    print(f"  LO MEJOR: {mejor.brazo} con k={int(mejor.k)}, lambda_d="
          f"{int(mejor.lam_d)} · {mejor.PRED25:.1%} "
          f"({(mejor.PRED25 - a) * 100:+.1f} pts sobre el frio)")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n": int(len(y)), "entre_proyectos": entre_p,
               "entre_desarrolladores": entre_d, "dentro": dentro,
               "frio": a, "oraculo_dev": o,
               "mejor": {k: (float(v) if isinstance(v, (int, float, np.floating))
                             else str(v)) for k, v in mejor.items()}},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
