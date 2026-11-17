"""E98 - Centrar lo que de verdad sale del artefacto.

E97 destapo, mirando tarea a tarea, que lo que devuelve el .joblib no es lo que
dicen las metricas. Sobre 400 tareas reales del bloque de prueba que el
artefacto no vio:

    sesgo mediano de las horas, tal como se despliega   x0,686
    quitando el factor de estilo                        x0,825
    lo justo                                            1,0

O sea que el artefacto ensena un 31 % menos de horas de las reales. Dos causas
distintas y las dos arreglables:

  1  el FACTOR DE ESTILO 0,8313 se aplica siempre. Se midio en E74 para el
     texto que escribe el generador, que el modelo sobreestima x1,38. Sobre
     texto de persona el sesgo medido era x1,002, o sea ninguno: ahi el factor
     solo encoge
  2  aun sin el factor queda un x0,825. Eso es del propio apilado y nunca se
     habia mirado, porque las metricas del proyecto se calculan sobre
     predicciones limpias de experimento, no sobre la salida del paquete

El centrado se estima en el bloque de VALIDACION y se aplica al de PRUEBA, que
es la unica forma honesta de medirlo.

  A  como se despliega hoy
  B  sin el factor de estilo
  C  B + centrado ajustado en validacion
  D  C + ajuste a la rejilla (lo que ya entra por E96)

El factor de estilo no se borra: pasa a ser CONDICIONAL al origen del texto, que
es lo que siempre debio ser. El adaptador del generador lo seguira aplicando.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
OUT = Path("results/centrar_artefacto")
S = "=" * 104
N_VAL, N_TEST, SEMILLA = 1200, 2000, 11


def met(y, p, etq):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    mar = np.abs(np.median(y) - y).mean()
    return dict(salida=etq, PRED25=float(np.mean(r <= .25)),
                MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                MAE_h=float(np.abs(p - y).mean()),
                SA=float(1 - np.abs(p - y).mean() / mar),
                sesgo=float(np.median(p / y)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import importlib.util
    spec = importlib.util.spec_from_file_location("e6", "src/171_estimador_v6.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")]
    rng = np.random.default_rng(SEMILLA)
    part = {}
    for b, n in (("val", N_VAL), ("test", N_TEST)):
        d = sw[sw.split_t3 == b].reset_index(drop=True)
        i = rng.choice(len(d), min(n, len(d)), replace=False)
        part[b] = d.iloc[i].reset_index(drop=True)
    print(f"validacion {len(part['val'])} · prueba {len(part['test'])} tareas "
          f"reales · {time.time() - t0:.0f} s", flush=True)

    est = m.EstimadorV6()
    fe = float(est.p["factor_estilo"])
    rej = np.asarray(est.p["rejilla"], float)
    salida = {}
    for b, d in part.items():
        r = est.estimar_tareas(d.title.fillna("").astype(str).tolist())
        salida[b] = (r.horas.to_numpy(float), d.effort_hours.to_numpy(float))
        print(f"  {b} estimado · {time.time() - t0:.0f} s", flush=True)

    hv, yv = salida["val"]
    ht, yt = salida["test"]
    # el centrado se ESTIMA en validacion, sobre las horas ya sin factor de estilo
    centro = float(np.exp(np.median(np.log(yv / (hv / fe)))))
    print(f"\n  factor de estilo del artefacto: {fe:.4f}")
    print(f"  centrado estimado EN VALIDACION: x{centro:.4f}")

    def ajusta(p):
        return rej[np.abs(np.log(rej)[None, :] -
                          np.log(np.clip(p, 1e-6, None))[:, None]).argmin(1)]

    sin = ht / fe
    cen = sin * centro
    brazos = [met(yt, ht, "A · como se despliega hoy"),
              met(yt, sin, "B · sin el factor de estilo"),
              met(yt, cen, "C · B + centrado (de validacion)"),
              met(yt, ajusta(cen), "D · C + ajustado a la rejilla"),
              met(yt, ajusta(sin), "E · B + rejilla, SIN centrar"),
              met(yt, ajusta(ht), "  (referencia: hoy + rejilla)")]
    t = pd.DataFrame(brazos)
    print(f"\n{S}\nSOBRE {len(yt):,} TAREAS REALES DEL BLOQUE DE PRUEBA\n{S}")
    print(t.round(4).to_string(index=False))

    print(f"\n{S}\nPOR TRAMO: SESGO ANTES Y DESPUES\n{S}")
    cortes, etq = [1, 2, 4, 8, 16], ["<=1", "1-2", "2-4", "4-8", "8-16", ">16"]
    g = np.digitize(yt, cortes)
    fin = ajusta(cen)
    print(f"  {'tramo':>6}{'n':>5}{'real med':>10}{'hoy':>9}{'corregido':>11}"
          f"{'sesgo hoy':>11}{'sesgo corr':>12}")
    for i, e in enumerate(etq):
        s = g == i
        if not s.any():
            continue
        print(f"  {e:>6}{int(s.sum()):>5}{np.median(yt[s]):>10.2f}"
              f"{np.median(ht[s]):>9.2f}{np.median(fin[s]):>11.2f}"
              f"{np.median(ht[s] / yt[s]):>11.2f}{np.median(fin[s] / yt[s]):>12.2f}")

    print(f"\n{S}\nLECTURA\n{S}")
    a, d_ = t.iloc[0], t.loc[t.factor2.idxmax()]
    print(f"  ±25 %    {a.PRED25:.1%} -> {d_.PRED25:.1%}  "
          f"({(d_.PRED25 - a.PRED25) * 100:+.1f} puntos)")
    print(f"  factor 2 {a.factor2:.1%} -> {d_.factor2:.1%}  "
          f"({(d_.factor2 - a.factor2) * 100:+.1f})")
    print(f"  MAE      {a.MAE_h:.2f} h -> {d_.MAE_h:.2f} h")
    print(f"  SA       {a.SA:.4f} -> {d_.SA:.4f}")
    print(f"  sesgo    x{a.sesgo:.3f} -> x{d_.sesgo:.3f}")
    print(f"\n  el factor de estilo que el adaptador del generador debe seguir")
    print(f"  aplicando por su cuenta: {fe:.4f} para la mediana")
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"n_val": len(yv), "n_test": len(yt), "factor_estilo": fe,
               "centrado": centro,
               "pred25_hoy": float(a.PRED25), "pred25_corregido": float(d_.PRED25),
               "sesgo_hoy": float(a.sesgo), "sesgo_corregido": float(d_.sesgo)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
