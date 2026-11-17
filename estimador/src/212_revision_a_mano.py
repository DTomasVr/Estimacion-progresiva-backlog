"""E97 - Revision a mano: que entra al artefacto, que sale, y que deberia salir.

Las metricas agregadas se calculan sobre predicciones LIMPIAS hechas en
experimentos (un HGB fuera de pliegue). El artefacto que se despliega no es eso:
pasa por el apilado de once canales, le aplica un factor de estilo, y para el
valor probable mezcla la analogia con la rejilla. Nadie ha comprobado tarea a
tarea que lo que SALE del artefacto es lo que las metricas dicen.

Hay una sospecha concreta. En estimar_tareas:

    h = exp( apilado.predict(Z) + log(factor_estilo) )

y factor_estilo = 0,8313 se aplica SIEMPRE. Ese numero se midio en E74 como la
correccion para el texto que escribe el GENERADOR, que el modelo sobreestima
x1,38. Para el texto real de una persona el sesgo medido era x1,002, o sea
ninguno. Si el factor se aplica tambien al texto real, el artefacto encoge un
17 % todas las predicciones sin motivo.

Se comprueba sobre tareas REALES del bloque de prueba, que el artefacto no vio
al construirse. Se imprime la tabla entrada -> salida -> real para leerla a
mano, y se mide el sesgo de las dos salidas (horas y valor probable) con el
factor y sin el.
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
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 58)
OUT = Path("results/revision_a_mano")
S = "=" * 110
N_MUESTRA = 400
SEMILLA = 7


def met(y, p, etq):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    return dict(salida=etq, PRED25=float(np.mean(r <= .25)),
                MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                MAE_h=float(np.abs(p - y).mean()),
                sesgo_mediano=float(np.median(p / y)))


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
    te = sw[sw.split_t3 == "test"].reset_index(drop=True)
    rng = np.random.default_rng(SEMILLA)
    sel = rng.choice(len(te), min(N_MUESTRA, len(te)), replace=False)
    d = te.iloc[sel].reset_index(drop=True)
    tit = d.title.fillna("").astype(str).tolist()
    y = d.effort_hours.to_numpy(float)
    print(f"{len(d)} tareas reales del bloque de prueba · "
          f"el artefacto no las vio · {time.time() - t0:.0f} s", flush=True)

    est = m.EstimadorV6()
    fe = float(est.p["factor_estilo"])
    r = est.estimar_tareas(tit)
    h = r.horas.to_numpy(float)
    vp = r.valor_probable.to_numpy(float)
    print(f"  factor_estilo del artefacto: {fe:.4f} "
          f"(se aplica SIEMPRE) · {time.time() - t0:.0f} s", flush=True)

    # las mismas salidas sin el factor
    h_sin = h / fe
    rej = est.p["rejilla"]
    vp_sin = rej[np.abs(np.log(rej)[None, :] -
                        np.log(np.clip(vp / fe, 1e-6, None))[:, None]).argmin(1)]

    filas = [met(y, h, "horas (con factor)"), met(y, h_sin, "horas SIN factor"),
             met(y, vp, "valor probable (con factor)"),
             met(y, vp_sin, "valor probable SIN factor")]
    t = pd.DataFrame(filas)
    print(f"\n{S}\nLO QUE SALE DEL ARTEFACTO, SOBRE TEXTO REAL\n{S}")
    print(t.round(4).to_string(index=False))

    print(f"\n{S}\n30 TAREAS, A MANO (ordenadas por horas reales)\n{S}")
    tab = pd.DataFrame({"titulo": [x[:56] for x in tit], "real": y,
                        "horas": h, "probable": vp, "sin factor": h_sin})
    tab["cociente"] = tab.probable / tab.real
    o = np.argsort(y)
    muestra = tab.iloc[o[np.linspace(0, len(o) - 1, 30).astype(int)]]
    print(muestra.round(2).to_string(index=False))

    print(f"\n{S}\nPOR TRAMO DE HORAS REALES\n{S}")
    cortes = [0, 1, 2, 4, 8, 16, 1e9]
    etq = ["<=1", "1-2", "2-4", "4-8", "8-16", ">16"]
    g = np.digitize(y, cortes[1:-1])
    print(f"  {'tramo':>6}{'n':>5}{'real med':>10}{'horas':>9}{'probable':>10}"
          f"{'sesgo h':>9}{'sesgo vp':>10}")
    for i, e in enumerate(etq):
        s = g == i
        if not s.any():
            continue
        print(f"  {e:>6}{int(s.sum()):>5}{np.median(y[s]):>10.2f}"
              f"{np.median(h[s]):>9.2f}{np.median(vp[s]):>10.2f}"
              f"{np.median(h[s] / y[s]):>9.2f}{np.median(vp[s] / y[s]):>10.2f}")

    print(f"\n{S}\nLECTURA\n{S}")
    con = float(t[t.salida == "horas (con factor)"].sesgo_mediano.iloc[0])
    sin = float(t[t.salida == "horas SIN factor"].sesgo_mediano.iloc[0])
    print(f"  sesgo mediano de las horas: con factor x{con:.3f} · "
          f"sin factor x{sin:.3f}  (1,0 es lo justo)")
    p_con = float(t[t.salida == "valor probable (con factor)"].PRED25.iloc[0])
    p_sin = float(t[t.salida == "valor probable SIN factor"].PRED25.iloc[0])
    print(f"  ±25 % del valor probable: con factor {p_con:.1%} · "
          f"sin factor {p_sin:.1%}")
    if abs(np.log(sin)) < abs(np.log(con)) and p_sin >= p_con:
        print("\n  EL FACTOR DE ESTILO NO DEBE APLICARSE AL TEXTO REAL. Se midio")
        print("  para el texto del generador; sobre texto de persona encoge la")
        print("  prediccion sin motivo. Tiene que ser condicional al origen.")
    elif abs(np.log(sin)) < abs(np.log(con)):
        print("\n  El factor sesga el centro pero el ±25 % no mejora al quitarlo;")
        print("  hay que mirarlo junto a la rejilla antes de tocar nada.")
    else:
        print("\n  El factor no perjudica sobre texto real; dejarlo como esta.")
    t.to_csv(OUT / "salidas.csv", index=False)
    tab.to_csv(OUT / "tarea_a_tarea.csv", index=False)
    json.dump({"n": int(len(d)), "factor_estilo": fe,
               "sesgo_con": con, "sesgo_sin": sin,
               "pred25_vp_con": p_con, "pred25_vp_sin": p_sin},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/ · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
