"""E99 - La banda por tarea y el umbral de dispersion, despues de quitar el sesgo.

E98 dejo el centro de la prediccion donde debia estar: el factor de estilo ya no
se aplica al texto de persona y el sesgo mediano paso de x0,82 a x1,00. Pero hay
dos constantes del artefacto que se calibraron CUANDO el centro estaba corrido, y
nadie las ha vuelto a mirar:

  factor_banda80 = 4,3157   la banda por tarea es [h/k , h*k], simetrica en log
                            y centrada en h. Si h se ha movido un 20 %, la banda
                            ya no cubre lo que dice
  umbral_dispersion = 1,52  decide cuando el valor probable viene de la analogia
                            y cuando del apilado. Se eligio maximizando el +-25 %
                            cuando la rejilla se aplicaba a medias y el centro
                            estaba corrido

Las dos se re-estiman en VALIDACION y se miden en PRUEBA.

De paso se prueba algo que la banda simetrica no puede hacer: el residuo por
tarea no es simetrico en log -hay mas cola por arriba que por abajo- asi que una
banda con dos cortes distintos deberia cubrir lo mismo siendo mas estrecha. Se
usa el cuantil CONFORME, igual que en la banda de proyecto de E89.

  A  la banda de hoy, k=4,3157 simetrica
  B  k simetrica re-estimada en validacion
  C  banda asimetrica conforme, estimada en validacion
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
OUT = Path("results/banda_y_umbral")
S = "=" * 100
N_VAL, N_TEST, SEMILLA, NIVEL = 1500, 2500, 23, .8
UMBRALES = (0., .4, .6, .8, 1.0, 1.2, 1.5242808066335687, 1.8, 2.2, 3.0, 99.)


def qconf(x, p, arriba):
    s = np.sort(np.asarray(x, float))
    m = len(s)
    j = (int(np.ceil((m + 1) * p)) if arriba else int(np.floor((m + 1) * p))) - 1
    return float(s[int(np.clip(j, 0, m - 1))])


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
    est = m.EstimadorV6()
    k_hoy = float(est.p["factor_banda80"])
    u_hoy = float(est.p["umbral_dispersion"])
    rej = np.asarray(est.p["rejilla"], float)
    print(f"validacion {len(part['val'])} · prueba {len(part['test'])} · "
          f"k de hoy {k_hoy:.4f} · umbral de hoy {u_hoy:.4f} · "
          f"{time.time() - t0:.0f} s", flush=True)

    # se replica el interior de estimar_tareas para quedarse con ana y disp
    dat = {}
    for b, d in part.items():
        tx = d.title.fillna("").astype(str).tolist()
        V, Xd, SV, Ms = est._vistas(tx)
        aj = est._cod(est.p["codificador_ajustado"], "_aj").encode(
            ["query: " + t for t in tx], normalize_embeddings=False,
            show_progress_bar=False).astype(np.float32)
        ac = est._cod(est.p["codificador_ajustado_C"], "_ac").encode(
            ["query: " + t for t in tx], normalize_embeddings=False,
            show_progress_bar=False).astype(np.float32)
        Z = np.column_stack([np.mean([mm.predict(Xd) for mm in est.p["bolsa"]], 0),
                             est.p["ridge"].predict(Ms), est.p["svr"].predict(Ms),
                             est.p["hgb_ajustado"].predict(aj),
                             est.p["hgb_ajustado_C"].predict(ac)])
        h = np.exp(est.p["apilado"].predict(Z))        # estilo persona: sin factor
        ana, disp = est._analogia(V)
        dat[b] = dict(h=h, ana=ana, disp=disp,
                      y=d.effort_hours.to_numpy(float))
        print(f"  {b} listo · {time.time() - t0:.0f} s", flush=True)

    def ajusta(p):
        return rej[np.abs(np.log(rej)[None, :] -
                          np.log(np.clip(p, 1e-6, None))[:, None]).argmin(1)]

    # ---------------- la banda ----------------
    v, t_ = dat["val"], dat["test"]
    r_val = np.log(v["y"] / v["h"])
    a = (1 - NIVEL) / 2
    k_sim = float(np.exp(qconf(np.abs(r_val), NIVEL, True)))
    lo_as = qconf(r_val, a, False)
    hi_as = qconf(r_val, 1 - a, True)

    def mide_banda(lo, hi, etq):
        dentro = (t_["y"] >= lo) & (t_["y"] <= hi)
        return dict(banda=etq, cobertura=float(dentro.mean()),
                    anchura_x=float(np.mean(hi / np.maximum(lo, 1e-9))),
                    anchura_mediana_x=float(np.median(hi / np.maximum(lo, 1e-9))))

    h = t_["h"]
    filas = [mide_banda(h / k_hoy, h * k_hoy, f"A · hoy, simetrica k={k_hoy:.3f}"),
             mide_banda(h / k_sim, h * k_sim,
                        f"B · simetrica re-estimada k={k_sim:.3f}"),
             mide_banda(h * np.exp(lo_as), h * np.exp(hi_as),
                        f"C · asimetrica conforme [x{np.exp(lo_as):.2f}, "
                        f"x{np.exp(hi_as):.2f}]")]
    tb = pd.DataFrame(filas)
    print(f"\n{S}\nLA BANDA POR TAREA AL {NIVEL:.0%} (objetivo de cobertura "
          f"{NIVEL:.2f})\n{S}")
    print(tb.round(4).to_string(index=False))

    # ---------------- el umbral de dispersion ----------------
    def vp(d, u):
        return ajusta(np.where(d["disp"] <= u, d["ana"], d["h"]))

    def p25(y, p):
        return float(np.mean(np.abs(p - y) / y <= .25))

    print(f"\n{S}\nEL UMBRAL DE DISPERSION\n{S}")
    print(f"  {'umbral':>10}{'% por analogia':>16}{'±25 % val':>12}"
          f"{'±25 % prueba':>14}{'factor2 prueba':>16}")
    fil_u = []
    for u in UMBRALES:
        pv, pt = vp(v, u), vp(t_, u)
        f2 = float(np.mean(np.maximum(pt / t_["y"], t_["y"] / pt) <= 2))
        fil_u.append(dict(umbral=u, frac_analogia=float((t_["disp"] <= u).mean()),
                          p25_val=p25(v["y"], pv), p25_test=p25(t_["y"], pt),
                          factor2_test=f2))
        marca = "  <- el de hoy" if abs(u - u_hoy) < 1e-6 else ""
        print(f"  {u:>10.2f}{(t_['disp'] <= u).mean():>15.1%}"
              f"{p25(v['y'], pv):>12.1%}{p25(t_['y'], pt):>14.1%}{f2:>16.1%}{marca}")
    tu = pd.DataFrame(fil_u)
    mejor = tu.loc[tu.p25_val.idxmax()]        # se ELIGE por validacion
    hoy = tu.loc[(tu.umbral - u_hoy).abs().idxmin()]
    print(f"\n  elegido por validacion: umbral {mejor.umbral:.2f} · "
          f"en prueba {mejor.p25_test:.1%}")
    print(f"  el de hoy ({u_hoy:.2f}): en prueba {hoy.p25_test:.1%}")

    print(f"\n{S}\nLECTURA\n{S}")
    A, B, C = filas
    print(f"  banda: hoy cubre {A['cobertura']:.1%} con anchura "
          f"x{A['anchura_mediana_x']:.2f}, cuando promete {NIVEL:.0%}")
    print(f"         asimetrica cubre {C['cobertura']:.1%} con anchura "
          f"x{C['anchura_mediana_x']:.2f}")
    if abs(C["cobertura"] - NIVEL) < abs(A["cobertura"] - NIVEL):
        print("         la asimetrica cumple mejor; conviene cambiarla")
    if mejor.p25_test - hoy.p25_test > .003:
        print(f"  umbral: mover de {u_hoy:.2f} a {mejor.umbral:.2f} da "
              f"{(mejor.p25_test - hoy.p25_test) * 100:+.1f} puntos")
    else:
        print(f"  umbral: el de hoy esta bien, no se toca "
              f"({(mejor.p25_test - hoy.p25_test) * 100:+.1f} puntos)")
    tb.to_csv(OUT / "banda.csv", index=False)
    tu.to_csv(OUT / "umbral.csv", index=False)
    json.dump({"k_hoy": k_hoy, "k_simetrica": k_sim,
               "asimetrica_lo": float(np.exp(lo_as)),
               "asimetrica_hi": float(np.exp(hi_as)),
               "umbral_hoy": u_hoy, "umbral_elegido": float(mejor.umbral),
               "cobertura_hoy": A["cobertura"], "cobertura_asim": C["cobertura"]},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/ · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
