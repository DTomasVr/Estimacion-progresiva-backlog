"""E79 - Usar el TEXTO de las tareas historicas, no solo sus horas.

Hoy el historial entra como un solo numero: el cociente de sumas de las k tareas
cerradas, que corrige por igual a todas las tareas que quedan. Eso tira toda la
informacion de QUE tareas eran. Si el usuario nos dice que «migrar el servicio de
informes» le costo 14 h, eso dice mucho mas sobre otras migraciones del backlog que
sobre «corregir un typo en el pie de pagina».

Aqui el desplazamiento se calcula POR TAREA, pesando las historicas por parecido de
texto (coseno sobre los vectores de e5-large, con softmax de temperatura tau):

    w(j,h)  = softmax( similitud(j,h) / tau )     sobre las k historicas
    off(j)  = log( SUMA_h w(j,h) r_h  /  SUMA_h w(j,h) p_h )   x  credibilidad

Se mantiene el cociente de SUMAS tambien en la version local, porque el objetivo
sigue siendo un total. Y la credibilidad usa el TAMANO EFECTIVO de muestra
k_ef = 1 / SUMA w^2: si todo el peso cae en una sola tarea historica, k_ef = 1 y
la correccion se frena casi del todo, aunque k sea 20. Eso es lo que evita que
concentrar el peso en un vecino ruidoso estropee la estimacion.

  A  en frio
  B  desplazamiento GLOBAL, regla fija k/(k+12)        el mecanismo actual
  C  desplazamiento LOCAL por similitud
  D  local, y cuando la evidencia local es debil se cae al global

tau se elige dejando fuera el proyecto evaluado, nunca mirando su resultado.

Mismo protocolo que E78: 78 proyectos con 25+ tareas, el error se mide SOLO sobre
las tareas que quedan por estimar, 30 sorteos, promediando la METRICA.
"""
import json
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 215)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/historial_por_texto")
S = "=" * 112
LAMBDA, SORTEOS, KS, MIN_TAREAS = 12, 30, (1, 3, 5, 10, 20), 25
TAUS = (0.02, 0.05, 0.10, 0.20, 0.50)


def mt(real, pred):
    real, pred = np.asarray(real, float), np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    m = float(np.median(np.log(pred / real)))
    return dict(PRED25=float(np.mean(r <= .25)), MdMRE=float(np.median(r)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                MAE_h=float(np.abs(pred - real).mean()),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    todo = df[df.is_software_task == 1].reset_index(drop=True)
    V = np.load(CACHE / "e5-large-todo.npy")
    assert len(V) == len(todo), f"{len(V)} vectores para {len(todo)} tareas"
    noap = (~todo.fuente_jira.eq("Apache")).to_numpy()
    sw, V = todo[noap].reset_index(drop=True), V[noap]
    pred = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    assert len(pred) == len(sw)
    Vn = (V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)).astype(
        np.float32)

    tam = sw.groupby("project").size()
    nom = list(tam[tam >= MIN_TAREAS].index)
    proy, real = sw.project.to_numpy(), sw.effort_hours.to_numpy(float)
    idx = {p: np.flatnonzero(proy == p) for p in nom}
    print(f"{len(nom)} proyectos · {sum(len(idx[p]) for p in nom):,} tareas · "
          f"vectores {Vn.shape} · {time.time() - t0:.0f} s", flush=True)

    # ---- una pasada: para cada (proyecto, k, sorteo) y cada tau, el total ----
    reg = []
    for p in nom:
        i = idx[p]
        r_i, p_i, V_i = real[i], pred[i], Vn[i]
        for k in KS:
            for s in range(SORTEOS):
                rng = np.random.default_rng(zlib.crc32(f"{p}|{k}|{s}".encode()))
                sel = rng.choice(len(i), k, replace=False)
                resto = np.setdiff1d(np.arange(len(i)), sel)
                rs, ps = r_i[sel], p_i[sel]
                rr, pr = r_i[resto], p_i[resto]
                fila = dict(p=p, k=k, s=s, real=float(rr.sum()),
                            A=float(pr.sum()))
                # B: global con la regla fija
                off_g = np.log(rs.sum() / ps.sum()) * k / (k + LAMBDA)
                fila["B"] = float((pr * np.exp(off_g)).sum())
                # C y D: local por similitud, para cada tau
                sim = V_i[resto] @ V_i[sel].T                    # (n_resto, k)
                for tau in TAUS:
                    z = sim / tau
                    z -= z.max(1, keepdims=True)
                    w = np.exp(z)
                    w /= w.sum(1, keepdims=True)
                    k_ef = 1.0 / np.maximum((w ** 2).sum(1), 1e-12)
                    cred = k_ef / (k_ef + LAMBDA)
                    off_l = np.log(np.maximum(w @ rs, 1e-9) /
                                   np.maximum(w @ ps, 1e-9))
                    fila[f"C{tau}"] = float((pr * np.exp(off_l * cred)).sum())
                    # D: lo que la evidencia local no cubre lo pone el global
                    mez = off_l * cred + (1 - cred) * off_g
                    fila[f"D{tau}"] = float((pr * np.exp(mez)).sum())
                reg.append(fila)
        if (nom.index(p) + 1) % 20 == 0:
            print(f"  {nom.index(p) + 1}/{len(nom)} proyectos · "
                  f"{time.time() - t0:.0f} s", flush=True)
    M = pd.DataFrame(reg)
    print(f"  {len(M):,} combinaciones · {time.time() - t0:.0f} s", flush=True)

    # ---- elegir tau dejando fuera el proyecto evaluado ---------------------
    def elegir_y_evaluar(pref):
        """Para cada proyecto, tau se fija con los OTROS proyectos."""
        col = np.empty(len(M))
        for p in nom:
            te = (M.p == p).to_numpy()
            otros = M[~te]
            # se elige por PRED25 medio sobre los otros proyectos, por k
            mejor = {}
            for k in KS:
                ok = otros[otros.k == k]
                pts = {tau: np.mean([mt(gg.real, gg[f"{pref}{tau}"]).get("PRED25")
                                     for _, gg in ok.groupby("s")]) for tau in TAUS}
                mejor[k] = max(pts, key=pts.get)
            # vectorizado por k: itertuples no deja indexar por nombre, y ademas
            # las columnas llevan punto ("C0.05"), que no es un identificador
            kk = M.k.to_numpy()
            for k in KS:
                m2 = te & (kk == k)
                if m2.any():
                    col[np.flatnonzero(m2)] = M.loc[m2, f"{pref}{mejor[k]}"].to_numpy()
        return col

    t1 = time.time()
    M["C"] = elegir_y_evaluar("C")
    M["D"] = elegir_y_evaluar("D")
    print(f"  tau elegido fuera de muestra · {time.time() - t1:.0f} s", flush=True)

    brazos = [("A · en frio", "A"), ("B · global, regla fija", "B"),
              ("C · local por similitud de texto", "C"),
              ("D · local, con respaldo global", "D")]
    out = []
    for k in KS:
        dk = M[M.k == k]
        for etq, c in brazos:
            ms = [mt(g.real, g[c]) for _, g in dk.groupby("s")]
            out.append(dict(k=k, brazo=etq,
                            **{x: float(np.mean([m[x] for m in ms])) for x in ms[0]}))
    t = pd.DataFrame(out)
    print(f"\n{S}\nUSAR EL TEXTO DE LAS HISTORICAS ({len(nom)} proyectos, error solo "
          f"sobre lo que queda)\n{S}")
    for c in ("PRED25", "MdMRE"):
        print(f"\n{c}:")
        print(t.pivot(index="brazo", columns="k", values=c).round(4).to_string())

    print(f"\n{S}\nLECTURA\n{S}")
    gana = 0
    for k in KS:
        b = float(t[(t.k == k) & t.brazo.str.startswith("B")].PRED25.iloc[0])
        c_ = float(t[(t.k == k) & t.brazo.str.startswith("C")].PRED25.iloc[0])
        d_ = float(t[(t.k == k) & t.brazo.str.startswith("D")].PRED25.iloc[0])
        m_ = max(c_, d_)
        gana += m_ > b + .005
        print(f"  k={k:2d}  global {b:.1%} · por texto {m_:.1%}  "
              f"({(m_ - b) * 100:+.1f} pts)")
    print(f"\n  gana en {gana} de {len(KS)} valores de k")
    t.to_csv(OUT / "resumen.csv", index=False)
    M.to_csv(OUT / "crudo.csv", index=False)
    json.dump({"proyectos": len(nom), "sorteos": SORTEOS, "taus": list(TAUS)},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/resumen.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
