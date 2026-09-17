"""The user estimates, the model corrects it -- now over everything the dataset allows.

Script 71 measured the alternative production design on the rows that carry a
human estimate, but it trained the correcting models on those rows alone: 8.679
of the 63.926 training tasks. The whole corpus was therefore idle while the
question being asked was precisely whether more data helps the correction.

This run uses everything that can honestly be used:

  1. The cold model is still fitted on the full training pool, but its
     prediction now enters the corrector as a feature. Because a model's
     in-sample prediction is optimistically good, the training rows get
     OUT-OF-FOLD cold predictions (5 folds over the pool) while validation,
     test and the external block get the prediction of the model fitted on the
     whole pool. This is the only way the corrector can benefit from the 54.906
     tasks that have no estimate without being fooled about how good the cold
     prediction is.

  2. A third estimate channel is opened. SiP and the public Jira block give
     estimates in hours; TAWOS gives story points, which are converted to hours
     by a log-linear fit on TAWOS training rows only, and every arm gets a flag
     saying which channel a row came from. Story points are a weak signal
     (Spearman 0,17 against hours on train), so TAWOS is always reported as its
     own scenario and never mixed into the hours numbers.

  3. The full test blocks are scored, not only their estimate-bearing part. In
     deployment a task either has a user estimate or it does not, so arm D1
     corrects when there is an estimate and falls back to the cold model when
     there is not. That is the number the product would actually show.

  4. An adoption curve: what the fixed test block looks like when 0%, 10% ...
     100% of the users bother to type an estimate.

Arms
  R0  cold model, no estimate                              (the reference)
  H0  the user's estimate, uncorrected
  H2  estimate x per-developer factor, shrunk to the global one  (no ML)
  C1  correction with the model: text + estimate + cold prediction,
      target log(actual/estimate)
  C2  correction without text: estimate + cold prediction + category
  C3  absolute model on text + estimate + cold prediction
  C4  weighted geometric blend estimate^w x cold^(1-w), w chosen on validation
  D1  deployment: the winning corrector where an estimate exists, cold elsewhere

Selection between arms and of w happen on validation only. Metrics, the
standardised accuracy and the paired verdicts are imported from script 71 so
the two tables can be read side by side.
"""
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

spec = importlib.util.spec_from_file_location("m71", "src/71_estimacion_usuario.py")
M71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M71)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)
DATA = "data/processed/agix_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
OUT = Path("results/estimacion_completa")
NUM, CAT, PROHIBIDAS = M71.NUM, M71.CAT, M71.PROHIBIDAS
SEP = "=" * 155


def mdaelog(y, p):
    """Symmetric median absolute log error, reported for contrast."""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    return float(np.median(np.abs(np.log(p / y))))


# Which criterion selects the arm is not a detail here: the three candidate
# criteria disagree, and they disagree for a reason worth stating.
#
#   MdMRE      is asymmetric -- the relative error of an undershoot caps at 1
#              while an overshoot is unbounded -- and about a third of the rows
#              have actual == estimate exactly. It therefore rewards leaving
#              the estimate untouched.
#   MdAElog    fixes the asymmetry but is still a median: it describes the
#              typical task and is blind to the tail.
#   SA         (Shepperd & MacDonell 2012) is built on the MEAN absolute error
#              and standardised against random guessing, so it sees the tail
#              and is comparable across data sets.
#
# The raw user estimate is excellent on the typical task and catastrophic on a
# minority of them: on a new client it has the best median log error of all the
# arms AND a mean error of 12,6 h with SA below zero, i.e. worse than guessing.
# An artefact that adds task estimates into a project total cannot live with
# that tail, so the arm is chosen by validation SA and the other two criteria
# are reported next to it.
def sa_val(y, p):
    return M71.sa(np.asarray(y, float), np.clip(np.asarray(p, float), .01, None))


def metricas(y, p):
    m = M71.metricas(y, p)
    lr = np.log(np.clip(np.asarray(p, float), .01, None) / np.asarray(y, float))
    m["MdAElog"] = float(np.median(np.abs(lr)))
    m["sesgo"] = float(np.exp(np.median(lr)))
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    emb = np.load(f"{EMB}.npy")
    ids = np.load(f"{EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fe"] = [pos[t] for t in sw.task_id]

    # ---------------------------------------------- canales de estimacion
    sw["est"] = sw.HoursEstimate.where(sw.HoursEstimate > 0)
    sw["canal"] = np.where(sw.est.notna(), "horas", "")
    pt = sw[(sw.source == "TAWOS") & (sw.story_point > 0)]
    ptr = pt[pt.split_t3 == "train"]
    pend, corte = np.polyfit(np.log(ptr.story_point.to_numpy(float)),
                             np.log(ptr.effort_hours.to_numpy(float)), 1)
    sw.loc[pt.index, "est"] = np.exp(corte) * pt.story_point.to_numpy(float) ** pend
    sw.loc[pt.index, "canal"] = "puntos"
    sw["es_puntos"] = (sw.canal == "puntos").astype(float)
    print(f"conversion de puntos ajustada en {len(ptr)} filas de entrenamiento TAWOS: "
          f"horas = {np.exp(corte):.2f} x puntos^{pend:.3f}")

    pool = sw[sw.split_t3 == "train"]
    con = sw[sw.est.notna()]
    tr, va = con[con.split_t3 == "train"], con[con.split_t3 == "val"]
    print(f"pool en frio {len(pool):,} / entrenamiento con estimacion {len(tr):,} "
          f"({(tr.canal == 'puntos').sum()} en puntos) / validacion {len(va):,}")

    # ---------------------------------------------- rasgos
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                       min_frequency=25).fit(pool[CAT].astype(str))
    sc = StandardScaler().fit(np.log1p(pool[NUM].to_numpy(float)))
    base = lambda d: np.hstack([emb[d.fe.to_numpy()], oh.transform(d[CAT].astype(str)),
                                sc.transform(np.log1p(d[NUM].to_numpy(float)))])
    usadas = set(CAT) | set(NUM) | {"HoursEstimate", "story_point"}
    assert not usadas & PROHIBIDAS, "una columna derivada del objetivo entro como rasgo"

    # frio: fuera de pliegue en entrenamiento, modelo completo fuera de el
    Xp, yp = base(pool), np.log(pool.effort_hours.to_numpy())
    oof = np.zeros(len(pool))
    for i, (ia, ib) in enumerate(KFold(5, shuffle=True, random_state=0).split(Xp)):
        oof[ib] = M71.hgb().fit(Xp[ia], yp[ia]).predict(Xp[ib])
        print(f"  pliegue {i + 1}/5 del modelo en frio listo", flush=True)
    frio = M71.hgb().fit(Xp, yp)
    log_frio = pd.Series(oof, index=pool.index)
    dentro = E.point_metrics(pool.effort_hours, np.exp(frio.predict(Xp)))["MdMRE"]
    fuera = E.point_metrics(pool.effort_hours, np.exp(oof))["MdMRE"]
    print(f"  frio fuera de pliegue: MdMRE {fuera:.4f} (dentro de muestra {dentro:.4f})")

    def lf(d):
        """log de la prediccion en frio: fuera de pliegue si la fila es de entrenamiento."""
        s = np.array(log_frio.reindex(d.index), dtype=float)
        falta = np.isnan(s)
        if falta.any():
            s[falta] = frio.predict(base(d[falta]))
        return s[:, None]

    le = lambda d: np.log(d.est.to_numpy(float))[:, None]
    ep = lambda d: d.es_puntos.to_numpy(float)[:, None]
    con_todo = lambda d: np.hstack([base(d), le(d), lf(d), ep(d)])
    sin_texto = lambda d: np.hstack([le(d), lf(d), ep(d), oh.transform(d[CAT].astype(str))])

    y_tr = np.log(tr.effort_hours.to_numpy())
    razon_tr = y_tr - np.log(tr.est.to_numpy(float))
    c1 = M71.hgb().fit(con_todo(tr), razon_tr)
    c2 = M71.hgb().fit(sin_texto(tr), razon_tr)
    c3 = M71.hgb().fit(con_todo(tr), y_tr)
    print("correctores ajustados")

    k_global = float(np.exp(np.median(razon_tr)))
    por_dev = (pd.DataFrame({"dev": tr.desarrollador, "r": razon_tr})
               .dropna().groupby("dev").r.agg(["sum", "size"]))

    def factor_dev(d, fuerza):
        mu = np.log(k_global)
        f = (por_dev["sum"] + fuerza * mu) / (por_dev["size"] + fuerza)
        return np.exp(d.desarrollador.map(f).fillna(mu).to_numpy(float))

    C1 = "C1 - correccion con el modelo (texto + estimacion + frio)"
    C2 = "C2 - correccion sin texto (estimacion + frio + categoria)"
    C3 = "C3 - modelo absoluto con estimacion y frio"
    C4 = "C4 - mezcla geometrica ponderada"
    R0 = "R0 - modelo en frio (sin estimacion)"
    H0 = "H0 - estimacion del usuario, tal cual"
    H2 = "H2 - estimacion x factor por desarrollador"

    def brazos(d, fuerza, w):
        e = d.est.to_numpy(float)
        f = np.exp(lf(d).ravel())
        return {R0: np.exp(frio.predict(base(d))),
                H0: e,
                H2: e * factor_dev(d, fuerza),
                C1: e * np.exp(c1.predict(con_todo(d))),
                C2: e * np.exp(c2.predict(sin_texto(d))),
                C3: np.exp(c3.predict(con_todo(d))),
                C4: e ** w * f ** (1 - w)}

    # ---------------------------------------------- seleccion en validacion
    y_va = va.effort_hours.to_numpy()
    e_va, f_va = va.est.to_numpy(float), np.exp(lf(va).ravel())
    pesos = {round(float(w), 2): sa_val(y_va, e_va ** w * f_va ** (1 - w))
             for w in np.arange(0, 1.01, .05)}
    w = max(pesos, key=pesos.get)
    fuerzas = {f: sa_val(y_va, e_va * factor_dev(va, f)) for f in (1, 5, 10, 20, 50, 100)}
    fuerza = max(fuerzas, key=fuerzas.get)
    bva = brazos(va, fuerza, w)
    sel = {k: sa_val(y_va, v) for k, v in bva.items()}
    sel_log = {k: mdaelog(y_va, v) for k, v in bva.items()}
    sel_mdmre = {k: E.point_metrics(y_va, v)["MdMRE"] for k, v in bva.items()}
    solo_c = [k for k in sel if k[0] == "C"]
    ganador = max(solo_c, key=sel.get)
    ganador_log = min(solo_c, key=sel_log.get)
    ganador_mdmre = min(solo_c, key=sel_mdmre.get)
    print(f"\npeso geometrico elegido w={w} / contraccion por desarrollador {fuerza}")
    print("SA      en validacion: " + " / ".join(f"{k[:2]} {v:+.4f}" for k, v in sel.items()))
    print("MdAElog en validacion: " + " / ".join(f"{k[:2]} {v:.4f}" for k, v in sel_log.items()))
    print("MdMRE   en validacion: " + " / ".join(f"{k[:2]} {v:.4f}" for k, v in sel_mdmre.items()))
    print(f"elegido por SA (criterio del trabajo): {ganador}")
    print(f"elegiria el MdAElog:                   {ganador_log}")
    print(f"elegiria el MdMRE:                     {ganador_mdmre}")

    # bandas conformes calibradas en validacion
    bandas = {}
    for nombre in (R0, ganador, H0):
        res = np.abs(np.log(y_va) - np.log(np.clip(bva[nombre], .01, None)))
        bandas[nombre] = {"factor80": float(np.exp(np.quantile(res, .8))),
                          "factor50": float(np.exp(np.quantile(res, .5)))}

    # ---------------------------------------------- lectura de prueba
    pruebas = {"SiP - prueba (misma empresa)": con[(con.source == "SiP") & (con.split_t3 == "test")],
               "TAWOS - prueba (puntos de historia)": con[(con.source == "TAWOS") & (con.split_t3 == "test")],
               "Jira - externo (cliente nuevo)": con[con.split_t3 == "externo"]}
    filas, preds = [], []
    for esc, d in pruebas.items():
        bz = brazos(d, fuerza, w)
        y = d.effort_hours.to_numpy()
        distinto = ~np.isclose(d.effort_hours, d.est)
        for sub, m in (("todas", np.ones(len(d), bool)), ("real != estimacion", distinto)):
            for nombre, p in bz.items():
                fila = dict(escenario=esc, subconjunto=sub, brazo=nombre, n=int(m.sum()),
                            cobertura=1.0, **metricas(y[m], p[m]))
                pc = np.clip(p[m], .01, None)
                if not nombre.startswith("R0"):
                    fila["vs_frio_R0"] = M71.comparar(y[m], pc, bz[R0][m])
                if not nombre.startswith("H0"):
                    fila["vs_usuario_H0"] = M71.comparar(y[m], pc, bz[H0][m])
                fila["elegido_en_val"] = nombre == ganador
                filas.append(fila)
        preds.append(pd.DataFrame({"escenario": esc, "task_id": d.task_id.to_numpy(),
                                   "fuente": d.source.to_numpy(), "canal": d.canal.to_numpy(),
                                   "real": y, "estimacion": d.est.to_numpy(),
                                   **{k[:2]: v for k, v in bz.items()}}))

    # ---------------------------------------------- bloques completos (despliegue)
    # Tres formas de montar el producto sobre el bloque entero: sin preguntar,
    # preguntando y mostrando el numero del usuario, y preguntando y corrigiendolo.
    D0 = "D0 - despliegue: estimacion cruda si hay, frio si no"
    D1 = f"D1 - despliegue: {ganador[:2]} si hay estimacion, frio si no"
    D2 = f"D2 - despliegue con el brazo que elegiria el MdMRE ({ganador_mdmre[:2]})"
    assert ganador[0] == "C", "el brazo de despliegue debe ser un corrector"
    completos = {"prueba fija completa": sw[sw.split_t3 == "test"],
                 "externo completo": sw[sw.split_t3 == "externo"]}
    for esc, d in completos.items():
        y = d.effort_hours.to_numpy()
        hay = d.est.notna().to_numpy()
        r0 = np.exp(frio.predict(base(d)))
        arms = {R0: r0}
        bh = brazos(d[hay], fuerza, w) if hay.any() else {}
        for nombre, brazo in ((D0, H0), (D1, ganador), (D2, ganador_mdmre)):
            p = r0.copy()
            if hay.any():
                p[hay] = bh[brazo]
            arms[nombre] = p
        for nombre, p in arms.items():
            fila = dict(escenario=esc, subconjunto="todas", brazo=nombre, n=len(d),
                        cobertura=float(hay.mean()), **metricas(y, p))
            if not nombre.startswith("R0"):
                fila["vs_frio_R0"] = M71.comparar(y, np.clip(p, .01, None), r0)
            fila["elegido_en_val"] = nombre == D1
            filas.append(fila)
        print(f"{esc}: {len(d):,} tareas, {hay.mean():.1%} con estimacion")

    r = pd.DataFrame(filas)
    r.to_csv(OUT / "resumen.csv", index=False)
    pd.concat(preds).to_parquet(OUT / "predicciones.parquet", index=False)

    # ---------------------------------------------- curva de adopcion
    d = sw[sw.split_t3 == "test"]
    y, hay = d.effort_hours.to_numpy(), d.est.notna().to_numpy()
    r0 = np.exp(frio.predict(base(d)))
    bh = brazos(d[hay], fuerza, w)
    variantes = {"corregida": bh[ganador], "cruda": bh[H0]}
    idx = np.flatnonzero(hay)
    curva = []
    for nombre, pb in variantes.items():
        serie = np.array(r0, copy=True)
        serie[hay] = pb
        for p in np.arange(0, 1.01, .1):
            ms = []
            for s in range(20):
                rng = np.random.default_rng(s)
                usa = np.zeros(len(d), bool)
                usa[rng.choice(idx, int(round(p * len(idx))), replace=False)] = True
                q = np.where(usa, serie, r0)
                m = E.point_metrics(y, q)
                ms.append((m["MdMRE"], m["PRED25"], m["MAE_h"], mdaelog(y, q)))
            a = np.mean(ms, 0)
            curva.append(dict(variante=nombre, adopcion=round(float(p), 2),
                              cobertura=round(float(p * hay.mean()), 4),
                              MdMRE=a[0], PRED25=a[1], MAE_h=a[2], MdAElog=a[3]))
            print(f"  {nombre:9s} adopcion {p:.0%} / cobertura {p * hay.mean():.1%} / "
                  f"MdMRE {a[0]:.4f} / MdAElog {a[3]:.4f} / MAE {a[2]:.2f} h")
    pd.DataFrame(curva).to_csv(OUT / "curva_adopcion.csv", index=False)

    json.dump({"w_geometrico": float(w), "contraccion_dev": fuerza, "k_global": k_global,
               "conversion_puntos": {"factor": float(np.exp(corte)), "exponente": float(pend),
                                     "n_train": int(len(ptr))},
               "val_SA": sel, "val_MdAElog": sel_log, "val_MdMRE": sel_mdmre,
               "pesos_val_SA": {str(k): v for k, v in pesos.items()},
               "ganador_val": ganador, "ganador_val_si_se_usara_MdMRE": ganador_mdmre,
               "ganador_val_si_se_usara_MdAElog": ganador_log,
               "bandas_val": bandas,
               "frio_oof_MdMRE": fuera, "frio_dentro_MdMRE": dentro,
               "n_pool": len(pool), "n_train_est": len(tr), "n_val_est": len(va)},
              open(OUT / "seleccion.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    cols = ["brazo", "n", "MdMRE", "MdAElog", "PRED25", "MAE_h", "sesgo", "spearman", "SA",
            "vs_frio_R0", "vs_usuario_H0"]
    for esc in list(pruebas) + list(completos):
        for sub in ("todas", "real != estimacion"):
            v = r[(r.escenario == esc) & (r.subconjunto == sub)]
            if not len(v):
                continue
            print(f"\n{SEP}\n{esc} / {sub} / n={v.n.iloc[0]:,}\n{SEP}")
            print(v.reindex(columns=cols).round(4).to_string(index=False))
    print("\nbandas conformes al 80% (factor multiplicativo): "
          + " / ".join(f"{k[:2]} x{v['factor80']:.2f}" for k, v in bandas.items()))
    print(f"\nescrito {OUT}/resumen.csv, predicciones.parquet, curva_adopcion.csv, seleccion.json")


if __name__ == "__main__":
    main()
