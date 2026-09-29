"""En frio y con estimacion del usuario, los dos sobre la configuracion NUEVA.

Los experimentos 71 y 77 midieron la estimacion del usuario sobre la
configuracion vieja: titulo + descripcion, corpus entero dominado por Apache y
con `source` y contadores entre los rasgos. Desde entonces la configuracion
cambio por medicion -- solo titulo, sin Apache, sin categoria -- y las dos
lineas de trabajo se quedaron desalineadas. Esto las vuelve a juntar.

Un efecto secundario util de quitar Apache: la densidad de filas con estimacion
humana sube del 14,1 % al 48,9 %, asi que el corrector casi no pierde datos al
entrenarse sobre el mismo corpus que el modelo en frio.

  R0  en frio: solo el texto del titulo. Es lo que el artefacto necesita.
  H0  la estimacion del usuario tal cual
  C1  correccion: texto + log(estimacion) + prediccion en frio fuera de pliegue
  C2  correccion sin texto: log(estimacion) + prediccion en frio
  D1  despliegue: C ganador si hay estimacion, frio si no -- sobre el bloque
      entero, que es lo que el producto ve

La prediccion en frio entra como rasgo FUERA DE PLIEGUE en las filas de
entrenamiento; si no, el corrector se fia de un rasgo mejor del que tendra.
El brazo se elige por SA en validacion, por lo que se midio en 77: el MdMRE y
el MdAElog son medianas y no ven la cola, y el estimado crudo del usuario es
excelente en la tarea tipica y desastroso en una minoria.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("m71", "src/71_estimacion_usuario.py")
M71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M71)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 235)
DATA = "data/processed/agix_effort_v4.parquet"
EMB_TIT = "data/processed/embeddings/v4_mpnet_titulos"
OUT = Path("results/frio_vs_estimacion")
SEP = "=" * 150

R0 = "R0 - en frio (solo el texto)"
H0 = "H0 - estimacion del usuario, tal cual"
C1 = "C1 - correccion: texto + estimacion + frio"
C2 = "C2 - correccion sin texto: estimacion + frio"


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0)


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    y = np.asarray(y, float)
    m = E.point_metrics(y, p)
    lr = np.log(p / y)
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lr))), PRED25=m["PRED25"],
                MAE_h=m["MAE_h"], sesgo=float(np.exp(np.median(lr))),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")),
                SA=M71.sa(y, p), suma_pred_real=float(p.sum() / y.sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    sw["apache"] = sw.fuente_jira.eq("Apache")
    sw = sw[~sw.apache].copy()
    emb = np.load(f"{EMB_TIT}.npy")
    ids = np.load(f"{EMB_TIT}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fe"] = [pos[t] for t in sw.task_id]

    sw["est"] = sw.HoursEstimate.where(sw.HoursEstimate > 0)
    pt = sw[(sw.source == "TAWOS") & (sw.story_point > 0)]
    ptr = pt[pt.split_t3 == "train"]
    pend, corte = np.polyfit(np.log(ptr.story_point.to_numpy(float)),
                             np.log(ptr.effort_hours.to_numpy(float)), 1)
    sw.loc[pt.index, "est"] = np.exp(corte) * pt.story_point.to_numpy(float) ** pend

    X = lambda d: emb[d.fe.to_numpy()]
    pool = sw[sw.split_t3 == "train"]
    con = sw[sw.est.notna()]
    tr, va = con[con.split_t3 == "train"], con[con.split_t3 == "val"]
    print(f"corpus sin Apache · pool en frío {len(pool):,} · con estimación "
          f"{len(tr):,} entrenamiento y {len(va):,} validación "
          f"({len(tr) / len(pool):.1%} de densidad)")

    # frío: fuera de pliegue en entrenamiento, modelo completo fuera
    Xp, yp = X(pool), np.log(pool.effort_hours.to_numpy())
    oof = np.zeros(len(pool))
    for i, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(Xp)):
        oof[b] = hgb().fit(Xp[a], yp[a]).predict(Xp[b])
    frio = hgb().fit(Xp, yp)
    log_frio = pd.Series(oof, index=pool.index)
    print(f"  frío fuera de pliegue MdMRE "
          f"{E.point_metrics(pool.effort_hours, np.exp(oof))['MdMRE']:.4f} "
          f"(dentro de muestra {E.point_metrics(pool.effort_hours, np.exp(frio.predict(Xp)))['MdMRE']:.4f})")

    def lf(d):
        s = np.array(log_frio.reindex(d.index), dtype=float)
        falta = np.isnan(s)
        if falta.any():
            s[falta] = frio.predict(X(d[falta]))
        return s[:, None]

    le = lambda d: np.log(d.est.to_numpy(float))[:, None]
    con_todo = lambda d: np.hstack([X(d), le(d), lf(d)])
    sin_texto = lambda d: np.hstack([le(d), lf(d)])
    y_tr = np.log(tr.effort_hours.to_numpy())
    razon = y_tr - np.log(tr.est.to_numpy(float))
    c1 = hgb().fit(con_todo(tr), razon)
    c2 = hgb().fit(sin_texto(tr), razon)

    def brazos(d):
        e = d.est.to_numpy(float)
        return {R0: np.exp(frio.predict(X(d))), H0: e,
                C1: e * np.exp(c1.predict(con_todo(d))),
                C2: e * np.exp(c2.predict(sin_texto(d)))}

    bv = brazos(va)
    y_va = va.effort_hours.to_numpy()
    sel = {k: M71.sa(y_va, np.clip(v, .01, None)) for k, v in bv.items()}
    sel_md = {k: E.point_metrics(y_va, v)["MdMRE"] for k, v in bv.items()}
    ganador = max((k for k in sel if k[0] == "C"), key=sel.get)
    print("  SA en validación: " + " · ".join(f"{k[:2]} {v:+.4f}" for k, v in sel.items()))
    print("  MdMRE en validación: " + " · ".join(f"{k[:2]} {v:.4f}" for k, v in sel_md.items()))
    print(f"  corrector elegido por SA: {ganador}")

    pruebas = {"prueba fija · con estimación":
                   con[(con.split_t3 == "test")],
               "prueba SiP · con estimación":
                   con[(con.split_t3 == "test") & (con.source == "SiP")],
               "externo no Apache · con estimación":
                   con[con.split_t3 == "externo"]}
    filas = []
    for esc, d in pruebas.items():
        bz = brazos(d)
        y = d.effort_hours.to_numpy()
        for nombre, p in bz.items():
            fila = dict(escenario=esc, brazo=nombre, n=len(d), cobertura=1.0,
                        **metricas(y, p))
            if not nombre.startswith("R0"):
                fila["vs_frio"] = M71.comparar(y, np.clip(p, .01, None), bz[R0])
            filas.append(fila)

    # bloques enteros, con caída a frío donde no hay estimación
    completos = {"prueba fija COMPLETA": sw[sw.split_t3 == "test"],
                 "externo no Apache COMPLETO": sw[sw.split_t3 == "externo"]}
    for esc, d in completos.items():
        y = d.effort_hours.to_numpy()
        hay = d.est.notna().to_numpy()
        r0 = np.exp(frio.predict(X(d)))
        bh = brazos(d[hay]) if hay.any() else {}
        for nombre, brazo in ((R0, None), ("D0 - cruda si hay, frio si no", H0),
                              (f"D1 - {ganador[:2]} si hay, frio si no", ganador)):
            p = r0.copy()
            if brazo is not None and hay.any():
                p[hay] = bh[brazo]
            fila = dict(escenario=esc, brazo=nombre, n=len(d), cobertura=float(hay.mean()),
                        **metricas(y, p))
            if brazo is not None:
                fila["vs_frio"] = M71.comparar(y, np.clip(p, .01, None), r0)
            filas.append(fila)

    r = pd.DataFrame(filas)
    r.to_csv(OUT / "resumen.csv", index=False)
    cols = ["brazo", "n", "cobertura", "MdMRE", "MdAElog", "PRED25", "MAE_h", "sesgo",
            "spearman", "SA", "suma_pred_real", "vs_frio"]
    for esc in list(pruebas) + list(completos):
        v = r[r.escenario == esc]
        print(f"\n{SEP}\n{esc} · n={v.n.iloc[0]:,}\n{SEP}")
        print(v.reindex(columns=cols).round(4).to_string(index=False))
    json.dump({"ganador": ganador, "SA_val": sel, "MdMRE_val": sel_md,
               "n_pool": len(pool), "n_train_est": len(tr), "n_val_est": len(va),
               "densidad_estimacion": len(tr) / len(pool)},
              open(OUT / "seleccion.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv y seleccion.json")


if __name__ == "__main__":
    main()
