"""El estimador v7: el v6 con UN solo codificador, el e5-large base.

E120 midio fuera de pliegue -GroupKFold por proyecto entero- que los dos
codificadores ajustados no aportan: anadir el canal ajustado a la base EMPEORA
el SA (0,0268 -> 0,0233) y el spearman (0,320 -> 0,302). En el v6 pesan 0,25 y
0,23 en el apilado porque el apilado se ajusta en split_t3 == "val", que tiene
tareas de los MISMOS proyectos que el entrenamiento de los codificadores: es la
fuga de E54, y premia justo lo que no generaliza a un proyecto nuevo.

Lo unico del v6 que los usa es el apilado de la cabeza A (la mediana por
tarea). El resto -la cabeza del total, sus bandas, el corrector con
estimacion, la tira HPD, el banco de vecinos- ya trabaja sobre el e5 base y se
hereda sin tocar. Asi que aqui:

  1. apilado de 3 canales (bolsa, ridge, svr), mismos modelos de canal del v6,
     reajustado con RidgeCV en val como se hizo siempre
  2. factor_banda80 con el cuantil CONFORME del residuo en val, por ajuste
     cruzado dentro de val para no medirlo sobre lo que ajusto el apilado
  3. umbral_dispersion re-barrido contra la nueva cabeza, como en 171
  4. se quitan hgb_ajustado(_C) y codificador_ajustado(_C)

factor_estilo NO se re-mide aqui (venia de E74 sobre otro corpus): se
comprueba que la cabeza nueva no cambia la escala sobre texto del generador.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
CACHE = Path("data/processed/embeddings/codificadores")
V6 = Path("models/estimador_pbe_v6.joblib")
V7 = Path("models/estimador_pbe_v7.joblib")
OUT = Path("results/estimador_v7")
K_VEC = 25
ALFAS = np.logspace(-2, 3, 20)

spec = importlib.util.spec_from_file_location("e6", "src/171_estimador_v6.py")
e6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = dict(joblib.load(V6))
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1]
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy(float)
    ly = np.log(y)
    va = (sw.split_t3 == "val").to_numpy()
    E5 = np.load(CACHE / "e5-large.npy")
    SV = np.hstack([sv.transform(tf.transform(tit)).astype(np.float32)
                    for tf, sv in zip((p["tfc"], p["tfw"]), p["svds"])])
    Xd = np.hstack([E5, SV])
    Ms = sparse.hstack([p["tfc"].transform(tit), p["tfw"].transform(tit)]).tocsr()
    Z = np.column_stack([np.mean([m.predict(Xd) for m in p["bolsa"]], 0),
                         p["ridge"].predict(Ms), p["svr"].predict(Ms)])
    print(f"canales listos · {time.time() - t0:.0f} s", flush=True)

    ap = RidgeCV(alphas=ALFAS).fit(Z[va], ly[va])
    lp = ap.predict(Z)
    print("  pesos " + " · ".join(f"{k} {w:+.3f}" for k, w in
                                  zip(("bolsa", "ridge", "svr"), ap.coef_))
          + f" · alfa {ap.alpha_:.2f}", flush=True)

    # banda: residuo de val con el apilado ajustado en la OTRA mitad de val
    iv = np.flatnonzero(va)
    res = np.zeros(len(iv))
    for a, b in KFold(5, shuffle=True, random_state=0).split(iv):
        m = RidgeCV(alphas=ALFAS).fit(Z[iv[a]], ly[iv[a]])
        res[b] = np.abs(ly[iv[b]] - m.predict(Z[iv[b]]))
    k80 = float(np.exp(e6.qconf(res, .8, True)))

    # umbral de acuerdo entre vecinos, contra la cabeza nueva (como en 171)
    Vn = E5 / np.maximum(np.linalg.norm(E5, axis=1, keepdims=True), 1e-9)
    BV, By, rej = p["vec_V"].astype(np.float32), p["vec_y"], p["rejilla"]
    ana, disp = np.zeros(len(y)), np.zeros(len(y))
    for i in range(0, len(y), 2048):
        s = Vn[i:i + 2048] @ BV.T
        j = np.argpartition(-s, K_VEC, axis=1)[:, :K_VEC]
        v = By[j]
        ana[i:i + 2048] = rej[np.abs(np.log(rej)[None, :] - np.log(
            np.clip(np.median(v, axis=1), 1e-6, None))[:, None]).argmin(1)]
        disp[i:i + 2048] = np.std(np.log(np.clip(v, 1e-6, None)), axis=1)
    mejor_u, mejor_v = float(np.median(disp)), -1.
    for u in np.quantile(disp[va], np.arange(.05, 1., .05)):
        q = np.where(disp <= u, ana, np.exp(lp))
        v_ = float(np.mean(np.abs(q[va] - y[va]) / y[va] <= .25))
        if v_ > mejor_v:
            mejor_u, mejor_v = float(u), v_

    viejo = {k: p[k] for k in ("factor_banda80", "umbral_dispersion")}
    for k in ("hgb_ajustado", "hgb_ajustado_C", "codificador_ajustado",
              "codificador_ajustado_C", "corrector"):
        p.pop(k, None)
    p.update({"apilado": ap, "factor_banda80": k80, "umbral_dispersion": mejor_u,
              "canales_apilado": ["bolsa", "ridge", "svr"], "version": "v7"})
    joblib.dump(p, V7, compress=3)
    info = {"pesos": dict(zip(("bolsa", "ridge", "svr"), ap.coef_.tolist())),
            "alfa": float(ap.alpha_), "factor_banda80": [viejo["factor_banda80"], k80],
            "umbral_dispersion": [viejo["umbral_dispersion"], mejor_u],
            "PRED25_val_umbral": mejor_v, "mb": V7.stat().st_size / 1e6}
    json.dump(info, open(OUT / "construccion.json", "w"), indent=1)
    print(json.dumps(info, indent=1))
    print(f"escrito {V7} · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
