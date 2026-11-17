"""E65 - Arreglar la compresion de la escala del proyecto.

E63 dejo el diagnostico: el numero de tareas se conoce exacto (coeficiente 1,001)
y el tamano tipico de tarea se predice con el 36 % de su dispersion real. La nube
de la figura 9 es horizontal.

Pero horizontal no quiere decir vacia. Midiendo el ORDEN en vez del rango:

    Spearman entre escala predicha y real   0,484
    sd predicha 0,308  ·  sd real 0,863     razon 0,357
    mapeando el rango al real manteniendo el orden predicho, el error absoluto
    mediano en log baja de 0,753 a 0,565

O sea que el modelo ordena los proyectos razonablemente bien y solo se queda
corto de recorrido. Eso es exactamente lo que arreglan tres familias de la
literatura, y las tres se prueban aqui:

  1 MAPEO DE CUANTILES (CDF matching). De la correccion de sesgo en climatologia:
    se sustituye el valor predicho por el valor real que ocupa su mismo puesto en
    la distribucion de referencia. Por construccion la salida tiene la dispersion
    correcta. Solo funciona si el orden es bueno, y aqui lo es a medias.

  2 CALIBRACION LINEAL EN LOG. Expandir alrededor del centro con la pendiente que
    pide la regresion. Es lo que probo E63 sobre el total; aqui se aplica solo al
    componente de ESCALA, que es el que esta comprimido.

  3 CAMBIO DE ETIQUETA (Lipton, Wang y Smola, 2018, "Detecting and Correcting for
    Label Shift with Black Box Predictors"). Es la que mejor encaja con el
    problema: cada proyecto es un conjunto sin etiquetar cuya distribucion de
    horas difiere de la de entrenamiento, y BBSE estima esa distribucion a partir
    de como se reparten las PREDICCIONES de sus tareas y de la matriz de
    confusion del modelo. Usa las n tareas del proyecto, no solo su mediana, asi
    que deberia ser mas estable que 1 y 2 en proyectos pequenos.

Todo se estima DEJANDO FUERA el proyecto que se evalua: el mapeo, la pendiente y
la matriz de confusion. Y se corrige el total reescalandolo por el factor que
cada metodo pide, manteniendo intacto el numero de tareas, que ya es exacto.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "8"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
OUT = Path("results/escala_proyecto_v2")
S = "=" * 122
MIN_TAREAS = 8
BORDES = np.array([0, .5, 1, 2, 4, 8, 16, 40, 1e9])      # tramos de horas


def mt(real, pred):
    real = np.asarray(real, float)
    pred = np.clip(np.asarray(pred, float), 1e-6, None)
    r = np.abs(pred - real) / real
    q = np.log(pred / real)
    m = float(np.median(q))
    lr = np.log(real)
    return dict(MAE_h=float(np.abs(pred - real).mean()),
                RMSE_h=float(np.sqrt(np.mean((pred - real) ** 2))),
                MdMRE=float(np.median(r)), MMRE=float(r.mean()),
                R2_log=float(1 - ((lr - np.log(pred)) ** 2).mean() / lr.var()),
                PRED25=float(np.mean(r <= .25)), PRED50=float(np.mean(r <= .50)),
                factor2=float(np.mean(np.maximum(pred / real, real / pred) <= 2)),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def bbse(C, p_obs):
    """Resuelve C q = p_obs con q >= 0 y suma uno.

    La inversa directa de la matriz de confusion es inestable cuando alguna clase
    es rara, que aqui pasa siempre. Se usa minimos cuadrados no negativos mas la
    restriccion de que sume uno, que es la variante robusta al uso."""
    A = np.vstack([C, np.ones((1, C.shape[1]))])
    b = np.concatenate([p_obs, [1.0]])
    q, _ = nnls(A, b)
    s = q.sum()
    return q / s if s > 1e-9 else np.full(C.shape[1], 1 / C.shape[1])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy()
    est = np.clip(np.load("results/modelo_total/pred_total.npy"), 1e-6, None)
    d = pd.DataFrame({"p": sw.project.to_numpy(), "real": y, "est": est})
    tam = d.groupby("p").size()
    d = d[d.p.isin(tam[tam >= MIN_TAREAS].index)].reset_index(drop=True)
    nom = list(pd.unique(d.p))
    pi = {p: k for k, p in enumerate(nom)}
    jp = d.p.map(pi).to_numpy()
    re_, es_ = d.real.to_numpy(), d.est.to_numpy()
    g = d.groupby("p")
    real_tot = g.real.sum().reindex(nom).to_numpy()
    pred_tot = g.est.sum().reindex(nom).to_numpy()
    n_t = g.size().reindex(nom).to_numpy().astype(float)
    # LA ESCALA ES LA MEDIA POR TAREA, NO LA MEDIANA. El total es una suma, asi
    # que lo que hay que corregir es la media: corregir la mediana y multiplicar
    # el total por ese factor mete el desajuste mediana-media que ya ha aparecido
    # tres veces esta noche, y en la primera version de este experimento hundio
    # el total (MdMRE 0,355 -> 0,598) aunque la escala mediana si mejoraba.
    esc_r = np.log(g.real.mean().reindex(nom).to_numpy())       # escala real
    esc_p = np.log(g.est.mean().reindex(nom).to_numpy())        # escala predicha
    P = len(nom)
    print(f"{P} proyectos · {len(d):,} tareas · {time.time() - t0:.0f} s", flush=True)

    # tramos de horas para el cambio de etiqueta
    cl_r = np.clip(np.digitize(re_, BORDES) - 1, 0, len(BORDES) - 2)
    cl_p = np.clip(np.digitize(es_, BORDES) - 1, 0, len(BORDES) - 2)
    K = len(BORDES) - 1
    centro = np.array([np.exp(np.mean(np.log(re_[cl_r == k]))) if (cl_r == k).any()
                       else .5 * (BORDES[k] + min(BORDES[k + 1], 200))
                       for k in range(K)])
    print("  centros geometricos de los tramos: " +
          " ".join(f"{c:.2f}" for c in centro))

    fac = {k: np.zeros(P) for k in ("cuantiles", "lineal", "etiqueta", "etiqueta_suave")}
    for i in range(P):
        fuera = np.ones(P, bool)
        fuera[i] = False
        # --- 1 mapeo de cuantiles: el puesto que ocupa, llevado al real
        u = float((esc_p[fuera] < esc_p[i]).mean())
        u = min(max(u, .5 / fuera.sum()), 1 - .5 / fuera.sum())
        esc_q = float(np.quantile(esc_r[fuera], u))
        fac["cuantiles"][i] = np.exp(esc_q - esc_p[i])
        # --- 2 calibracion lineal en log sobre la escala
        b, a = np.polyfit(esc_p[fuera], esc_r[fuera], 1)
        fac["lineal"][i] = np.exp(a + b * esc_p[i] - esc_p[i])
        # --- 3 cambio de etiqueta con la matriz de confusion de los demas
        m = fuera[jp]
        C = np.zeros((K, K))
        for k in range(K):
            s = m & (cl_r == k)
            if s.sum():
                C[:, k] = np.bincount(cl_p[s], minlength=K) / s.sum()
            else:
                C[k, k] = 1.
        mi = jp == i
        obs = np.bincount(cl_p[mi], minlength=K) / mi.sum()
        q = bbse(C, obs)
        # El factor se compara contra lo que el MODELO predice para ESTE proyecto,
        # no contra la media del entrenamiento: el modelo ya adapta un poco, y
        # dividir por la media global contaria esa parte dos veces.
        media_est = float(q @ centro)
        fac["etiqueta"][i] = media_est / max(pred_tot[i] / n_t[i], 1e-9)
        # variante que mezcla la estimacion con la distribucion de referencia,
        # porque con 8 o 20 tareas el reparto observado es muy ruidoso
        pri = np.bincount(cl_r[m], minlength=K) / m.sum()
        media_sua = float((.5 * q + .5 * pri) @ centro)
        fac["etiqueta_suave"][i] = media_sua / max(pred_tot[i] / n_t[i], 1e-9)
    print(f"factores calculados · {time.time() - t0:.0f} s", flush=True)

    brazos = {"0 · sin tocar": pred_tot}
    for k, f in fac.items():
        brazos[f"1 · {k}"] = pred_tot * f
        brazos[f"2 · {k}, a medias"] = pred_tot * np.sqrt(f)
    t = pd.DataFrame([dict(brazo=n, **mt(real_tot, v)) for n, v in brazos.items()])
    t = t.sort_values("MdMRE")
    print(f"\n{S}\nTOTAL DEL PROYECTO CON LA ESCALA CORREGIDA\n{S}")
    print(t[["brazo", "MAE_h", "RMSE_h", "MdMRE", "MMRE", "R2_log", "PRED25",
             "PRED50", "factor2", "SSPB_pc"]].round(4).to_string(index=False))

    print(f"\n{S}\nY LA ESCALA EN SI, QUE ES LO QUE SE QUERIA ARREGLAR\n{S}")
    fs = [dict(metodo="sin tocar", sd=float(esc_p.std()),
               err_log=float(np.median(np.abs(esc_p - esc_r))),
               spearman=float(pd.Series(esc_p).corr(pd.Series(esc_r), method="spearman")),
               pendiente=float(np.polyfit(esc_p, esc_r, 1)[0]))]
    for k, f in fac.items():
        e = esc_p + np.log(f)
        fs.append(dict(metodo=k, sd=float(e.std()),
                       err_log=float(np.median(np.abs(e - esc_r))),
                       spearman=float(pd.Series(e).corr(pd.Series(esc_r), method="spearman")),
                       pendiente=float(np.polyfit(e, esc_r, 1)[0])))
    fs.append(dict(metodo="REAL (referencia)", sd=float(esc_r.std()), err_log=0.,
                   spearman=1., pendiente=1.))
    c = pd.DataFrame(fs)
    print(c.round(4).to_string(index=False))
    print("  sd deberia acercarse a la del real y err_log bajar")
    t.to_csv(OUT / "totales.csv", index=False)
    c.to_csv(OUT / "escala.csv", index=False)
    np.save(OUT / "factores.npy", np.column_stack([fac[k] for k in fac]))
    json.dump({"metodos": list(fac), "bordes": BORDES[:-1].tolist()},
              open(OUT / "meta.json", "w"), indent=1)
    print(f"\nescrito {OUT}/totales.csv · {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
