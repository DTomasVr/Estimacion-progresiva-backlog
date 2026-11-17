"""Does the corpus's domain concentration hurt transfer to a different domain?

72,3 % of the software corpus is Apache -- distributed data infrastructure
(Arrow, Beam, Flink, Ignite, Carbondata, Geode, NiFi, Hive) written by and for
experts. PBE's tasks will be product/business work in Spanish. If a model
trained mostly on Apache transfers badly to non-Apache work, the corpus is not
just small in the useful direction, it is pointed the wrong way -- and that
changes what augmentation should produce.

Leave-one-domain-out on the frozen blocks. Features are the 768-d text vector
alone, so what moves is the text domain and nothing else; the full feature set
is kept as a control.

  entrenamiento: todo / solo Apache / todo menos Apache / solo v3 (SiP+TAWOS)
  lectura:       prueba SiP (comercial, lo mas parecido a PBE)
                 prueba TAWOS
                 externo Apache / externo no Apache
"""
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
DATA = "data/processed/pbe_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
NUM = ["n_links", "n_links_estructurales", "n_componentes", "tiene_asignado"]
CAT = ["task_kind", "source"]
SEP = "=" * 120


def hgb():
    return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                         l2_regularization=1., random_state=0)


def metricas(y, p):
    p = np.clip(np.asarray(p, float), .01, None)
    m = E.point_metrics(y, p)
    lr = np.log(p / np.asarray(y, float))
    return dict(MdMRE=m["MdMRE"], MdAElog=float(np.median(np.abs(lr))),
                PRED25=m["PRED25"], MAE_h=m["MAE_h"],
                sesgo=float(np.exp(np.median(lr))),
                spearman=float(pd.Series(y).corr(pd.Series(p), method="spearman")))


def main():
    df = pd.read_parquet(DATA)
    sw = df[df.is_software_task == 1].copy()
    emb = np.load(f"{EMB}.npy")
    ids = np.load(f"{EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    sw["fe"] = [pos[t] for t in sw.task_id]
    sw["apache"] = sw.fuente_jira.eq("Apache")

    tr = sw[sw.split_t3 == "train"]
    oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                       min_frequency=25).fit(tr[CAT].astype(str))
    sc = StandardScaler().fit(np.log1p(tr[NUM].to_numpy(float)))
    solo_texto = lambda d: emb[d.fe.to_numpy()]
    completo = lambda d: np.hstack([emb[d.fe.to_numpy()],
                                    oh.transform(d[CAT].astype(str)),
                                    sc.transform(np.log1p(d[NUM].to_numpy(float)))])

    entrenos = {
        "todo (63.926)": tr,
        "solo Apache": tr[tr.apache],
        "todo menos Apache": tr[~tr.apache],
        "solo v3: SiP+TAWOS": tr[tr.source != "JiraPublico"],
    }
    ext = sw[sw.split_t3 == "externo"]
    pruebas = {
        "prueba SiP (comercial)": sw[(sw.split_t3 == "test") & (sw.source == "SiP")],
        "prueba TAWOS": sw[(sw.split_t3 == "test") & (sw.source == "TAWOS")],
        "externo Apache": ext[ext.apache],
        "externo no Apache": ext[~ext.apache],
    }
    print("entrenamientos: " + " · ".join(f"{k} {len(v):,}" for k, v in entrenos.items()))
    print("pruebas:        " + " · ".join(f"{k} {len(v):,}" for k, v in pruebas.items()))

    filas = []
    for rasgos, f in (("solo texto (768)", solo_texto), ("completo (control)", completo)):
        for nombre, d in entrenos.items():
            m = hgb().fit(f(d), np.log(d.effort_hours.to_numpy()))
            for bloque, p in pruebas.items():
                filas.append(dict(rasgos=rasgos, entrenamiento=nombre, bloque=bloque,
                                  n_train=len(d), n=len(p),
                                  **metricas(p.effort_hours.to_numpy(),
                                             np.exp(m.predict(f(p))))))
            print(f"  ajustado [{rasgos}] {nombre}", flush=True)

    r = pd.DataFrame(filas)
    r.to_csv("results/dominio.csv", index=False)
    cols = ["entrenamiento", "n_train", "MdMRE", "MdAElog", "PRED25", "sesgo", "spearman"]
    for rasgos in r.rasgos.unique():
        for bloque in pruebas:
            v = r[(r.rasgos == rasgos) & (r.bloque == bloque)]
            print(f"\n{SEP}\n{rasgos} · {bloque} · n={v.n.iloc[0]:,}\n{SEP}")
            print(v[cols].round(4).to_string(index=False))
    print("\nescrito results/dominio.csv")


if __name__ == "__main__":
    main()
