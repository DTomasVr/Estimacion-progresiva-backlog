"""E31 - Dos preguntas que deciden la arquitectura.

El diagnostico de E30 deja el error muy mal repartido. Por decil del valor real:

    decil 0 (<=0,25 h)   MdAElog 1,854
    deciles 3-5 (1,5-3 h) MdAElog 0,354-0,437
    decil 9 (>=15 h)     MdAElog 1,947

O sea: el modelo es razonable en el centro y se rompe en los dos extremos. Eso
es exactamente lo que le pasa a una regresion que estima una tendencia central
con poca senal: aplasta todo hacia el medio. La pregunta que decide si merece la
pena cambiar de arquitectura es si los extremos son IMPREDECIBLES o solo estan
siendo aplastados. Si un clasificador binario los separa bien, la senal existe y
la regresion la esta tirando; si no los separa, no hay nada que rescatar.

  PREGUNTA 1  que AUC saca un clasificador de "esto es una microtarea" (<=0,5 h)
              y uno de "esto es una tarea grande" (>=16 h)

  PREGUNTA 2  la descripcion, que no estamos usando, aporta algo. Esta en el
              52 % de las tareas. Se prueba como canal aparte y con el control
              obligatorio de permutarla, que es lo que hundio a la descripcion
              de proyecto en E1.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/extremos"); OUT.mkdir(parents=True, exist_ok=True)
S = "=" * 116


def encode_desc(textos):
    f = CACHE / "e5-large-desc.npy"
    if f.exists():
        return np.load(f)
    import torch
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    st = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
    st.max_seq_length = 256
    v = st.encode(["query: " + t for t in textos], batch_size=24,
                  show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    np.save(f, v)
    print(f"  descripciones codificadas {v.shape} en {time.time() - t0:.0f} s", flush=True)
    del st; torch.cuda.empty_cache()
    return v


def main():
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    y = sw.effort_hours.to_numpy(); ly = np.log(y)
    tr = (sw.split_t3 == "train").to_numpy()
    te = sw.split_t3.isin(["test", "externo"]).to_numpy()
    E5 = np.load(CACHE / "e5-large.npy")
    desc = sw.description.fillna("").astype(str).str.slice(0, 1200).tolist()
    D = encode_desc(desc)
    tiene_d = np.array([len(t.split()) > 0 for t in desc])
    print(f"{len(sw):,} tareas · train {tr.sum():,} · prueba {te.sum():,} · "
          f"con descripcion {tiene_d.mean():.1%}")

    # ---------------------------------------------------------------- pregunta 1
    print(f"\n{S}\nPREGUNTA 1 - HAY SENAL EN LOS EXTREMOS\n{S}")
    filas = []
    for etq, obj in (("microtarea  y <= 0,5 h", y <= .5),
                     ("muy pequena y <= 0,25 h", y <= .25),
                     ("grande      y >= 16 h", y >= 16),
                     ("muy grande  y >= 40 h", y >= 40)):
        c = HistGradientBoostingClassifier(max_iter=250, max_depth=6, learning_rate=.06,
                                           l2_regularization=1., random_state=0)
        c.fit(E5[tr], obj[tr])
        p = c.predict_proba(E5[te])[:, 1]
        base = obj[te].mean()
        filas.append(dict(objetivo=etq, prevalencia=base, AUC=roc_auc_score(obj[te], p),
                          PR_AUC=average_precision_score(obj[te], p),
                          lift_PR=average_precision_score(obj[te], p) / base))
    t1 = pd.DataFrame(filas)
    print(t1.round(4).to_string(index=False))
    print("  AUC 0,5 = nada · 0,7 = util · PR_AUC hay que leerlo contra la prevalencia")

    # ---------------------------------------------------------------- pregunta 2
    print(f"\n{S}\nPREGUNTA 2 - LA DESCRIPCION APORTA\n{S}")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(sw))
    Dcero = np.where(tiene_d[:, None], D, 0)
    brazos = {
        "A · solo el titulo": E5,
        "B · titulo + descripcion": np.hstack([E5, Dcero]),
        "C · titulo + descripcion PERMUTADA (control)": np.hstack([E5, Dcero[perm]]),
        "D · titulo + marca de tiene-descripcion": np.hstack([E5, tiene_d[:, None].astype(np.float32)]),
    }
    filas = []
    for etq, X in brazos.items():
        r = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                          l2_regularization=1., random_state=0)
        r.fit(X[tr], ly[tr])
        p = r.predict(X[te])
        e = np.abs(np.exp(p) - y[te]) / y[te]
        filas.append(dict(brazo=etq, MdAElog=float(np.median(np.abs(ly[te] - p))),
                          R2log=float(1 - ((ly[te] - p) ** 2).mean() / ly[te].var()),
                          PRED25=float(np.mean(e <= .25)),
                          spearman=float(pd.Series(y[te]).corr(pd.Series(np.exp(p)), method="spearman"))))
        print(f"  {etq:46s} listo", flush=True)
    t2 = pd.DataFrame(filas)
    print(t2.round(4).to_string(index=False))
    d = t2.set_index("brazo")
    print(f"\n  senal real de la descripcion (B menos el control C): "
          f"R2 {d.R2log['B · titulo + descripcion'] - d.R2log['C · titulo + descripcion PERMUTADA (control)']:+.4f}")

    # solo sobre las tareas que SI tienen descripcion
    k = te & tiene_d
    print(f"\n  restringido a las {k.sum():,} tareas de prueba que si tienen descripcion:")
    for etq, X in brazos.items():
        r = HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                          l2_regularization=1., random_state=0)
        r.fit(X[tr], ly[tr]); p = r.predict(X[k])
        print(f"    {etq:46s} MdAElog {np.median(np.abs(ly[k] - p)):.4f} · "
              f"R2log {1 - ((ly[k] - p) ** 2).mean() / ly[k].var():.4f}")
    t1.to_csv(OUT / "extremos.csv", index=False); t2.to_csv(OUT / "descripcion.csv", index=False)
    print(f"\nescrito {OUT}/extremos.csv y descripcion.csv")


if __name__ == "__main__":
    main()
