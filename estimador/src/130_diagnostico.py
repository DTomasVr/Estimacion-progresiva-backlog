"""E30 - Donde falla el modelo, exactamente. Diagnostico antes de tocar nada.

Ocho cortes sobre el bloque de prueba, cada uno pensado para descartar o
confirmar una hipotesis concreta sobre el error:

  1 COMPRESION      regresar log(real) sobre log(pred). Si la pendiente es >1 el
                    modelo esta encogido hacia la media y el rango que predice es
                    mas estrecho que el real. Es el fallo clasico de cualquier
                    regresion con poca senal, y decide si conviene expandir.
  2 POR DECIL       error por decil del valor REAL. Si sube en los extremos y es
                    minimo en el centro, es el mismo encogimiento visto de otra
                    forma, no un fallo de casos raros.
  3 POR PROYECTO    cuanto del error queda explicado por un desplazamiento fijo
                    por proyecto. Cota superior de lo que puede dar cualquier
                    canal de contexto de proyecto.
  4 LONGITUD        error por numero de palabras del titulo. Los titulos cortos
                    no tienen de donde sacar senal.
  5 ATOMOS          aciertos sobre valores frecuentes frente a valores raros.
  6 DESCRIPCION     cuantas tareas tienen descripcion y si el error difiere.
                    Se descarto usarla; hay que comprobar que fue buena idea.
  7 SESGO LOCAL     sesgo dentro de cada tramo de la PREDICCION. Si hay sesgo
                    sistematico por tramo, se arregla con una calibracion y es
                    dinero gratis.
  8 PEORES CASOS    los textos concretos donde mas se equivoca, para mirarlos.
"""
import sys, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from scipy import sparse
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/diagnostico"); OUT.mkdir(parents=True, exist_ok=True)
S = "=" * 118


def main():
    f = joblib.load("models/estimador_agix_v3.joblib")
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str)
    y = sw.effort_hours.to_numpy(); ly = np.log(y)
    E5 = np.load(CACHE / "e5-large.npy"); AJ = np.load(CACHE / "e5-ajustado.npy")
    Ms = sparse.hstack([f["tfc"].transform(tit), f["tfw"].transform(tit)]).tocsr()
    Xd = np.hstack([E5] + [sv.transform(tf.transform(tit)).astype(np.float32)
                           for tf, sv in zip((f["tfc"], f["tfw"]), f["svds"])])
    Z = np.column_stack([np.mean([m.predict(Xd) for m in f["bolsa"]], 0),
                         f["ridge"].predict(Ms), f["svr"].predict(Ms),
                         f["hgb_ajustado"].predict(AJ)])
    lp = f["apilado"].predict(Z)
    sw["lp"] = lp; sw["ly"] = ly; sw["res"] = ly - lp
    te = sw[sw.split_t3.isin(["test", "externo"])].copy()
    tr = sw[sw.split_t3 == "train"]
    print(f"prueba {len(te):,} tareas  ·  entrenamiento {len(tr):,}")

    # 1 - compresion
    b, a = np.polyfit(te.lp, te.ly, 1)
    print(f"\n{S}\n1 - COMPRESION\n{S}")
    print(f"  log(real) = {a:+.3f} {b:+.3f} * log(pred)   -> pendiente {b:.3f}")
    print(f"  sd de la prediccion {te.lp.std():.3f}  ·  sd del real {te.ly.std():.3f}"
          f"  ·  razon {te.lp.std()/te.ly.std():.3f}")
    print(f"  interpretacion: el modelo cubre solo el {te.lp.std()/te.ly.std():.0%} del rango real."
          f"  Para insesgar en media habria que expandir x{b:.2f} alrededor de la media.")

    # 2 - por decil del real
    print(f"\n{S}\n2 - ERROR POR DECIL DEL VALOR REAL\n{S}")
    te["dec"] = pd.qcut(te.ly, 10, labels=False, duplicates="drop")
    g = te.groupby("dec").apply(lambda d: pd.Series(dict(
        n=len(d), real_mediana=np.exp(d.ly.median()), pred_mediana=np.exp(d.lp.median()),
        sesgo_x=np.exp(d.res.median() * -1), MdAElog=d.res.abs().median())))
    print(g.round(3).to_string())

    # 3 - cuanto explica un desplazamiento por proyecto
    print(f"\n{S}\n3 - CUANTO ARREGLARIA CONOCER EL PROYECTO\n{S}")
    off = tr.groupby("project").res.median()
    glob = float(tr.res.median())
    aj = te.lp + te.project.map(off).fillna(glob).to_numpy()
    aj_or = te.lp + te.groupby("project").res.transform("median").to_numpy()
    for etq, v in (("sin nada", te.lp), ("offset del entrenamiento", aj),
                   ("offset oraculo (cota)", aj_or)):
        r = (te.ly - v).abs()
        print(f"  {etq:26s} MdAElog {r.median():.4f}  ·  "
              f"R2log {1 - ((te.ly - v) ** 2).mean() / te.ly.var():.4f}  ·  "
              f"PRED25 {np.mean(np.abs(np.exp(v) - np.exp(te.ly)) / np.exp(te.ly) <= .25):.4f}")

    # 4 - longitud
    print(f"\n{S}\n4 - ERROR POR LONGITUD DEL TITULO\n{S}")
    te["nw"] = te.title.fillna("").str.split().str.len()
    te["tramo_nw"] = pd.cut(te.nw, [0, 4, 6, 8, 11, 15, 200])
    print(te.groupby("tramo_nw").apply(lambda d: pd.Series(dict(
        n=len(d), MdAElog=d.res.abs().median(),
        PRED25=np.mean(np.abs(np.exp(d.lp) - np.exp(d.ly)) / np.exp(d.ly) <= .25)))
    ).round(4).to_string())

    # 5 - atomos
    print(f"\n{S}\n5 - VALORES FRECUENTES FRENTE A RAROS\n{S}")
    c = tr.effort_hours.value_counts()
    frec = set(c[c >= 60].index)
    te["es_frecuente"] = te.effort_hours.isin(frec)
    print(te.groupby("es_frecuente").apply(lambda d: pd.Series(dict(
        n=len(d), MdAElog=d.res.abs().median(),
        PRED25=np.mean(np.abs(np.exp(d.lp) - np.exp(d.ly)) / np.exp(d.ly) <= .25)))
    ).round(4).to_string())

    # 6 - descripcion
    print(f"\n{S}\n6 - LA DESCRIPCION QUE NO ESTAMOS USANDO\n{S}")
    te["desc_larga"] = te.description.fillna("").str.split().str.len()
    print(f"  tareas con descripcion no vacia: {(te.desc_larga > 0).mean():.1%}"
          f"  ·  mediana de palabras {te.desc_larga[te.desc_larga > 0].median():.0f}")
    te["tramo_d"] = pd.cut(te.desc_larga, [-1, 0, 20, 60, 150, 100000])
    print(te.groupby("tramo_d").apply(lambda d: pd.Series(dict(
        n=len(d), MdAElog=d.res.abs().median()))).round(4).to_string())

    # 7 - sesgo local por tramo de la prediccion
    print(f"\n{S}\n7 - SESGO DENTRO DE CADA TRAMO DE LA PREDICCION\n{S}")
    te["tramo_p"] = pd.qcut(te.lp, 12, labels=False, duplicates="drop")
    h = te.groupby("tramo_p").apply(lambda d: pd.Series(dict(
        n=len(d), pred=np.exp(d.lp.median()), real=np.exp(d.ly.median()),
        factor_que_falta=np.exp(d.res.median()), MdAElog=d.res.abs().median())))
    print(h.round(3).to_string())
    print("  si 'factor_que_falta' se aleja de 1 de forma monotona, una calibracion isotonica lo arregla")

    # 8 - peores casos
    print(f"\n{S}\n8 - VEINTE CASOS DONDE MAS SE EQUIVOCA\n{S}")
    w = te.reindex(te.res.abs().sort_values(ascending=False).index).head(20)
    for _, r in w.iterrows():
        print(f"  real {r.effort_hours:8.1f} h  pred {np.exp(r.lp):6.2f} h  "
              f"[{r.project[:14]:14s}] {str(r.title)[:72]}")
    te[["task_id", "project", "title", "effort_hours", "lp", "res"]].to_csv(
        OUT / "residuos.csv", index=False)
    g.to_csv(OUT / "por_decil.csv"); h.to_csv(OUT / "sesgo_tramo.csv")
    print(f"\nescrito {OUT}/residuos.csv")


if __name__ == "__main__":
    main()
