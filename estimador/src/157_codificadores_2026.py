"""E51 - Codificadores de 2025 y 2026, que en E4 no existian o no se probaron.

E4 comparo cinco codificadores en marzo y gano e5-large, que es de 2024. Desde
entonces han salido modelos claramente mejores en las pruebas de recuperacion, y
sobre todo una familia que E4 no toco: los que aceptan INSTRUCCION. A esos se les
puede decir para que se quiere el vector -- "representa esta tarea de software
para estimar su esfuerzo" -- y eso suele mover bastante mas que cambiar de
arquitectura.

Ahora ademas codificar es barato: con el recorte dinamico y los titulos de 18
piezas de mediana, cada modelo tarda minutos.

  intfloat/multilingual-e5-large-instruct   mismo tamano que el actual, checkpoint
                                            con instrucciones
  Alibaba-NLP/gte-multilingual-base         305M, contexto largo, 2024
  jinaai/jina-embeddings-v3                 570M, adaptadores por tarea
  Qwen/Qwen3-Embedding-0.6B                 ya en cache de E4, como referencia

Para que la comparacion aisle al codificador, todos pasan por EL MISMO HGB sobre
las mismas particiones congeladas. Lo unico que cambia es quien produce los
vectores. Y se prueba tambien la concatenacion del mejor nuevo con e5-large, por
si aportan cosas distintas.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
pd.set_option("display.width", 210)
CACHE = Path("data/processed/embeddings/codificadores")
OUT = Path("results/codificadores_2026")
S = "=" * 118
INSTRUCCION = ("Instruct: Represent this software development task for estimating "
               "how many hours it will take\nQuery: ")
MODELOS = {
    "e5-large-instruct": ("intfloat/multilingual-e5-large-instruct", INSTRUCCION),
    "gte-multilingual": ("Alibaba-NLP/gte-multilingual-base", ""),
    "jina-v3": ("jinaai/jina-embeddings-v3", ""),
}


def metricas(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), .01, None)
    q = np.log(p / y)
    r = np.abs(p - y) / y
    m = float(np.median(q))
    return dict(MAE_h=float(np.abs(p - y).mean()),
                MdAElog=float(np.median(np.abs(q))),
                zeta_pc=float((np.exp(np.median(np.abs(q))) - 1) * 100),
                R2log=float(1 - ((np.log(y) - np.log(p)) ** 2).mean() / np.log(y).var()),
                PRED25=float(np.mean(r <= .25)),
                factor2=float(np.mean(np.maximum(p / y, y / p) <= 2)),
                rho=float(pd.Series(y).corr(pd.Series(p), method="spearman")),
                SSPB_pc=float(np.sign(m) * (np.exp(abs(m)) - 1) * 100))


def codificar(etq, nombre, prefijo, textos):
    f = CACHE / f"{etq}.npy"
    if f.exists():
        return np.load(f)
    import torch
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    st = SentenceTransformer(nombre, device="cuda", trust_remote_code=True)
    st.max_seq_length = 64
    v = st.encode([prefijo + t for t in textos], batch_size=32,
                  show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    np.save(f, v)
    print(f"  {etq:22s} {v.shape} en {time.time() - t0:.0f} s", flush=True)
    del st
    torch.cuda.empty_cache()
    return v


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[df.is_software_task == 1].copy()
    sw = sw[~sw.fuente_jira.eq("Apache")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str).tolist()
    y = sw.effort_hours.to_numpy()
    ly = np.log(y)
    tr = (sw.split_t3 == "train").to_numpy()
    bl = {"validacion": (sw.split_t3 == "val").to_numpy(),
          "prueba fija": (sw.split_t3 == "test").to_numpy(),
          "prueba SiP": ((sw.split_t3 == "test") & (sw.source == "SiP")).to_numpy(),
          "externo no Apache": (sw.split_t3 == "externo").to_numpy()}

    vistas = {"e5-large (el actual)": np.load(CACHE / "e5-large.npy"),
              "qwen3-0.6B (de E4)": np.load(CACHE / "qwen3-0.6B.npy")}
    for etq, (nombre, pre) in MODELOS.items():
        try:
            vistas[etq] = codificar(etq, nombre, pre, tit)
        except Exception as e:
            print(f"  {etq:22s} FALLO: {type(e).__name__}: {str(e)[:110]}", flush=True)

    def hgb():
        return HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=.06,
                                             l2_regularization=1., random_state=0)

    filas = {}
    for etq, V in vistas.items():
        p = np.exp(hgb().fit(V[tr], ly[tr]).predict(V))
        filas[etq] = p
        print(f"  {etq:22s} evaluado", flush=True)
    # concatenacion del mejor nuevo con el actual
    nuevos = [k for k in vistas if k not in ("e5-large (el actual)", "qwen3-0.6B (de E4)")]
    if nuevos:
        val = {k: metricas(y[bl["validacion"]], filas[k][bl["validacion"]])["R2log"]
               for k in nuevos}
        mejor = max(val, key=val.get)
        C = np.hstack([vistas["e5-large (el actual)"], vistas[mejor]])
        filas[f"e5-large + {mejor}"] = np.exp(hgb().fit(C[tr], ly[tr]).predict(C))
        print(f"  concatenado con {mejor} (mejor en validacion) evaluado", flush=True)

    t = pd.DataFrame([dict(bloque=b, codificador=k, **metricas(y[m], p[m]))
                      for b, m in bl.items() for k, p in filas.items()])
    cols = ["codificador", "MAE_h", "MdAElog", "zeta_pc", "R2log", "PRED25",
            "factor2", "rho"]
    for b in bl:
        print(f"\n{S}\n{b}  (el MISMO HGB sobre cada juego de vectores)\n{S}")
        print(t[t.bloque == b][cols].sort_values("R2log", ascending=False)
              .round(4).to_string(index=False))
    t.to_csv(OUT / "resumen.csv", index=False)
    json.dump({"instruccion": INSTRUCCION}, open(OUT / "meta.json", "w",
                                                 encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\nescrito {OUT}/resumen.csv")


if __name__ == "__main__":
    main()
