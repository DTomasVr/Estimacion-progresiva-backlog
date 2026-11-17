"""Flags dataset: compress each task's 768-d vector into category flags + hours.

The idea being tested: the multilingual vector already "classifies" a task, so a
model may not need the vector at all -- only the categories it implies. This
script turns every vector into binary flags in three independent ways and writes
one clean dataset (task, block, hours, flags) that the next script trains on
without ever touching the vectors again.

  F1  grupo_*     k-means clusters of the vectors; k chosen on validation
  F2  catpred_*   a classifier on the vector predicts the harmonised task kind
                  (bug, feature, task...) -- the categoriser stage of the
                  original PBE pipeline design
  F3  tipo_* / area_*
                  zero-shot semantic flags: cosine to short category descriptions
                  written in Spanish AND English and averaged, so they work the
                  same on PBE's Spanish tasks. One work-type flag (argmax) plus
                  every technical-area flag whose standardised similarity passes
                  a threshold chosen on validation (0, 1 or several areas)
  F4  catreal_*   the source's own task kind, as a "perfect category" reference

Everything that is fitted -- cluster centres, the classifier, similarity means and
spreads -- is fitted on the training block only. Selection of k and of the area
threshold uses the validation block only. Test blocks are never read here except
to be written into the file.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, "src")
import exp_common as E  # noqa: E402

warnings.filterwarnings("ignore")
DATA = "data/processed/pbe_effort_v4.parquet"
EMB = "data/processed/embeddings/v4_mpnet_congelado"
MODELO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OUT = Path("data/processed/flags")
RES = Path("results/flags")

TIPOS = {
    "error": ("corregir un error o fallo del sistema", "fix a bug, crash or incorrect behaviour"),
    "funcionalidad": ("implementar una nueva funcionalidad", "implement a new feature"),
    "mejora": ("mejorar una funcionalidad existente", "improve or enhance an existing feature"),
    "refactorizacion": ("refactorizar o limpiar el código", "refactor or clean up the code"),
    "documentacion": ("escribir o actualizar documentación", "write or update documentation"),
    "pruebas": ("escribir o arreglar pruebas automatizadas", "write or fix automated tests"),
    "despliegue": ("configurar, desplegar o publicar una versión", "configure, deploy or release a version"),
    "investigacion": ("investigar, analizar o diseñar una solución", "investigate, analyse or design a solution"),
    "soporte": ("atender una consulta o dar soporte a un usuario", "answer a question or support a user"),
    "dependencias": ("actualizar dependencias o librerías", "upgrade dependencies or libraries"),
}
AREAS = {
    "interfaz": ("interfaz de usuario, pantallas y botones", "user interface, screens, buttons and layout"),
    "base_datos": ("base de datos, consultas SQL y tablas", "database, SQL queries, tables and schema"),
    "api": ("API, servicios web e integraciones", "API, web services, endpoints and integrations"),
    "seguridad": ("seguridad, autenticación y permisos", "security, authentication, login and permissions"),
    "rendimiento": ("rendimiento, lentitud y consumo de memoria", "performance, slowness and memory usage"),
    "infraestructura": ("compilación, integración continua y servidores", "build, continuous integration, servers and infrastructure"),
    "datos": ("informes, exportación e importación de datos", "reports, data export and import"),
    "movil": ("aplicación móvil Android o iOS", "mobile app on Android or iOS"),
}


def onehot(etiquetas, prefijo, categorias):
    return pd.DataFrame({f"{prefijo}{c}": (etiquetas == c).astype(np.int8)
                         for c in categorias})


def mdmre_val(F, y, tr, va):
    """Validation MdMRE of a plain ridge on the flags: the selection yardstick."""
    m = Ridge(alpha=1.0).fit(F[tr], np.log(y[tr]))
    return E.point_metrics(y[va], np.exp(m.predict(F[va])))["MdMRE"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(DATA)
    sw = df[(df.is_software_task == 1) & df.split_t3.isin(["train", "val", "test", "externo"])].copy()
    sw = sw.reset_index(drop=True)
    emb = np.load(f"{EMB}.npy")
    ids = np.load(f"{EMB}.ids.npy", allow_pickle=True)
    pos = {t: i for i, t in enumerate(ids)}
    X = emb[[pos[t] for t in sw.task_id]].astype(np.float32)
    sw["bloque"] = sw.split_t3.map({"train": "train", "val": "val",
                                    "test": "prueba_d1", "externo": "externo"})
    tr = (sw.bloque == "train").to_numpy()
    va = (sw.bloque == "val").to_numpy()
    y = sw.effort_hours.to_numpy()
    print(f"tareas {len(sw):,} · " + " · ".join(f"{b} {int((sw.bloque == b).sum()):,}"
                                                  for b in ("train", "val", "prueba_d1", "externo")))
    info, seleccion = {}, []

    # ------------------------------------------------------------ F1 grupos
    t0 = time.time()
    mejor = None
    for k in (8, 16, 32, 64, 128):
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=8192).fit(X[tr])
        lab = km.predict(X)
        F = onehot(lab, "g", range(k)).to_numpy()
        v = mdmre_val(F, y, tr, va)
        seleccion.append(dict(familia="F1 grupos", parametro=f"k={k}", MdMRE_val=round(v, 4)))
        print(f"  F1 k={k:3d}  MdMRE val {v:.4f}")
        if mejor is None or v < mejor[0] - 0.002:          # empate -> k menor
            mejor = (v, k, km, lab)
    _, k, km, lab = mejor
    grupos = pd.DataFrame((lab[:, None] == np.arange(k)).astype(np.int8),
                          columns=[f"grupo_{i:03d}" for i in range(k)])
    dist = km.transform(X[tr])
    titulos = sw.title[tr].to_numpy()
    info["F1"] = {"k": int(k), "segundos": round(time.time() - t0, 1),
                  "ejemplos_por_grupo": {f"grupo_{i:03d}": [str(t)[:90] for t in titulos[np.argsort(dist[:, i])[:3]]]
                                         for i in range(k)}}

    # ------------------------------------------------------------ F2 categoria predicha
    t0 = time.time()
    cats = sorted(sw.task_kind[tr].unique())
    clf = LogisticRegression(max_iter=2000, C=2.0).fit(X[tr], sw.task_kind[tr])
    pred = clf.predict(X)
    acc = accuracy_score(sw.task_kind[va], pred[va])
    f1 = f1_score(sw.task_kind[va], pred[va], average="macro")
    catpred = onehot(pred, "catpred_", cats)
    v = mdmre_val(catpred.to_numpy(), y, tr, va)
    seleccion.append(dict(familia="F2 categoria predicha", parametro=f"acc={acc:.3f} f1={f1:.3f}",
                          MdMRE_val=round(v, 4)))
    print(f"  F2 categorizador: exactitud val {acc:.3f} · F1 macro {f1:.3f} · MdMRE val {v:.4f}")
    info["F2"] = {"categorias": cats, "exactitud_val": acc, "f1_macro_val": f1,
                  "segundos": round(time.time() - t0, 1)}

    # ------------------------------------------------------------ F3 flags semanticos
    t0 = time.time()
    import torch
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(MODELO, device="cuda" if torch.cuda.is_available() else "cpu")

    def prototipos(dic):
        v = st.encode([t for par in dic.values() for t in par], normalize_embeddings=True)
        v = v.reshape(len(dic), 2, -1).mean(1)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    PT, PA = prototipos(TIPOS), prototipos(AREAS)
    s_tipo, s_area = X @ PT.T, X @ PA.T
    tipo = np.array(list(TIPOS))[s_tipo.argmax(1)]
    tipos = onehot(tipo, "tipo_", list(TIPOS))
    z = (s_area - s_area[tr].mean(0)) / s_area[tr].std(0)
    mejor = None
    for umbral in (0.5, 1.0, 1.5, 2.0):
        areas = pd.DataFrame((z > umbral).astype(np.int8), columns=[f"area_{a}" for a in AREAS])
        F = pd.concat([tipos, areas], axis=1).to_numpy()
        v = mdmre_val(F, y, tr, va)
        seleccion.append(dict(familia="F3 flags semanticos", parametro=f"z>{umbral}", MdMRE_val=round(v, 4)))
        print(f"  F3 umbral z>{umbral}  MdMRE val {v:.4f}  areas por tarea {areas.sum(1).mean():.2f}")
        if mejor is None or v < mejor[0] - 0.002:
            mejor = (v, umbral, areas)
    _, umbral, areas = mejor
    info["F3"] = {"umbral_z": umbral, "tipos": {k: list(v) for k, v in TIPOS.items()},
                  "areas": {k: list(v) for k, v in AREAS.items()},
                  "prevalencia_tipo": tipos[tr].mean().round(4).to_dict(),
                  "prevalencia_area": areas[tr].mean().round(4).to_dict(),
                  "segundos": round(time.time() - t0, 1)}

    # ------------------------------------------------------------ F4 categoria real
    catreal = onehot(sw.task_kind.to_numpy(), "catreal_", cats)

    flags = pd.concat([grupos, catpred, tipos, areas, catreal], axis=1)
    salida = pd.concat([sw[["task_id", "bloque", "source", "project", "effort_hours"]],
                        flags], axis=1)
    salida.to_parquet(OUT / "pbe_flags.parquet", index=False)
    salida.to_csv(OUT / "pbe_flags.csv.gz", index=False, compression="gzip")
    json.dump(info, open(OUT / "diccionario_flags.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    pd.DataFrame(seleccion).to_csv(RES / "seleccion_flags.csv", index=False)

    tam_v = X.nbytes / 1e6
    tam_f = (OUT / "pbe_flags.parquet").stat().st_size / 1e6
    print(f"\nescrito {OUT}/pbe_flags.parquet: {len(salida):,} filas × {flags.shape[1]} flags")
    print(f"  F1 {grupos.shape[1]} · F2 {catpred.shape[1]} · F3 {tipos.shape[1] + areas.shape[1]} · F4 {catreal.shape[1]}")
    print(f"  vectores en memoria {tam_v:,.0f} MB  ->  dataset de flags {tam_f:,.1f} MB en disco")
    ej = salida[salida.bloque == "train"].iloc[0]
    print("  ejemplo:", sw.title.iloc[0][:70], "->",
          [c for c in flags.columns if ej[c] == 1])


if __name__ == "__main__":
    main()
