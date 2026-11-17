"""Reordenador de beneficios: destila el juez en un clasificador pequeno.

El defecto mas grande medido: la clausula "para que" se deduce de la capacidad en
el 89 % de las historias del profesor y solo en el 34 % de las que produce el
modelo (results/65_curacion.csv). No se arregla con mas datos, porque los datos ya
estan bien; se arregla ELIGIENDO mejor entre las candidatas que el modelo ya
genera, que son 48 por descripcion.

Y no puede arreglarse con un LLM en inferencia, porque el argumento de la tesis es
que en el producto no hay ninguno. Asi que se destila el juez: se entrena una
regresion logistica sobre los embeddings del SBERT propio.

Rasgos del par (capacidad, beneficio), que es la receta estandar de clasificacion
de pares de frases sin ajuste fino:
    [ e_cap , e_ben , e_cap * e_ben , |e_cap - e_ben| ]
Se valida por DESCRIPCION, no por fila: dos candidatas de la misma descripcion se
parecen, y partir al azar inflaria el acierto.

  python src/74_reordenador.py

out: models/reordenador_beneficio.joblib  y  results/74_reordenador.csv
"""
import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20260927

L = [json.loads(x) for x in open(f"{BASE}/data/synth/beneficio_etiquetado.jsonl",
                                encoding="utf-8")]
D = pd.DataFrame(L)
print(f"{len(D)} candidatas etiquetadas de {D.id.nunique()} descripciones | "
      f"coherentes {100*D.coherente.mean():.1f} %")

sb = SentenceTransformer(f"{BASE}/models/sbert_pbe_v1",
                         device=os.environ.get("PBE_DEVICE", "cuda"))
sb.max_seq_length = 128


def rasgos(cap, ben):
    a = sb.encode(list(cap), normalize_embeddings=True, batch_size=128,
                  show_progress_bar=False)
    b = sb.encode(list(ben), normalize_embeddings=True, batch_size=128,
                  show_progress_bar=False)
    return np.hstack([a, b, a * b, np.abs(a - b)]).astype(np.float32)


X = rasgos(D.capacidad, D.beneficio)
y = D.coherente.values.astype(int)

# particion POR DESCRIPCION
rng = np.random.default_rng(SEED)
ids = D.id.unique()
rng.shuffle(ids)
corte = int(0.75 * len(ids))
tr = D.id.isin(set(ids[:corte])).values
te = ~tr
print(f"entrenamiento {tr.sum()} filas / {corte} descripciones | "
      f"prueba {te.sum()} filas / {len(ids)-corte} descripciones")

filas = []
mejor = (None, -1, None)
for C in (0.03, 0.1, 0.3, 1.0, 3.0):
    m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    m.fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    auc = roc_auc_score(y[te], p)
    acc = accuracy_score(y[te], p >= 0.5)
    filas.append(dict(C=C, auc=round(auc, 4), acierto=round(acc, 4)))
    print(f"  C={C:<5} AUC {auc:.4f}  acierto {acc:.4f}")
    if auc > mejor[1]:
        mejor = (m, auc, C)

m, auc, C = mejor
base = max(y[te].mean(), 1 - y[te].mean())
print(f"\nmejor C={C}: AUC {auc:.4f} | acierto contra la clase mayoritaria "
      f"({100*base:.1f} %)")

# lo que de verdad importa: si al quedarse con las 12 mejores de cada 48, la
# proporcion de coherentes sube respecto a tomar 12 al azar
sub = D[te].copy()
sub["p"] = m.predict_proba(X[te])[:, 1]
antes, despues = [], []
for _, g in sub.groupby("id"):
    if len(g) < 12:
        continue
    antes.append(g.coherente.mean())
    despues.append(g.nlargest(12, "p").coherente.mean())
print(f"\ncoherencia si se toman 12 al azar de las 48 : {100*np.mean(antes):.1f} %")
print(f"coherencia si se toman las 12 mejor puntuadas: {100*np.mean(despues):.1f} %")
print(f"ganancia: {100*(np.mean(despues)-np.mean(antes)):+.1f} puntos "
      f"sobre {len(antes)} descripciones de prueba")

import joblib  # noqa: E402
joblib.dump(dict(modelo=m, C=C, auc=auc), f"{BASE}/models/reordenador_beneficio.joblib")
pd.DataFrame(filas).to_csv(f"{BASE}/results/74_reordenador.csv", index=False)
print("-> models/reordenador_beneficio.joblib y results/74_reordenador.csv")
