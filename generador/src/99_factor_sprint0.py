"""Factor de escala de las plantillas del Sprint 0, medido contra tareas fundacionales REALES
no vistas. Para cada plantilla se buscan sus vecinas reales mas parecidas (SBERT del
estimador no: SBERT propio del generador) y se compara lo que el modelo dice de la plantilla
con lo que tardaron de verdad esas vecinas. Asi el factor es por tipo de trabajo, no global."""
import importlib.util, re, sys, json
from pathlib import Path
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
s = importlib.util.spec_from_file_location("e6", "src/171_estimador_v6.py")
e6 = importlib.util.module_from_spec(s); s.loader.exec_module(e6)
est = e6.EstimadorV6(paquete=Path("models/estimador_pbe_v7.joblib"), dispositivo="cpu")
sys.path.insert(0, r"F:\Software Task generation Model\src")
spec = importlib.util.spec_from_file_location("pl", r"F:\Software Task generation Model\src\19_pipeline.py")
pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
plantillas = [a for h in pl._HABILITADORES for a, _ in h["tareas"]]
F = re.compile(r"\b(set ?up|setup|bootstrap|scaffold|configure|integrate)\b|\b(ci|cd|deploy|deployment|docker|pipeline|staging)\b"
               r"|\bdatabase\b|\bschema\b|\bmigrations?\b|\b(authentication|login|sign ?up|registration|password|oauth|jwt|permissions?)\b"
               r"|\b(payment|gateway|notification|push|email service|storage|upload|map|geolocation|real.?time|websocket|report)\b", re.I)
df = pd.read_parquet("data/processed/pbe_effort_v4.parquet", columns=["title", "effort_hours", "split"])
df = df[df.split.isin(["test", "val"]) & df.title.map(lambda t: bool(F.search(str(t))))].reset_index(drop=True)
print(f"tareas reales de base tecnica no vistas: {len(df)}")
from sentence_transformers import SentenceTransformer
sb = SentenceTransformer(r"F:\Software Task generation Model\models\sbert_pbe_v1", device="cpu")
Ep = sb.encode(plantillas, normalize_embeddings=True, show_progress_bar=False)
Er = sb.encode(df.title.astype(str).tolist(), normalize_embeddings=True, show_progress_bar=False)
S = Ep @ Er.T
K = 15
media_p = np.asarray(est._media_tarea(plantillas), float)
med_p = est.estimar_tareas(plantillas, estilo="persona")["horas"].to_numpy(float)
filas = []
for i, t in enumerate(plantillas):
    j = np.argsort(-S[i])[:K]
    real = df.effort_hours.to_numpy(float)[j]
    filas.append((t, media_p[i], real.mean(), med_p[i], np.median(real), S[i, j].mean()))
R = pd.DataFrame(filas, columns=["plantilla", "media_modelo", "media_real_vecinas", "mediana_modelo", "mediana_real_vecinas", "sim"])
f_media = R.media_real_vecinas.sum() / R.media_modelo.sum()
f_mediana = np.exp(np.mean(np.log(R.mediana_real_vecinas / R.mediana_modelo)))
print(f"\n{K} vecinas reales por plantilla · parecido medio {R.sim.mean():.2f}")
print(f"cabeza B (media, la que suma el total): modelo {R.media_modelo.mean():.2f} h · real {R.media_real_vecinas.mean():.2f} h"
      f"  ->  factor {f_media:.3f}")
print(f"cabeza A (mediana, la que se ensena):   modelo {np.median(R.mediana_modelo):.2f} h · real {np.median(R.mediana_real_vecinas):.2f} h"
      f"  ->  factor {f_mediana:.3f}")
rng = np.random.default_rng(0)
bs = [R.sample(len(R), replace=True, random_state=int(rng.integers(1e9))) for _ in range(2000)]
fb = [b.media_real_vecinas.sum() / b.media_modelo.sum() for b in bs]
print(f"IC95 del factor de la media (remuestreando plantillas): [{np.quantile(fb,.025):.3f}, {np.quantile(fb,.975):.3f}]")
print("\npor plantilla (media modelo -> media real de sus vecinas):")
for r in R.sort_values("media_modelo", ascending=False).itertuples():
    print(f"  {r.media_modelo:5.1f} -> {r.media_real_vecinas:5.1f} h   {r.plantilla}")
json.dump({"factor_media": f_media, "factor_mediana": f_mediana, "k_vecinas": K, "n_reales": len(df),
           "ic95_media": [float(np.quantile(fb, .025)), float(np.quantile(fb, .975))]},
          open(sys.argv[1], "w"), indent=1)
