"""Serverless deployment of the pipeline on Modal: pay per second of compute,
nothing while idle.

Why this shape:
  * the models (SBERT + two mT5 + the two translators) are baked into the image,
    not downloaded at runtime -- otherwise every cold start pays the download
  * @modal.enter loads them once per container, so warm requests only pay
    generation (~5 s GPU, ~13 s CPU)
  * scaledown_window keeps a container alive a few minutes: a user creating
    several projects in a row never pays a cold start
  * cpu=... is the cheaper deployment; measured at 13 s per backlog on 4 threads,
    which is faster than an 8B LLM on a dedicated GPU (21.6 s)

    pip install modal && modal setup
    modal deploy src/34_modal_app.py          # CPU (cheapest)
    PBE_GPU=1 modal deploy src/34_modal_app.py   # with an L4

Only the model folders and the retrieval tables are added to the image; the
training data and the corpora stay out.
"""
import os

import modal

USA_GPU = os.environ.get("PBE_GPU") == "1"
LOCAL = "F:/Software Task generation Model"
imagen = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.11.0", "transformers==5.16.1", "sentence-transformers==6.0.1",
                 "pandas", "pyarrow", "sentencepiece", "protobuf")
    .add_local_dir(f"{LOCAL}/models/sbert_pbe_v1", "/modelos/sbert_pbe_v1", copy=True)
    .add_local_dir(f"{LOCAL}/models/mt5-small-pruned_hist_pbe_v1", "/modelos/abajo", copy=True)
    .add_local_dir(f"{LOCAL}/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1", "/modelos/arriba", copy=True)
    .add_local_file(f"{LOCAL}/src/19_pipeline.py", "/app/pipeline.py", copy=True)
    .add_local_file(f"{LOCAL}/data/processed/pipeline_emb.npz", "/datos/pipeline_emb.npz", copy=True)
    .add_local_file(f"{LOCAL}/data/processed/s2s_o2o_train.parquet", "/datos/s2s_o2o_train.parquet", copy=True)
    .add_local_file(f"{LOCAL}/data/processed/s2s_hist_train.parquet", "/datos/s2s_hist_train.parquet", copy=True)
)
app = modal.App("pbe-generador", image=imagen)


@app.cls(gpu="L4" if USA_GPU else None, cpu=4, memory=8192,
         scaledown_window=300,      # stay warm 5 min: bursts of projects skip the cold start
         min_containers=0)          # 0 = nothing is paid while nobody generates
class Generador:
    @modal.enter()
    def cargar(self):
        import importlib.util
        import sys
        os.environ["PBE_DEVICE"] = "cuda" if USA_GPU else "cpu"
        sys.path.insert(0, "/app")
        spec = importlib.util.spec_from_file_location("pl", "/app/pipeline.py")
        pl = importlib.util.module_from_spec(spec)
        # the paths inside the image differ from the local ones
        codigo = open("/app/pipeline.py", encoding="utf-8").read()
        codigo = (codigo.replace(r'BASE = r"F:\Software Task generation Model"', 'BASE = "/app"')
                  .replace('f"{BASE}/models/sbert_pbe_v1"', '"/modelos/sbert_pbe_v1"')
                  .replace('f"{BASE}/models/mt5-small-pruned_hist_pbe_v1"', '"/modelos/abajo"')
                  .replace('f"{BASE}/models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1"', '"/modelos/arriba"')
                  .replace('f"{BASE}/data/processed/pipeline_emb.npz"', '"/datos/pipeline_emb.npz"')
                  .replace('f"{BASE}/data/processed/s2s_o2o_train.parquet"', '"/datos/s2s_o2o_train.parquet"')
                  .replace('f"{BASE}/data/processed/s2s_hist_train.parquet"', '"/datos/s2s_hist_train.parquet"')
                  .replace('CONJUNTO = "dos_modelos"', 'CONJUNTO = "no"'))   # one generator in production
        exec(compile(codigo, "pipeline.py", "exec"), pl.__dict__)
        self.pl = pl
        self.p = pl.Pipeline(traducir=True)

    @modal.fastapi_endpoint(method="POST")
    def generar(self, peticion: dict):
        d = (peticion.get("descripcion") or "").strip()
        if len(d) < 15:
            return {"error": "descripcion demasiado corta"}
        return self.p(d[:1200])


@app.local_entrypoint()
def probar():
    g = Generador()
    r = g.generar.remote({"descripcion": "Una aplicación web para reservar servicios de "
                                         "limpieza a domicilio con pagos en la app."})
    print(r["tiempos"], len(r["historias"]), "historias")
