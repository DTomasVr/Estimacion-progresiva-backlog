"""Los tres codificadores a ONNX int8, y se decide por METRICAS, no por desvio.

fp16 movio las salidas un 1,5 % sobre diez textos y el guardian lo tiro, pero
diez textos no dicen si PRED(25) o el SA cambian. Aqui:

  1. predicciones fp32 (el artefacto tal cual) sobre el bloque split_t3 == test
  2. cada codificador -> ONNX fp32 -> int8 dinamico (pesos de las lineales)
  3. las mismas predicciones con los int8 enchufados en el estimador
  4. PRED25, MdMRE, MAE, SA y spearman de las dos versiones

Los originales NO se tocan: los int8 quedan en models/onnx/. El artefacto solo
cambia de backend si las metricas aguantan.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "src")
OUT = Path("results/codificadores_int8")
DIR = Path("models/onnx")
TOL = .25
# nombre, ruta, max_len, normaliza (el e5 base lleva modulo Normalize)
COD = [("e5_base", "intfloat/multilingual-e5-large", 512, True),
       ("e5_ajustado", "models/e5_ajustado", 48, False),
       ("e5_ajustado_C", "models/e5_ajustado_C", 48, False)]


def met(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, None)
    r = np.abs(p - y) / y
    mar = np.abs(np.median(y) - y).mean()
    return dict(PRED25=float(np.mean(r <= TOL)), MdMRE=float(np.median(r)),
                MAE_h=float(np.abs(p - y).mean()),
                SA=float(1 - np.abs(p - y).mean() / mar),
                spearman=float(spearmanr(p, y).statistic))


class CodOnnx:
    """Sustituto de SentenceTransformer.encode: tokeniza, ORT, media, (norma)."""

    def __init__(self, onnx, ruta, maxlen, norma):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        o = ort.SessionOptions()
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.s = ort.InferenceSession(str(onnx), o, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(ruta)
        self.maxlen, self.norma = maxlen, norma

    def encode(self, textos, batch_size=64, normalize_embeddings=False, **_):
        out = []
        for i in range(0, len(textos), batch_size):
            t = self.tok(textos[i:i + batch_size], padding=True, truncation=True,
                         max_length=self.maxlen, return_tensors="np")
            h = self.s.run(None, {"input_ids": t["input_ids"].astype(np.int64),
                                  "attention_mask": t["attention_mask"].astype(np.int64)})[0]
            m = t["attention_mask"][..., None].astype(np.float32)
            e = (h * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            if self.norma:
                e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            out.append(e.astype(np.float32))
        return np.vstack(out)


def exporta(nombre, ruta):
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel, AutoTokenizer
    q = DIR / f"{nombre}.int8.onnx"
    if q.exists():
        return q
    f32 = DIR / f"{nombre}.fp32" / "model.onnx"
    f32.parent.mkdir(parents=True, exist_ok=True)
    base = AutoModel.from_pretrained(ruta).eval()

    class Envoltura(torch.nn.Module):
        # transformers 5 ya no tiene attention_mask en segunda posicion
        def __init__(self, b):
            super().__init__()
            self.b = b

        def forward(self, input_ids, attention_mask):
            return self.b(input_ids=input_ids, attention_mask=attention_mask,
                          return_dict=True).last_hidden_state

    m = Envoltura(base)
    t = AutoTokenizer.from_pretrained(ruta)(["query: hola mundo"], return_tensors="pt")
    with torch.no_grad():
        torch.onnx.export(
            m, (t["input_ids"], t["attention_mask"]), str(f32),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={"input_ids": {0: "b", 1: "n"},
                          "attention_mask": {0: "b", 1: "n"},
                          "last_hidden_state": {0: "b", 1: "n"}},
            opset_version=17, dynamo=False)
    del m, base
    quantize_dynamic(str(f32), str(q), weight_type=QuantType.QInt8)
    for f in f32.parent.iterdir():
        f.unlink()
    f32.parent.rmdir()
    return q


def carga_est():
    import importlib.util
    spec = importlib.util.spec_from_file_location("e6", "src/171_estimador_v6.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.EstimadorV6()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    sw = df[(df.is_software_task == 1) & ~df.fuente_jira.eq("Apache")
            & (df.split_t3 == "test")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str).tolist()
    y = sw.effort_hours.to_numpy(float)
    print(f"{len(y):,} tareas de prueba", flush=True)

    ref = OUT / "fp32.parquet"
    if not ref.exists():
        est = carga_est()
        est.estimar_tareas(tit)[["horas", "valor_probable"]].to_parquet(ref)
        del est
    a = pd.read_parquet(ref)
    print(f"fp32 hecho · {time.time() - t0:.0f} s", flush=True)

    for nombre, ruta, _, _ in COD:
        q = exporta(nombre, ruta)
        print(f"  {nombre}: {q.stat().st_size / 1e6:.0f} MB · "
              f"{time.time() - t0:.0f} s", flush=True)

    est = carga_est()
    for (nombre, ruta, ml, nz), attr in zip(COD, ("_st", "_aj", "_ac")):
        setattr(est, attr, CodOnnx(DIR / f"{nombre}.int8.onnx", ruta, ml, nz))
    t1 = time.time()
    b = est.estimar_tareas(tit)[["horas", "valor_probable"]]
    seg = time.time() - t1
    b.to_parquet(OUT / "int8.parquet")

    filas = []
    for c in ("horas", "valor_probable"):
        filas.append({"salida": c, "version": "fp32", **met(y, a[c])})
        filas.append({"salida": c, "version": "int8", **met(y, b[c])})
    r = pd.DataFrame(filas)
    r.to_csv(OUT / "metricas.csv", index=False)
    print(r.round(4).to_string(index=False))
    dv = np.abs(b.horas.to_numpy() / a.horas.to_numpy() - 1)
    mismo = float(np.mean(b.valor_probable.to_numpy() == a.valor_probable.to_numpy()))
    res = {"n": len(y), "desvio_horas_mediano": float(np.median(dv)),
           "desvio_horas_p99": float(np.quantile(dv, .99)),
           "valor_probable_identico": mismo,
           "mb_int8": {n: (DIR / f"{n}.int8.onnx").stat().st_size / 1e6 for n, *_ in COD},
           "seg_int8_cpu": seg, "ms_por_tarea_int8_cpu": 1000 * seg / len(y)}
    json.dump(res, open(OUT / "resumen.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print(f"· {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
