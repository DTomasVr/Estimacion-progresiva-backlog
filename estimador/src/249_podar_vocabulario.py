"""Poda del vocabulario de los tres e5-large, encima del int8 de 246.

De cada codificador, la tabla de embeddings de palabra son 250.002 x 1024: casi
la mitad de los parametros, y los titulos usan una fraccion. Se conservan solo
los tokens que aparecen; el resto se manda a <unk>.

Para medirlo honestamente el vocabulario se saca de las tareas que NO estan en
el bloque de prueba (split_t3 != test, Apache incluido), y se evalua sobre el
bloque de prueba: asi entra en la cuenta el coste real de toparse con un token
que se podo. Se mide tambien la cobertura sobre las 199 tareas del generador.

El tokenizador no se toca: se le pone delante una tabla id_viejo -> id_nuevo.
Los ids 0-3 (<s>, <pad>, </s>, <unk>) se quedan en su sitio, porque XLM-R
calcula las posiciones a partir del id de <pad> = 1.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
OUT = Path("results/poda_vocabulario")
DIR = Path("models/onnx_podado")
spec = importlib.util.spec_from_file_location("i8", "src/246_codificadores_int8.py")
i8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(i8)


class CodPodado(i8.CodOnnx):
    def __init__(self, onnx, ruta, maxlen, norma, lut):
        super().__init__(onnx, ruta, maxlen, norma)
        self.lut = lut

    def encode(self, textos, batch_size=64, normalize_embeddings=False, **_):
        out = []
        for i in range(0, len(textos), batch_size):
            t = self.tok(textos[i:i + batch_size], padding=True, truncation=True,
                         max_length=self.maxlen, return_tensors="np")
            h = self.s.run(None, {"input_ids": self.lut[t["input_ids"]].astype(np.int64),
                                  "attention_mask": t["attention_mask"].astype(np.int64)})[0]
            m = t["attention_mask"][..., None].astype(np.float32)
            e = (h * m).sum(1) / np.maximum(m.sum(1), 1e-9)
            if self.norma:
                e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
            out.append(e.astype(np.float32))
        return np.vstack(out)


def ids_de(tok, textos):
    s = set()
    for i in range(0, len(textos), 4000):
        for x in tok(textos[i:i + 4000])["input_ids"]:
            s.update(x)
    return s


def exporta(nombre, ruta, keep):
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel
    q = DIR / f"{nombre}.podado.int8.onnx"
    if q.exists():
        return q
    f32 = DIR / f"{nombre}.fp32" / "model.onnx"
    f32.parent.mkdir(parents=True, exist_ok=True)
    base = AutoModel.from_pretrained(ruta).eval()
    W = base.embeddings.word_embeddings.weight.data[torch.tensor(keep)]
    emb = torch.nn.Embedding(len(keep), W.shape[1], padding_idx=1)
    emb.weight.data.copy_(W)
    base.embeddings.word_embeddings = emb
    base.config.vocab_size = len(keep)

    class Envoltura(torch.nn.Module):
        def __init__(self, b):
            super().__init__()
            self.b = b

        def forward(self, input_ids, attention_mask):
            return self.b(input_ids=input_ids, attention_mask=attention_mask,
                          return_dict=True).last_hidden_state

    m = Envoltura(base)
    x = torch.tensor([[0, 5, 6, 2]])
    with torch.no_grad():
        torch.onnx.export(
            m, (x, torch.ones_like(x)), str(f32),
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


def main():
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/pbe_effort_v4.parquet")
    q = lambda s: ["query: " + t for t in s.fillna("").astype(str)]
    tok = AutoTokenizer.from_pretrained(i8.COD[0][1])
    V = len(tok)
    fuera = ids_de(tok, q(df[df.split_t3 != "test"].title))
    keep = np.array(sorted(fuera | {0, 1, 2, 3}))
    lut = np.full(V, 3, np.int64)
    lut[keep] = np.arange(len(keep))

    sw = df[(df.is_software_task == 1) & ~df.fuente_jira.eq("Apache")
            & (df.split_t3 == "test")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str).tolist()
    y = sw.effort_hours.to_numpy(float)

    def cobertura(textos):
        ids = tok(textos)["input_ids"]
        tot = sum(len(x) for x in ids)
        oov = sum(sum(lut[i] == 3 and i != 3 for i in x) for x in ids)
        tit_oov = np.mean([any(lut[i] == 3 and i != 3 for i in x) for x in ids])
        return oov / tot, float(tit_oov)

    gen = json.load(open("results/prueba_extremo/salidas.json", encoding="utf-8"))
    tg = ["query: " + t for c in gen for h in c["fp32"]["historias"] for t in h["tareas"]]
    c_test = cobertura(["query: " + t for t in tit])
    c_gen = cobertura(tg)
    print(f"vocabulario {V:,} -> {len(keep):,} ({len(keep) / V:.1%})\n"
          f"  tokens fuera del vocabulario: prueba {c_test[0]:.3%} "
          f"({c_test[1]:.1%} de los titulos tienen alguno) · generador {c_gen[0]:.3%} "
          f"({c_gen[1]:.1%})", flush=True)

    for nombre, ruta, _, _ in i8.COD:
        f = exporta(nombre, ruta, keep)
        print(f"  {nombre}: {f.stat().st_size / 1e6:.0f} MB · {time.time() - t0:.0f} s",
              flush=True)

    est = i8.carga_est()
    for (n, ruta, ml, nz), attr in zip(i8.COD, ("_st", "_aj", "_ac")):
        setattr(est, attr, CodPodado(DIR / f"{n}.podado.int8.onnx", ruta, ml, nz, lut))
    b = est.estimar_tareas(tit)[["horas", "valor_probable"]]
    b.to_parquet(OUT / "podado.parquet")
    a = pd.read_parquet("results/codificadores_int8/fp32.parquet")
    c8 = pd.read_parquet("results/codificadores_int8/int8.parquet")
    filas = []
    for c in ("horas", "valor_probable"):
        for v, d in (("fp32", a), ("int8", c8), ("int8 podado", b)):
            filas.append({"salida": c, "version": v, **i8.met(y, d[c])})
    r = pd.DataFrame(filas)
    r.to_csv(OUT / "metricas.csv", index=False)
    print(r.round(4).to_string(index=False))
    res = {"vocab": V, "conservados": int(len(keep)),
           "oov_tokens_prueba": c_test[0], "titulos_con_oov_prueba": c_test[1],
           "oov_tokens_generador": c_gen[0], "titulos_con_oov_generador": c_gen[1],
           "desvio_horas_vs_int8_mediano": float(np.median(np.abs(
               b.horas.to_numpy() / c8.horas.to_numpy() - 1))),
           "mb": {n: (DIR / f"{n}.podado.int8.onnx").stat().st_size / 1e6
                  for n, *_ in i8.COD}}
    json.dump(res, open(OUT / "resumen.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print(f"· {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
