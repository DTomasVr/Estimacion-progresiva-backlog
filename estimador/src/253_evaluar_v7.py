"""Evaluacion del v7 contra el v6, y del v7 en su forma de produccion.

Cinco comprobaciones:

  1. regresion: el v6 con la clase refactorizada da EXACTAMENTE lo mismo que antes
  2. bloque de prueba (split_t3 == test, 4.225 tareas): horas, valor probable y
     cobertura de la banda por tarea, v6 contra v7
  3. el v7 con su unico codificador en ONNX int8 y vocabulario podado (corpus
     entero + todo token de alfabeto latino), que es lo que iria a produccion
  4. la escala sobre texto del GENERADOR (las 199 tareas de la prueba de extremo a
     extremo): factor_estilo se midio con el v3 y no se re-mide, asi que la
     mediana v7/v6 tiene que quedar cerca de 1
  5. el total del proyecto: la cabeza B no cambia, asi que tiene que salir igual

Ojo al leer el punto 2: en el bloque de prueba fijo el v6 juega con ventaja,
porque sus codificadores ajustados vieron tareas de los mismos proyectos. La
prueba honesta de proyecto nuevo es la de E120 (fuera de pliegue).
"""
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
OUT = Path("results/estimador_v7")
DIR = Path("models/onnx_podado")
BASE = "intfloat/multilingual-e5-large"


def mod(n, r):
    s = importlib.util.spec_from_file_location(n, r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


e6 = mod("e6", "src/171_estimador_v6.py")
i8 = mod("i8", "src/246_codificadores_int8.py")
po = mod("po", "src/249_podar_vocabulario.py")


def vocab_produccion(tok, df):
    v = tok.convert_ids_to_tokens(list(range(len(tok))))
    lat = re.compile(r"^[▁]?[A-Za-z0-9áéíóúñüÁÉÍÓÚÑÜ'’‘“”–—\-\.,:;/()&%+#_!?\"]+$")
    keep = {i for i, t in enumerate(v) if lat.match(t)} | {0, 1, 2, 3}
    keep |= po.ids_de(tok, ["query: " + x for x in df.title.fillna("").astype(str)])
    keep = np.array(sorted(keep))
    lut = np.full(len(tok), 3, np.int64)
    lut[keep] = np.arange(len(keep))
    return keep, lut


def main():
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
    sw = df[(df.is_software_task == 1) & ~df.fuente_jira.eq("Apache")
            & (df.split_t3 == "test")].reset_index(drop=True)
    tit = sw.title.fillna("").astype(str).tolist()
    y = sw.effort_hours.to_numpy(float)
    gen = json.load(open("results/prueba_extremo/salidas.json", encoding="utf-8"))
    tg = [t for c in gen for h in c["fp32"]["historias"] for t in h["tareas"]]
    cols = ["horas", "valor_probable", "banda_baja", "banda_alta"]

    v6 = e6.EstimadorV6(paquete=Path("models/estimador_agix_v6.joblib"))
    d6 = v6.estimar_tareas(tit)[cols]
    g6 = v6.estimar_tareas(tg, estilo="generador")[cols]
    ref = pd.read_parquet("results/codificadores_int8/fp32.parquet")
    regresion = float(np.max(np.abs(d6.horas.to_numpy() / ref.horas.to_numpy() - 1)))
    cer = [2.0] * 4 + [None] * 6
    pr6 = v6.estimar_proyecto(tg[:10], cerradas=cer)
    del v6
    print(f"v6 · regresion {regresion:.1e} · {time.time() - t0:.0f} s", flush=True)

    v7 = e6.EstimadorV6(paquete=Path("models/estimador_agix_v7.joblib"))
    d7 = v7.estimar_tareas(tit)[cols]
    g7 = v7.estimar_tareas(tg, estilo="generador")[cols]
    pr7 = v7.estimar_proyecto(tg[:10], cerradas=cer)
    print(f"v7 fp32 · {time.time() - t0:.0f} s", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    keep, lut = vocab_produccion(tok, df)
    f = po.exporta("e5_base_prod", BASE, keep)
    v7._st = po.CodPodado(f, BASE, 512, True, lut)
    t1 = time.time()
    d8 = v7.estimar_tareas(tit)[cols]
    ms = 1000 * (time.time() - t1) / len(tit)
    g8 = v7.estimar_tareas(tg, estilo="generador")[cols]
    print(f"v7 int8 podado · {time.time() - t0:.0f} s", flush=True)

    filas = []
    for nom, d in (("v6 (3 codificadores)", d6), ("v7 fp32 (1 codificador)", d7),
                   ("v7 int8 podado", d8)):
        for c in ("horas", "valor_probable"):
            filas.append({"modelo": nom, "salida": c, **i8.met(y, d[c])})
        filas[-1]["cobertura_banda"] = float(np.mean(
            (y >= d.banda_baja) & (y <= d.banda_alta)))
        filas[-1]["ancho_banda_x"] = float(np.median(d.banda_alta / d.banda_baja))
    r = pd.DataFrame(filas)
    r.to_csv(OUT / "prueba.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(4).to_string(index=False))

    res = {
        "regresion_v6_desvio_max": regresion,
        "generador_mediana_v7_sobre_v6": float(np.median(g7.horas / g6.horas)),
        "generador_mediana_v7int8_sobre_v6": float(np.median(g8.horas / g6.horas)),
        "generador_valor_probable_igual_v6_v7": float(np.mean(
            g7.valor_probable.to_numpy() == g6.valor_probable.to_numpy())),
        "total_v6": pr6["total_horas"], "total_v7": pr7["total_horas"],
        "vocab_produccion": int(len(keep)),
        "mb_codificador": f.stat().st_size / 1e6,
        "ms_por_tarea_cpu_int8": ms,
    }
    json.dump(res, open(OUT / "resumen.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    print(f"· {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
