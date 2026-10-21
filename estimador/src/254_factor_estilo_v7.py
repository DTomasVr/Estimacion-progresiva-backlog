"""Re-medir el factor de estilo del generador para el v7 (y el v6, de control).

El factor_estilo del artefacto (0,8313) viene de E74, medido con el modelo de
entonces sobre results/reescritura/comparacion.csv: 800 titulos reales con sus
horas, reescritos al estilo del generador. E-v7 vio que la cabeza A del v7 da
un 17 % menos que la del v6 sobre texto del generador, asi que el factor
heredado ya no corresponde.

Mismo procedimiento que 89_correcciones.py: en cada bloque,
k = exp(mediana(log(real / prediccion_sobre_reescrito))), y el factor es la
media geometrica de los dos. Se mide con el v6 (que deberia dar un valor
cercano al guardado si el procedimiento reproduce), con el v7 fp32 y con el v7
en produccion (int8 podado), y se decide si sirve para recalibrar el v7.
"""
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
OUT = Path("results/estimador_v7")
V7 = Path("models/estimador_agix_v7.joblib")


def mod(n, r):
    s = importlib.util.spec_from_file_location(n, r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def factor(r, pred):
    ks = {}
    for b in r.bloque.unique():
        m = (r.bloque == b).to_numpy()
        ks[b] = float(np.exp(np.median(np.log(r.effort_hours.to_numpy()[m] / pred[m]))))
    return float(np.exp(np.mean(np.log(list(ks.values()))))), ks


def main():
    e6 = mod("e6", "src/171_estimador_v6.py")
    ev = mod("ev", "src/253_evaluar_v7.py")
    po = ev.po
    r = pd.read_csv("results/reescritura/comparacion.csv")
    tx = r.reescrito.fillna("").astype(str).tolist()
    to = r.title.fillna("").astype(str).tolist()
    res = {}
    for nom, paq in (("v6", "models/estimador_agix_v6.joblib"), ("v7", str(V7))):
        est = e6.EstimadorV6(paquete=Path(paq))
        pr = est.estimar_tareas(tx).horas.to_numpy()
        po_ = est.estimar_tareas(to).horas.to_numpy()
        res[nom] = {"factor": factor(r, pr), "sesgo_original": factor(r, po_)}
        if nom == "v7":
            from transformers import AutoTokenizer
            df = pd.read_parquet("data/processed/agix_effort_v4.parquet")
            tok = AutoTokenizer.from_pretrained(ev.BASE)
            keep, lut = ev.vocab_produccion(tok, df)
            est._st = po.CodPodado(Path("models/onnx_podado/e5_base_prod.podado.int8.onnx"),
                                   ev.BASE, 512, True, lut)
            res["v7 int8 podado"] = {
                "factor": factor(r, est.estimar_tareas(tx).horas.to_numpy()),
                "sesgo_original": factor(r, est.estimar_tareas(to).horas.to_numpy())}
        del est
    print(json.dumps(res, indent=1))
    # NO se guarda en el paquete. Con este procedimiento el v6 da 0,584, no el
    # 0,8313 que lleva guardado: el corpus lo reescribio un LLM local imitando el estilo,
    # no del generador actual, y no reproduce la medida original. Un factor que
    # no reproduce no sirve para recalibrar; el v7 hereda el del v6.
    json.dump(res, open(OUT / "factor_estilo.json", "w"), indent=1)


if __name__ == "__main__":
    main()
