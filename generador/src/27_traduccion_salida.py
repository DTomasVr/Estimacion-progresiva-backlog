"""Is the Spanish the user finally reads faithful to what the model generated?

There are no human reference translations, so quality is measured three ways,
each with a floor or a ceiling so the number means something:

  fidelidad   multilingual-SBERT cosine between the English item and its Spanish
              translation, against a floor built by pairing translations with
              the WRONG source (if the floor were close, the metric would be
              measuring nothing)
  formato     does "As a X, I want Y, so that Z" survive as "Como X, quiero Y,
              para Z" -- the structure is what makes a story readable as one
  terminos    technical tokens (API, OAuth, camelCase, dotted.names) that must
              come through untouched, since translating them breaks the task

Plus wall-clock, because the pipeline's selling point is speed.
"""
import importlib.util
import re
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import MarianMTModel, MarianTokenizer

BASE = r"F:\Software Task generation Model"
spec = importlib.util.spec_from_file_location("pl", f"{BASE}/src/19_pipeline.py")
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)
SEED = 20260922
N_PROY = 30
PAT_ES = re.compile(r"\bcomo\s+[^,]{2,60},?\s+(?:quiero|necesito|deseo|me gustaría|quisiera)", re.I)
TECNICO = re.compile(r"\b(?:[A-Z]{2,}[A-Za-z0-9]*|[a-z]+(?:[A-Z][a-z]+)+|\w+\.\w+|\w+-\w+)\b")


def main():
    t0 = time.time()
    sb = SentenceTransformer(pl.SBERT, device="cuda")
    p = pl.Pipeline()
    D = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_test.parquet")
    D = D[D.estilo == "historia"].groupby("parent_uid").entrada.first()
    D = D.sample(N_PROY, random_state=SEED)
    p(D.iloc[0], k_historias=3)                       # warm-up

    historias, tareas, t_pipe = [], [], []
    for d in D:
        t = time.time()
        r = p(d)
        t_pipe.append(time.time() - t)
        historias += [h["historia"] for h in r["historias"]]
        tareas += [x for h in r["historias"] for x in h["tareas"]]
    print(f"[{time.time()-t0:5.0f}s] {len(historias)} historias y {len(tareas)} tareas "
          f"de {N_PROY} proyectos; pipeline {np.median(t_pipe):.2f}s/proyecto", flush=True)

    tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    mt = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es", dtype=torch.float16).cuda().eval()

    def traducir(textos, bs=64):
        out, tiempos = [], []
        for a in range(0, len(textos), bs):
            lote = textos[a:a + bs]
            t = time.time()
            prot = [pl.proteger(x) for x in lote]
            x = tok([p for p, _ in prot], return_tensors="pt", padding=True, truncation=True,
                    max_length=128).to("cuda")
            with torch.no_grad():
                y = mt.generate(**x, num_beams=1, max_new_tokens=128)
            out += [pl.restaurar(s_, f) for s_, (_, f) in
                    zip(tok.batch_decode(y, skip_special_tokens=True), prot)]
            tiempos.append((time.time() - t) / len(lote))
        return out, float(np.mean(tiempos))

    filas = []
    for nombre, textos in [("historias", historias), ("tareas", tareas)]:
        es, seg = traducir(textos)
        E1 = sb.encode(textos, batch_size=128, convert_to_tensor=True, normalize_embeddings=True,
                       show_progress_bar=False)
        E2 = sb.encode(es, batch_size=128, convert_to_tensor=True, normalize_embeddings=True,
                       show_progress_bar=False)
        fid = (E1 * E2).sum(1).cpu().numpy()
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(textos))
        piso = (E1 * E2[perm]).sum(1).cpu().numpy()          # wrong-pair floor
        fmt_en = np.mean([bool(pl.PAT_HIST.match(t)) for t in textos])
        fmt_es = np.mean([bool(PAT_ES.search(t)) for t in es])
        tec = []
        for a, b in zip(textos, es):
            t_en = set(TECNICO.findall(a))
            if t_en:
                tec.append(len(t_en & set(TECNICO.findall(b))) / len(t_en))
        filas.append(dict(tipo=nombre, n=len(textos), fidelidad=fid.mean(), piso=piso.mean(),
                          fidelidad_p10=np.quantile(fid, .1),
                          formato_en=fmt_en, formato_es=fmt_es,
                          terminos=float(np.mean(tec)) if tec else np.nan, ms_por_item=seg * 1000))
        print(f"  {nombre:<10} fidelidad {fid.mean():.3f} (piso {piso.mean():.3f}, p10 {np.quantile(fid,.1):.3f})  "
              f"formato en {fmt_en:.0%} -> es {fmt_es:.0%}  terminos {np.mean(tec) if tec else 0:.0%}  "
              f"{seg*1000:.1f} ms/item", flush=True)
        if nombre == "historias":
            peores = np.argsort(fid)[:5]
            with open(f"{BASE}/results/27_traduccion_muestras.txt", "w", encoding="utf-8") as fh:
                fh.write("=== 8 al azar\n")
                for i in rng.choice(len(textos), 8, replace=False):
                    fh.write(f"EN: {textos[i]}\nES: {es[i]}\n  fidelidad {fid[i]:.3f}\n\n")
                fh.write("=== 5 peores\n")
                for i in peores:
                    fh.write(f"EN: {textos[i]}\nES: {es[i]}\n  fidelidad {fid[i]:.3f}\n\n")

    R = pd.DataFrame(filas).round(4)
    R.to_csv(f"{BASE}/results/27_traduccion_salida.csv", index=False)
    n_items = len(historias) + len(tareas)
    coste = sum(f["ms_por_item"] * f["n"] for f in filas) / 1000 / N_PROY
    print(f"\ncoste de traducir un backlog completo: {coste:.2f} s "
          f"({n_items/N_PROY:.0f} items por proyecto) sobre {np.median(t_pipe):.2f}s de generacion")
    print(R.to_string(index=False))


if __name__ == "__main__":
    main()
