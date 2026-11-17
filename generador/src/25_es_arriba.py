"""Spanish inputs for the upper half.

Agix sends a Spanish kickoff description, but the One2One upper model only ever
saw English inputs, so it drifts off topic ("As a fabric developer..." for a
cleaning-services app). The lower half does not have this problem because 30 %
of its training inputs were translated.

Same recipe here: translate the INPUT with opus-mt-en-es and add it as an extra
example with the SAME real English story as target. Targets are never machine
generated. val/test keep both language versions, tagged, so English performance
and the new Spanish capability are measured separately.

out: data/processed/s2s_o2oes_{train,val,test}.parquet + traducciones_arriba.parquet
"""
import time

import pandas as pd
import sentencepiece as spm
import torch
from transformers import MarianMTModel, MarianTokenizer

BASE = r"F:\Software Task generation Model"
P_TRAIN_ES = 0.5          # half of train also appears in Spanish
SEED = 20260922
MAX_IN, MAX_OUT, EOS = 256, 96, 1
t0 = time.time()

partes = {}
for sp_ in ["train", "val", "test", "dalpiaz_test"]:
    partes[sp_] = pd.read_parquet(f"{BASE}/data/processed/s2s_o2o_{sp_}.parquet")
unicas = pd.concat([d[["parent_uid", "entrada"]] for d in partes.values()]).drop_duplicates("parent_uid")
print(f"[{time.time()-t0:5.0f}s] {len(unicas):,} descripciones unicas a traducir", flush=True)

tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
mt = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es", dtype=torch.float16).cuda().eval()
textos = unicas.entrada.tolist()
orden = sorted(range(len(textos)), key=lambda i: len(textos[i]))      # length-sorted: less padding
trad = [None] * len(textos)
BS = 96
for a in range(0, len(orden), BS):
    sel = orden[a:a + BS]
    x = tok([textos[i] for i in sel], return_tensors="pt", padding=True, truncation=True,
            max_length=160).to("cuda")
    with torch.no_grad():
        y = mt.generate(**x, num_beams=1, max_new_tokens=200)
    for i, t in zip(sel, tok.batch_decode(y, skip_special_tokens=True)):
        trad[i] = t
    if (a // BS) % 20 == 0:
        print(f"    {a+len(sel):>6,}/{len(textos):,}  {time.time()-t0:5.0f}s", flush=True)
unicas["entrada_es"] = trad
unicas.to_parquet(f"{BASE}/data/processed/traducciones_arriba.parquet", index=False)
es = dict(zip(unicas.parent_uid, unicas.entrada_es))
del mt
torch.cuda.empty_cache()
print(f"[{time.time()-t0:5.0f}s] traduccion lista; ejemplo: {trad[len(trad)//2][:120]}", flush=True)

sp = spm.SentencePieceProcessor(model_file=f"{BASE}/models/pruned_tok/spiece.model")
PREF = {"issue": "feature -> issue: ", "historia": "feature -> user story: "}

for nombre, d in partes.items():
    d = d.copy()
    d["idioma"] = "en"
    copia = d.sample(frac=P_TRAIN_ES, random_state=SEED) if nombre == "train" else d
    copia = copia.copy()
    copia["idioma"] = "es"
    copia["entrada"] = copia.parent_uid.map(es)
    copia = copia[copia.entrada.notna()]
    # no retrieved examples in the prompt: measured useless in 24_calidad_arriba.py
    copia["src"] = [PREF[e] + t + " | examples: " for e, t in zip(copia.estilo, copia.entrada)]
    d["src"] = [PREF[e] + t + " | examples: " for e, t in zip(d.estilo, d.entrada)]
    j = pd.concat([d, copia], ignore_index=True)
    j["input_ids"] = [x[:MAX_IN - 1] + [EOS] for x in sp.encode(j.src.tolist())]
    j["labels"] = [x[:MAX_OUT - 1] + [EOS] for x in sp.encode(j.objetivo.tolist())]
    salida = "o2oes_" + ("dalpiaz_test" if nombre == "dalpiaz_test" else nombre)
    j.to_parquet(f"{BASE}/data/processed/s2s_{salida}.parquet", index=False)
    print(f"  {nombre:<13} {len(d):>7,} en + {len(copia):>7,} es = {len(j):>7,}", flush=True)

# style stage: only rows whose TARGET is a real user story
import re  # noqa: E402
PAT = re.compile(r"as\s+(?:an?|the)\s+[^,.]{2,60}[,;]?\s+i\s*(?:'d|\s+would)?\s*"
                 r"(?:want|would\s+like|need|wish|should|must|am\s+able)", re.I)
for nombre in ["train", "val"]:
    j = pd.read_parquet(f"{BASE}/data/processed/s2s_o2oes_{nombre}.parquet")
    j = j[j.objetivo.str.match(PAT)]
    if nombre == "train":
        j = pd.concat([j] * 2, ignore_index=True)
    j.to_parquet(f"{BASE}/data/processed/s2s_o2oesfmt_{nombre}.parquet", index=False)
    print(f"  estilo {nombre}: {len(j):,} ({j.idioma.value_counts().to_dict()})")
pd.read_parquet(f"{BASE}/data/processed/s2s_o2oes_test.parquet").to_parquet(
    f"{BASE}/data/processed/s2s_o2oesfmt_test.parquet", index=False)
print(f"TOTAL {(time.time()-t0)/60:.1f} min")
