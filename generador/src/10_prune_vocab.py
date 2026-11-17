"""Prune mT5's 250k-piece vocabulary to what our en+es software corpus uses.

In mT5 most parameters are the vocabulary (small: ~256M of 300M; base: ~384M
of 580M, input embedding and untied LM head). Dropping unused pieces shrinks
memory for weights, gradients and -- above all -- the logits, which is what
kept mT5-base from fitting in 6.5 GB.

Vocabulary sources (TRAIN projects only, so val/test text never shapes it):
  en: summaries of every issue in train projects + text of hierarchy nodes
      + the seq2seq train sources/targets
  es: parent titles translated for train (sbert_pairs) + PBE-Dataset
      text_es.parquet (90 %; the other 10 % is held out to measure coverage)
Always kept: pad/eos/unk and every single-character piece of Latin scripts,
so an unseen word still decomposes into kept pieces instead of <unk>.

The SentencePiece model itself is edited (pieces removed), so tokenization
of new text only uses kept pieces. Old->new id map rebuilds the embeddings.

Outputs: models/mt5-{small,base}-pruned/  +  data/processed/s2s_pruned_{split}.parquet
"""
import json
import os
import sys
import time
import unicodedata
import zlib
from collections import Counter

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
from sentencepiece import sentencepiece_model_pb2 as pb
from transformers import MT5ForConditionalGeneration, T5Tokenizer

sys.path.insert(0, os.path.dirname(__file__))
from textclean import limpiar  # noqa: E402

BASE = r"F:\Software Task generation Model"
PBE = r"C:\Users\User\Documents\Programacao\PBE-Dataset"
CACHE = f"{BASE}/models/hf_cache"
SP_MODEL = f"{CACHE}/models--google--mt5-small/snapshots/73fb5dbe4756edadc8fbe8c769b0a109493acf7a/spiece.model"
t0 = time.time()


def fase(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


def latino(ch):
    try:
        n = unicodedata.name(ch)
    except ValueError:
        return False
    return ch.isascii() or "LATIN" in n or unicodedata.category(ch)[0] in "PNS"


sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
V = sp.get_piece_size()

# ------------------------------------------------------------------ corpora
S = pd.read_parquet(f"{BASE}/data/processed/split_projects.parquet").set_index("project_uid").split
# titles only from nodes.parquet (descriptions of 2.8M nodes don't fit in RAM next
# to a training run); description text comes from the cleaned SBERT train pairs
N = pd.read_parquet(f"{BASE}/data/interim/nodes.parquet", columns=["project_uid", "summary"])
en = N.summary[N.project_uid.map(S) == "train"].dropna().map(limpiar).tolist()
del N
tr = pd.read_parquet(f"{BASE}/data/processed/s2s_train.parquet", columns=["src", "tareas"])
en += tr.src.tolist() + [" ; ".join(t) for t in tr.tareas]
SP = pd.read_parquet(f"{BASE}/data/processed/sbert_pairs.parquet",
                     columns=["split", "anchor_en", "anchor_es", "positive"])
SPt = SP[SP.split == "train"]
en += SPt.anchor_en.unique().tolist() + SPt.positive.unique().tolist()
es = SP[SP.split == "train"].anchor_es.dropna().unique().tolist()
TE = pd.read_parquet(f"{PBE}/data/processed/text_es.parquet")
hold = TE.task_id.map(lambda s: zlib.crc32(s.encode()) % 10 == 0)
es += TE.text_es[~hold].dropna().tolist()
es_hold = TE.text_es[hold].dropna().tolist()
fase(f"corpus en {len(en):,} textos, es {len(es):,}, es reservado {len(es_hold):,}")

cont = Counter()
for a in range(0, len(en) + len(es), 50_000):
    lote = (en + es)[a:a + 50_000]
    for ids in sp.encode(lote):
        cont.update(ids)
fase(f"piezas vistas {len(cont):,} de {V:,}")

keep = set(cont) | {0, 1, 2}
m0 = pb.ModelProto()
m0.ParseFromString(open(SP_MODEL, "rb").read())
keep |= {i for i, p in enumerate(m0.pieces) if p.type == pb.ModelProto.SentencePiece.BYTE}   # byte_fallback needs all 256
for i in range(V):
    p = sp.id_to_piece(i)
    c = p.lstrip("\u2581")
    if len(c) <= 1 and all(latino(ch) for ch in c):
        keep.add(i)
keep = sorted(keep)
viejo_a_nuevo = {o: n for n, o in enumerate(keep)}
fase(f"vocabulario podado: {len(keep):,} piezas ({100*len(keep)/V:.1f} %)")

# ---------------------------------------------------- edit sentencepiece model
m = pb.ModelProto()
m.ParseFromString(open(SP_MODEL, "rb").read())
kept = set(keep)
nuevas = [p for i, p in enumerate(m.pieces) if i in kept]
del m.pieces[:]
m.pieces.extend(nuevas)
os.makedirs(f"{BASE}/models/pruned_tok", exist_ok=True)
nuevo_sp = f"{BASE}/models/pruned_tok/spiece.model"
open(nuevo_sp, "wb").write(m.SerializeToString())
sp2 = spm.SentencePieceProcessor(model_file=nuevo_sp)
assert sp2.get_piece_size() == len(keep)

# identity check on training text: same pieces, remapped ids
mues = en[:2000] + es[:2000]
iguales = sum([viejo_a_nuevo.get(i, -1) for i in a] == b for a, b in zip(sp.encode(mues), sp2.encode(mues)))
# held-out spanish: unk rate and token inflation vs original
o = sp.encode(es_hold)
n = sp2.encode(es_hold)
unk = sum(x.count(sp2.unk_id()) for x in n) / sum(map(len, n))
infl = sum(map(len, n)) / sum(map(len, o))
cob = dict(identicos_en_corpus=round(iguales / len(mues), 4), unk_es_reservado=round(unk, 5),
           inflacion_tokens_es_reservado=round(infl, 4), vocab=len(keep))
fase(f"cobertura: {cob}")

# HF fast tokenizer: filter the original tokenizer.json vocab with the same ids
# (T5Tokenizer(vocab_file=...) silently ignores the file in transformers 5)
from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("google/mt5-small", cache_dir=CACHE)
j = json.loads(tok.backend_tokenizer.to_str())
assert len(j["model"]["vocab"]) == V
j["model"]["vocab"] = [j["model"]["vocab"][i] for i in keep]
tok._tokenizer = Tokenizer.from_str(json.dumps(j))
tok.save_pretrained(f"{BASE}/models/pruned_tok")
tok = AutoTokenizer.from_pretrained(f"{BASE}/models/pruned_tok")
hf_ok = sum(tok(x, add_special_tokens=False).input_ids == y for x, y in zip(mues, sp2.encode(mues)))
cob["hf_igual_a_sentencepiece"] = round(hf_ok / len(mues), 4)
fase(f"tokenizador HF: {len(tok):,} piezas, igual a sentencepiece en {hf_ok}/{len(mues)}")
assert hf_ok / len(mues) > 0.99

# ----------------------------------------------------------- prune models
idx = torch.tensor(keep)
for nombre in ["google/mt5-small", "google/mt5-base"]:
    mod = MT5ForConditionalGeneration.from_pretrained(nombre, cache_dir=CACHE, dtype=torch.float32)
    antes = sum(p.numel() for p in mod.parameters())
    W = mod.shared.weight.data[idx].clone()
    H = mod.lm_head.weight.data[idx].clone()
    mod.shared = torch.nn.Embedding.from_pretrained(W, freeze=False)
    mod.encoder.embed_tokens = mod.shared
    mod.decoder.embed_tokens = mod.shared
    mod.lm_head = torch.nn.Linear(W.shape[1], len(keep), bias=False)
    mod.lm_head.weight.data = H
    mod.config.vocab_size = len(keep)
    despues = sum(p.numel() for p in mod.parameters())
    d = f"{BASE}/models/{nombre.split('/')[-1]}-pruned"
    mod.save_pretrained(d)
    tok.save_pretrained(d)
    fase(f"{nombre}: {antes/1e6:.0f}M -> {despues/1e6:.0f}M parametros  -> {d}")
    del mod

# ------------------------------------------------------ re-tokenize seq2seq
MAX_IN, MAX_OUT = 256, 192
sep = sp2.encode(";")
eos = 1


def objetivo(tt):
    ids = []
    for k, t in enumerate(tt):
        extra = (sep if k else []) + sp2.encode(t)
        if len(ids) + len(extra) + 1 > MAX_OUT:
            break
        ids += extra
    return ids + [eos]


for split in ["train", "val", "test", "dalpiaz"]:
    d = pd.read_parquet(f"{BASE}/data/processed/s2s_{split}.parquet")
    d["input_ids"] = [x[:MAX_IN - 1] + [eos] for x in sp2.encode(d.src.tolist())]
    if "labels" in d:
        d["labels"] = [objetivo(t) for t in d.tareas]
    d.to_parquet(f"{BASE}/data/processed/s2s_pruned_{split}.parquet", index=False)
fase("s2s re-tokenizado")
json.dump(cob, open(f"{BASE}/results/10_prune_vocab.json", "w"), indent=1)
