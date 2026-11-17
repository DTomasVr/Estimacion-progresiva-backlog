"""Contrastive fine-tune of paraphrase-multilingual-mpnet-base-v2 for
retrieval ("what planned work goes under this parent").

Loss: CachedMultipleNegativesSymmetricRankingLoss (GradCache) -- in-batch
negatives in both directions (parent->child and child->parent) with VRAM set
by mini_batch_size, not by batch size, so the batch can be large (more
negatives) without spilling past ~6.5 GB into WDDM shared memory.
LR scaled with sqrt(batch/64) from the 2e-5 baseline.

Selection on the VAL projects (IR, en and es queries). The test split is only
touched by 07_eval_sbert.py.

usage: python src/06_finetune_sbert.py [--mini]
"""
import json
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.losses import CachedMultipleNegativesSymmetricRankingLoss
from sentence_transformers.training_args import BatchSamplers
from transformers import TrainerCallback

BASE = r"F:\Software Task generation Model"
NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MINI = "--mini" in sys.argv
BATCH = 256
MINI_BATCH = 32
LR = 2e-5 * math.sqrt(BATCH / 64)
EPOCHS = 1
MAXLEN = 128
SEED = 20260922
OUT = f"{BASE}/models/sbert_pbe_v1" + ("_mini" if MINI else "")


def ir_evaluator(df, lang, nombre, max_corpus=None):
    """queries = unique parents, corpus = unique children, relevant = its children."""
    col = "anchor_es" if lang == "es" else "anchor_en"
    df = df[df[col].notna()]
    corpus_txt = df.positive.drop_duplicates()
    if max_corpus:
        corpus_txt = corpus_txt.sample(min(max_corpus, len(corpus_txt)), random_state=SEED)
        df = df[df.positive.isin(set(corpus_txt))]
    cid = {t: f"c{i}" for i, t in enumerate(corpus_txt)}
    qs = df.groupby("parent_uid").agg(q=(col, "first"), pos=("positive", list))
    queries = {f"q{i}": q for i, q in enumerate(qs.q)}
    rel = {f"q{i}": {cid[p] for p in pos} for i, pos in enumerate(qs.pos)}
    corpus = {v: k for k, v in cid.items()}
    return InformationRetrievalEvaluator(queries, corpus, rel, name=nombre,
                                         mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 10],
                                         precision_recall_at_k=[10, 50], map_at_k=[100],
                                         batch_size=128, show_progress_bar=False)


class Reloj(TrainerCallback):
    """Wall-clock, ETA and VRAM peak every `cada` steps."""

    def __init__(self, cada):
        self.cada, self.t0 = cada, None

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args, state, control, **kw):
        s = state.global_step
        if s % self.cada == 0 or s == state.max_steps:
            dt = time.time() - self.t0
            eta = (state.max_steps - s) * dt / s
            loss = next((h["loss"] for h in reversed(state.log_history) if "loss" in h), float("nan"))
            print(f"  paso {s:>5}/{state.max_steps}  {dt/s:5.2f} s/paso  "
                  f"ETA {eta/60:5.1f} min  loss {loss:.4f}  "
                  f"VRAM pico {torch.cuda.max_memory_allocated()/2**30:.2f} GB", flush=True)


def main():
    t0 = time.time()
    P = pd.read_parquet(f"{BASE}/data/processed/sbert_pairs.parquet")
    tr = P[P.split == "train"].sample(frac=1, random_state=SEED)
    va = P[P.split == "val"]
    anchor = np.where(tr.usa_es, tr.anchor_es, tr.anchor_en)
    train = Dataset.from_dict({"anchor": list(anchor), "positive": tr.positive.tolist()})
    print(f"train {len(train):,} pares ({tr.usa_es.mean():.0%} ancla es)  val {len(va):,}")

    model = SentenceTransformer(NAME, device="cuda")
    model.max_seq_length = MAXLEN
    loss = CachedMultipleNegativesSymmetricRankingLoss(model, mini_batch_size=MINI_BATCH)
    mc = 3000 if MINI else None
    ev = SequentialEvaluator([ir_evaluator(va, "en", "val_en", mc), ir_evaluator(va, "es", "val_es", mc)],
                             main_score_function=lambda s: float(np.mean(s)))

    base = ev(model)
    base_s = {k: round(v, 4) for k, v in base.items() if "ndcg@10" in k or "mrr@10" in k or "recall@10" in k}
    print(f"[{time.time()-t0:5.0f}s] congelado en val: {base_s}", flush=True)

    pasos = len(train) // BATCH
    eval_cada = 30 if MINI else max(50, pasos // 8)
    args = SentenceTransformerTrainingArguments(
        output_dir=OUT + "_ckpt", num_train_epochs=EPOCHS,
        max_steps=60 if MINI else -1,
        per_device_train_batch_size=BATCH, learning_rate=LR, warmup_steps=0.1,
        lr_scheduler_type="linear", weight_decay=0.01, bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps", eval_steps=eval_cada, save_strategy="steps", save_steps=eval_cada,
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_sequential_score", greater_is_better=True,
        logging_steps=10, dataloader_num_workers=0, report_to="none", seed=SEED,
        disable_tqdm=True)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train, loss=loss,
                                         evaluator=ev, callbacks=[Reloj(10 if MINI else 25)])
    trainer.train()

    fin = ev(model)
    fin_s = {k: round(v, 4) for k, v in fin.items() if "ndcg@10" in k or "mrr@10" in k or "recall@10" in k}
    print(f"[{time.time()-t0:5.0f}s] ajustado (mejor ckpt) en val: {fin_s}", flush=True)
    model.save(OUT)
    json.dump(dict(batch=BATCH, mini_batch=MINI_BATCH, lr=LR, epochs=EPOCHS, maxlen=MAXLEN,
                   n_train=len(train), congelado_val=base_s, ajustado_val=fin_s,
                   log=trainer.state.log_history, segundos=round(time.time() - t0)),
              open(f"{BASE}/results/06_sbert{'_mini' if MINI else ''}.json", "w"), indent=1)
    print(f"guardado {OUT}  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
