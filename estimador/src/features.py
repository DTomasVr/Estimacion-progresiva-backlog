"""Feature construction shared by the prediction experiments.

Everything that could leak is fitted on the training frame only and then
applied to the others: one-hot vocabularies, scalers, and especially the
project-level effort statistics, which are computed from training rows alone
and fall back to the global training median for projects never seen.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["n_words", "title_words", "desc_len", "has_code", "has_desc",
           "effort_gt_calendar"]


def _numeric_block(d):
    out = pd.DataFrame(index=d.index)
    out["n_words"] = d.text.str.split().str.len().fillna(0)
    out["title_words"] = d.title.fillna("").str.split().str.len().fillna(0)
    out["desc_len"] = d.description.fillna("").str.len()
    out["has_code"] = d.has_code.fillna(0)
    out["has_desc"] = d.has_desc.fillna(0)
    out["effort_gt_calendar"] = d.effort_gt_calendar.fillna(0)
    out["log_words"] = np.log1p(out.n_words)
    out["log_desc"] = np.log1p(out.desc_len)
    return out.to_numpy(float)


class ProjectStats:
    """Median and count of log-effort per project, from training rows only."""

    def fit(self, tr):
        g = tr.groupby("project").effort_hours
        self.median_ = np.log(g.median())
        self.count_ = np.log1p(g.size())
        self.global_median_ = float(np.log(tr.effort_hours.median()))
        self.global_count_ = 0.0
        return self

    def transform(self, d):
        m = d.project.map(self.median_).fillna(self.global_median_)
        c = d.project.map(self.count_).fillna(self.global_count_)
        seen = d.project.isin(self.median_.index).astype(float)
        return np.column_stack([m, c, seen])


class FeatureBuilder:
    """Assemble embeddings + categorical + numeric + project blocks."""

    def __init__(self, use_kind=True, use_priority=True, use_source=True,
                 use_numeric=True, use_project=True, use_emb=True):
        self.opts = dict(kind=use_kind, priority=use_priority,
                         source=use_source, numeric=use_numeric,
                         project=use_project, emb=use_emb)

    def _cat_cols(self):
        cols = []
        if self.opts["kind"]:
            cols.append("task_kind")
        if self.opts["priority"]:
            cols.append("priority")
        if self.opts["source"]:
            cols.append("source")
        return cols

    def fit_transform(self, tr, others, emb=None, pos=None, kind_override=None):
        """kind_override: {frame_index: array of categories} to substitute the
        true task_kind with predicted ones (frame 0 is `tr`)."""
        frames = [tr] + list(others)
        if kind_override is not None:
            frames = [f.copy() for f in frames]
            for i, k in kind_override.items():
                frames[i]["task_kind"] = k

        blocks = [[] for _ in frames]
        cat_cols = self._cat_cols()
        if cat_cols:
            oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            blocks[0].append(oh.fit_transform(frames[0][cat_cols]))
            for i in range(1, len(frames)):
                blocks[i].append(oh.transform(frames[i][cat_cols]))

        if self.opts["numeric"]:
            sc = StandardScaler()
            blocks[0].append(sc.fit_transform(_numeric_block(frames[0])))
            for i in range(1, len(frames)):
                blocks[i].append(sc.transform(_numeric_block(frames[i])))

        if self.opts["project"]:
            ps = ProjectStats().fit(frames[0])
            sc2 = StandardScaler()
            blocks[0].append(sc2.fit_transform(ps.transform(frames[0])))
            for i in range(1, len(frames)):
                blocks[i].append(sc2.transform(ps.transform(frames[i])))

        if self.opts["emb"] and emb is not None:
            for i, f in enumerate(frames):
                blocks[i].insert(0, emb[[pos[t] for t in f.task_id]])

        return [np.hstack(b) for b in blocks]


def load_embeddings_and_pos(model_name=None):
    import sys
    sys.path.insert(0, "src")
    enc = __import__("15_encode_sbert")
    name = model_name or enc.DEFAULT
    emb = enc.load_embeddings(name)
    ids = pd.read_parquet("data/processed/agix_effort_clean.parquet",
                          columns=["task_id"]).task_id.to_numpy()
    return emb, {t: i for i, t in enumerate(ids)}
