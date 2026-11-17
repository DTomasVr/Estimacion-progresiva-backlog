# Task & NFR Generation Module — Project Spec

> Handoff document for a Claude Code session. This module replaces the LLM-based
> task/acceptance-criteria/NFR generation component of the thesis system (Agix) with a
> **trained, dataset-grounded model** reusing an existing multilingual SBERT backbone.
> Goal: turn an opaque external-LLM API call into a model we design, train, and evaluate —
> the "bigger CS" contribution the advisor asked for.

---

## 1. Problem definition

**Input:** a project name / description (short natural-language text).

**Outputs (three coupled predictions):**
1. **Task set** — the set of task descriptions the project would plausibly need. *One input → many text outputs* (variable-size set).
2. **Acceptance criteria** — generated *per task* (Given/When/Then style), conditioned on the individual task, not the project.
3. **Non-functional requirements** — multi-label classification over the **ISO 25010** category set (performance efficiency, compatibility, usability, reliability, security, maintainability, portability, functional suitability), optionally with generated NFR statement text per fired category.

**Explicitly dropped:** Gantt / critical-path output. Rationale: it assumes fixed sequential dependencies and known durations, which cuts against agile's iterative premise. Kanban / burndown are the agile-native alternatives. (Flag to advisor + co-author before deep implementation, since it changes the documented C1–C7 architecture.)

**Not the goal:** story-point / effort regression. That lives in a separate module.

---

## 2. Core technical decision (read this first)

The pooled SBERT vector is optimized for **similarity, not reconstruction**. Mean-pooled, cosine-tuned embeddings capture gist ("payments project needing auth tasks") but throw away the token-level detail a decoder needs to produce precise task wording.

**Consequence — do NOT bottleneck generation through the pooled vector.**
- Vector → does **retrieval** and **gist-level conditioning**.
- Generator → reads the **retrieved text**, not just the vector.

This single decision drives the whole architecture below.

---

## 3. Architecture / pipeline

### Stage 0 — Corpus + index (offline)
- Build training rows: `(project_description → [tasks] → [acceptance_criteria per task] → [NFR labels/statements])`.
- Embed every project description with the existing multilingual SBERT; store vectors + payloads.
- **Use `pgvector` in the existing PostgreSQL instance** (from the Miraflores stack) — no new infra vs. standing up FAISS separately; ANN search lives in the DB we already run.
- Also embed each **task** individually (needed for candidate dedup + the AC stage).

### Stage 1 — Retrieve
- New project description → SBERT embed → top-k neighbour projects via pgvector.
- Pull their task sets into a candidate pool; **deduplicate** by cosine similarity between task embeddings (collapse 40 variants of "set up CI" into one).
- This pool is BOTH the candidate set AND the generator's grounding context.

### Stage 2 — Task-set model (the "new model" — three paths, increasing CS depth)

- **Path A — learned ranker** *(build this first)*
  - Train `f(project_embedding, candidate_task_embedding) → relevance ∈ [0,1]`.
  - Output tasks above a threshold as the predicted set.
  - Pure "train on vectorized text," reuses SBERT directly, trivially evaluable (P/R/F1 vs. ground-truth task sets).
  - Limitation: can only select from retrieved tasks, can't invent novel ones. That's fine for a first result and becomes the **baseline** for Path B.
  - Realistic: full working pipeline + numbers in ~1 week.

- **Path B — conditioned set decoder** *(headline contribution)*
  - Decoder takes retrieved candidate tasks as encoder input (concatenated text) **+** project embedding as a conditioning prefix/soft-prompt → emits the task set.
  - Multilingual SBERT ⇒ pair with **mT5 or ByT5** decoder to keep the pipeline language-agnostic.
  - Train **One2Seq** first (whole set as one delimiter-separated target); upgrade to **One2Set** (control codes + bipartite matching) for order-invariance and the parallel-set-generation novelty angle.
  - Handles *absent* tasks (not in any neighbour) that Path A structurally cannot.

- **Path C — hybrid retrieve-then-edit** *(likely "future work")*
  - Path B decoder framed as adapting/merging retrieved tasks rather than free generation. Best quality, most engineering.

### Stage 3 — NFR head (parallel, off the same embedding)
- MLP classifier head on project embedding → ISO 25010 categories.
- Multi-label: sigmoid outputs + BCE loss, per-category threshold.
- Cleanly evaluable (per-category F1). **Lead the results chapter with this** — it's the most tractable, most defensible sub-component.
- Optional: generate NFR statement text per fired category via template/retrieval.

### Stage 4 — Acceptance criteria (per generated task)
- For each Stage 2 task: SBERT-embed → retrieve tasks-with-AC from corpus → generate/adapt Given/When/Then conditioned on *that task*.
- Same decoder family as Stage 2, or a lighter retrieval-and-fill step.

---

## 4. Datasets

### A. Project → task/issue (Stage 0 core)
| Dataset | Scale | What it gives | Access |
|---|---|---|---|
| **TAWOS** (MSR 2022) | 500k+ issues, 44 projects, 13 Jira repos | Title, Description, Type per issue | github.com/SOLAR-group/TAWOS (SQL/Mongo dump) |
| **Public Jira Dataset** (Montgomery, MSR 2022) | 1,822 projects, 2.7M issues, 16 Jiras | Requirements/dev/maintenance issue types | zenodo.org/records/15719919 |
| **JOSSE** (Glasgow 2022) | Apache/JBoss/Spring | Task description + actual effort + expert estimates | zenodo.org/records/7022735 |
| **Choetkiertikul SP dataset** (IEEE TSE 2018) | 16 projects, ~23k issues | Title + description; canonical benchmark | github.com/SEAnalytics/datasets |
| **Semantically-enriched Jira** (Diamantopoulos, MSR 2023) | — | Semantic annotations on Jira issues | MSR 2023 artifact |
| **JIRA Repository Dataset** (Ortu, PROMISE 2015) | — | Early clean source (social focus) | PROMISE 2015 |

**Primary base: TAWOS** (scale + Title/Description/Type is trainable for Path B).

### B. Requirements / NFR (Stage 3)
| Dataset | Scale | Notes |
|---|---|---|
| **PROMISE NFR** | 625 reqs (255 FR, 370 NFR) | Decade-long field benchmark |
| **PROMISE_exp** | 969 (444 FR, 525 NFR) | Extended PROMISE |
| **PURE** | 79 docs, 34,268 sentences | Raw corpus, unlabeled |
| **FR_NFR** | 6,118 (3,964 FR, 2,154 NFR) | Binary FR/NFR |
| **GitReq** | — | Newer gold-standard for quality requirements; arxiv.org/abs/2606.21810 |

### C. AI-era / newer task framing (supplementary)
- **SWE-bench family** — real GitHub issues. `SWE-bench-train` = 19k instances / 37 repos; `SWE-bench Verified` = 500 human-validated; **`SWE-bench-Live`** = 1,319 instances created Jan 2024–Apr 2025, 93 repos, updated monthly. github.com/SWE-bench/SWE-bench
- ⚠️ Task shape is *issue → patch*, NOT *project → task set*. Use as **analysis/vocabulary reference**, not primary training data.

### D. Bulk GitHub sources (only if scraping)
- **GH Archive** (gharchive.org) — hourly public-event stream, queryable via BigQuery. Viable bulk-discovery route.
- **GHTorrent** — largely inactive/frozen; archival only.

---

## 5. Known data gaps (budget preprocessing for these)

1. **Missing project-level description.** Jira project entries are often just a name + key, not a paragraph. Stage 0 must **source/synthesize** the project description separately — pull the linked repo README / homepage blurb per project. This is itself a documentable methods contribution ("we augmented TAWOS with project-level descriptions derived from READMEs").
2. **Issues ≠ planned tasks.** Backlog items are reactive (bugs/requests), not expert-authored canonical task lists. Frame ground truth honestly: the model learns "what tasks co-occur in projects like this," which *is* the goal — just not the same as a canonical decomposition.

---

## 6. GitHub scraping decision

**Do NOT scrape from scratch. Supplement instead.**
- Train on validated Jira datasets (A) + requirements sets (B).
- Add a **small, heavily filtered** scrape of recent (2024–2025) AI-era repos (LLM apps, agent frameworks, RAG, MLOps) ONLY to capture the post-LLM task-vocabulary shift ("add prompt template," "tune retrieval chunk size," "add eval harness," "wire up vector store") that pre-2022 corpora can't teach.

**If scraping, make it not-fussy:**
- Discover via GH Archive/BigQuery → pull detail via GitHub **GraphQL API** (REST is capped at 5,000 req/hr authenticated).
- Filter *before* storing: min stars, substantive README present, issue templates used, English-only (language detection), created in target window, drop bot/duplicate/question issues.
- Respect repo licenses for anything redistributed. Keep it to a few hundred curated projects, not a noisy mass dump.

---

## 7. Tech stack (already in place)

- ✅ **Multilingual SBERT**, fine-tuned, running locally → retrieval backbone done.
- ✅ **PostgreSQL** (Miraflores stack) → add `pgvector` extension for the index.
- ✅ **FastAPI** → serving layer.
- To add: **mT5 / ByT5** decoder (Path B), PyTorch training loop, evaluation harness.

---

## 8. Evaluation plan

- **NFR (Stage 3):** per-category precision / recall / F1 (multi-label). Report first.
- **Task set (Stage 2):** P/R/F1 vs. held-out ground-truth task sets; for generative output, ROUGE / BERTScore between predicted and actual task sets.
- **Acceptance criteria (Stage 4):** small human-rated relevance study (5–10 projects rated by advisor/peers) alongside automatic metrics.
- **Baselines:** run GPT-4 / Llama **zero-shot** on the same held-out projects for Stages 2 & 4. This comparison *is* the "we replaced the LLM with a trained, grounded, evaluable model" argument. Keep the LLM-prompting papers as the state-of-the-art reference set.

---

## 9. Suggested first tasks for the Claude Code session

1. **Scaffold** the repo: data ingestion, embedding store (pgvector), training, eval as separate modules.
2. **Stage 0 loader for TAWOS**: parse the dump, extract `(project, issue_title, issue_description, issue_type)`, write the README-augmentation step for project descriptions.
3. **pgvector index**: schema for project + task embeddings; ANN query for top-k retrieval.
4. **Stage 1 retrieval + dedup**: candidate-pool builder with cosine-similarity task deduplication.
5. **Path A ranker**: train `f(project_emb, task_emb) → relevance`; full end-to-end pipeline + first P/R/F1 numbers.
6. **Stage 3 NFR head**: multi-label MLP on PROMISE/PROMISE_exp mapped to ISO 25010; per-category F1.
7. *(then)* **Path B decoder**: mT5/ByT5, One2Seq → One2Set.

---

## 10. Open questions to resolve early

- How many complete `project → task-set` pairs does the self-built dataset currently hold? (Gates Path A vs. Path B — a conditioned decoder wants low-thousands of paired projects; a ranker needs far fewer.)
- Should AC training reuse the task corpus or need a separate `task → AC` labeled set?
- Confirm the Gantt-drop and the C1–C7 architecture change with advisor + co-author before implementation.

---

## 11. Reference paper sets (for the thesis write-up)

- **Set / keyphrase generation (build-your-own-model methods):** One2Set (ACL 2021); *A Branching Decoder for Set Generation* (ICLR 2024); *Rethinking Model Selection and Decoding for Keyphrase Generation* (EMNLP 2023, arXiv 2310.06374); *MetaKP: On-Demand Keyphrase Generation* (arXiv 2407.00191).
- **Multi-label over embeddings:** QUEST (EMNLP 2024 Findings); ProtoSiTex (arXiv 2510.12534); LemWec (ESWA 2025).
- **NFR classification:** DReqANN (Scientific Reports 2024, open, s41598-024-52802-0); CEUR-WS Vol-4141 paper 7.
- **RAG in RE (retrieval backbone context):** Arora et al. RE'24 (test scenarios from NL requirements); *From Domain Documents to Requirements* (arXiv 2507.07689); *Generative AI for Requirements Engineering: An SLR* (Software: Practice & Experience 2026).
- **Baselines (LLM prompting — comparison targets):** GeneUS (arXiv 2404.01558); *Can LLMs Generate User Stories and Assess Their Quality?* (arXiv 2507.15157); RAGcceptance_M2RE multimodal AC (arXiv 2508.06888); SimAC (Automated Software Engineering 2024).
