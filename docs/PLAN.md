# Parallel Boundary-Aware Reranking — Plan & Methodology

## 1. Research question

BoundaryMORPH (Caplan et al., arXiv:2609.27213, 2026) allocates a cross-encoder (CE)
budget `B` to decide **top-k membership** among `N` retrieved candidates. It uses a GP
whose prior mean is the dense-retrieval rank, and it spends each CE call on the most
ambiguous incumbent/challenger pair at the top-k boundary. The paper notes that the
method is **strictly sequential** (App. A). It says "the GP posterior is updated after
each CE call" and names batch acquisition ("e.g. auto-kriging") only as future work.

**Question.** At a fixed total CE budget, can we pick `b > 1` documents per round
(fewer sequential rounds, batched CE inference) and so (a) cut wall-clock latency
and (b) keep the selection quality of sequential BoundaryMORPH? When does batching
help, and when does it hurt?

## 2. What we take from the paper (verified against the HTML text)

| Component | Paper specification | Our implementation |
|---|---|---|
| Objective | choose `S ⊆ C`, `|S| = k`, maximising Σ rel | same |
| Kernel | vMF: `K(d,d') = exp(κ(cos(d,d') − 1))` on unit embeddings | same |
| Bandwidth | `κ = log 2 / (1 − median_{i<j} cos)` per query; `κ = 1` if degenerate | same |
| Prior mean | `μ0 = (N−1−pos)/(N−1)`, pos = MIPS position | same (pos 0-indexed, top = 1.0) |
| Query | enters as a zero-cost observation, `y = 1` | same; **μ0(q) is unspecified** (see §6) |
| Posterior | residual GP on `e = y − μ0(A)`, z-standardised over A, then de-standardised | same |
| Noise | `σn² = 1e-4` on the observed block, escalated to 1e-2, then pinv | same |
| Acquisition | `argmin_{j∈S_t, i∉S_t} (μ_j − μ_i)/(σ_j + σ_i)`; query whichever of the pair has the larger σ | same |
| Final set | top-k by μ: scored docs at their CE score, unscored at μ; ties broken by MIPS | same |
| Metrics | nCG@k (human qrels), CE-nCG@k (CE as oracle); nDCG/recall explicitly not used | nCG@k and CE-nCG@k are primary; nDCG@10 is reported separately as *ordering* quality |

Paper setting that we **cannot** match: N = 10,000 from Qwen3-Embedding-4B, the
multilingual-e5-large kernel, the zerank-2 CE, and an A10G GPU. We run on CPU, so we use
smaller models (§4). Absolute numbers are therefore not comparable to the paper's tables.
Only the *relative* comparisons inside this project are meaningful.

## 3. Architecture

```
src/pbar/
  data.py          download + parse BEIR NFCorpus (corpus, queries, qrels per split), checksum
  retrieval.py     dense bi-encoder, corpus embedding cache, MIPS top-N candidate pools
  crossencoder.py  CE model wrapper + CE score cache for whole pools
  oracle.py        budget-enforcing CE oracle. Counts calls and rounds; backed by the cache
                   (simulation) or by the live model (latency runs). One interface serves both.
  metrics.py       nCG@k, CE-nCG@k, nDCG@k, recall@N (unit-tested)
  selectors/       one class per method; all share select(ctx, oracle, k) -> ranking
     baselines.py  MIPS-only, SingleStage (CE on MIPS top-B), Full-CE (upper bound)
     morph.py      sequential BoundaryMORPH            (Milestone 2)
     batch.py      batch extensions                     (Milestone 3)
  gp.py            residual GP posterior (Alg. 1)      (Milestone 2)
  experiment.py    config -> run grid -> per-query JSONL + summary CSV
configs/*.yaml     every experiment is a config file; results record the config hash
scripts/           thin CLIs: build_pools, score_pools, run
tests/             metrics, data integrity, oracle budget accounting, GP correctness
```

**Key design choice: simulation and live execution are separate.** The CE is
deterministic, so we score every (query, candidate) pair once and cache the scores. All
quality experiments then call an oracle that serves cached scores and *enforces the
budget* (a selector cannot see an unpaid score). This makes the large grid (methods ×
B × b × k × queries) cheap and exactly reproducible. Latency is measured in separate runs
with the live CE, where batch size really changes the cost. A test checks that the cached
and live scores agree within padding-induced tolerance.

## 4. Dataset & models

* **BEIR NFCorpus**: 3,633 medical documents, graded qrels (1 = partially, 2 = highly
  relevant). Splits: train / dev / test.
  * **dev** is used for every tuning decision (batch diversity weights, any variant choice).
  * **test** is used only for frozen configurations. Results tables report test.
  * The models are pretrained (e5, an MS MARCO cross-encoder) and are not fine-tuned on
    NFCorpus, so no NFCorpus label reaches any model. BoundaryMORPH itself has no trained
    parameters.
* **Bi-encoder**: `intfloat/e5-base-v2`, used for both the candidate pool and the
  kernel (one embedder, which removes an ambiguity in the paper; §6).
* **Cross-encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2`, sigmoid output in [0,1],
  to match the paper's `f_CE ∈ [0,1]`.
* **Pool size**: N = 1000 by default (configurable). The corpus is only 3.6k documents.

## 5. Experimental methodology

**Grid.** budgets B ∈ {10, 25, 50}; batch sizes b ∈ {1, 2, 4, 8}; context sizes
k ∈ {20, 50, 100}. Equal budget means every method makes exactly B CE calls.
The number of sequential rounds is ⌈B/b⌉.

**Methods.**
* MIPS-only (B = 0), SingleStage (CE on MIPS top-B, one batched round), Full-CE (B = N, upper bound).
* BoundaryMORPH (sequential, b = 1).
* Batch variants (b > 1), each one an ablation of the selection rule:
  1. *Top-b pairs*: the b smallest-gap boundary pairs, no diversity (naive).
  2. *Kriging believer*: greedy within a round; fantasise y = μ(d) for each pick and
     update σ. GP variance does not depend on y, so the σ update is exact. This yields
     diversity through the kernel.
  3. *Believer + explicit diversity*, or a pessimistic/optimistic "kriging liar" fantasy
     (ablation of the fantasy value).
  4. *Random boundary sampling* (control: is the acquisition rule doing anything?).
* Paper ablations reproduced: no MIPS prior (μ0 = const), no boundary acquisition (max-σ).

**Metrics, kept separate.**
* *Selection quality*: nCG@k (human qrels, ideal set within the pool, as in the paper)
  and CE-nCG@k (agreement with the CE oracle; isolates the algorithm from CE domain error).
* *Ranking/ordering quality*: nDCG@10 of the final ordering. This is a secondary metric,
  since BoundaryMORPH chooses membership, not order.
* *Retrieval quality*: recall@N and nCG@k of the pool itself (the ceiling for any reranker).
* *Budget efficiency*: fraction of CE calls spent on docs that end in the final top-k,
  plus duplicate-information rate (mean max kernel similarity within a batch).
* *Cost*: CE calls, sequential rounds, CE wall-clock, GP overhead wall-clock, total
  per-query latency (median and p90), all **measured** on our CPU.

**Statistics.** Paired per-query comparisons: paired bootstrap 95% CIs and the Wilcoxon
signed-rank test against sequential MORPH, with a Holm correction across the grid.
We report the effect size, not just "wins".

**Query subsets.** The paper keeps queries with ≥ 20 relevant documents ("diffuse"
queries). We report all test queries **and** that subset, fixed a priori (not chosen
after seeing results).

## 6. Methodological issues & limitations identified

1. **Regime mismatch.** The paper targets B < k with N ≫ k (diffuse RAG). With k = 20 and
   B = 50, plain SingleStage can already score the whole MIPS top-50, and boundary
   reasoning matters little. We therefore also report k ∈ {50, 100}. Results at
   k = 20, B ≥ k should be read as a different regime, not as a test of the paper.
2. **Paper ambiguities, resolved explicitly as config options:**
   * μ0(q) is undefined. Default: μ0(q) = 1, so the query's residual is 0. Alternative: 0.
   * With one observation (the query alone), std(e) = 0 and the z-score is undefined.
     Default: use scale 1 when |A| < 2 or std < 1e-8. Default ddof = 0.
   * The pool comes from Qwen3-Emb-4B while the kernel uses e5. We use one embedder for both.
   * The final ranking mixes raw CE scores with GP means. This is valid only because μ is
     on the CE scale (the prior is in [0,1] and the residuals are de-standardised). We do it
     as the paper does, and we flag it.
3. **CE quality caps the human-label gains.** An MS MARCO MiniLM CE on medical text can be
   wrong. CE-nCG measures algorithmic fidelity; nCG measures end quality. Full-CE is the
   CE ceiling, and it can sit *below* 100% nCG.
4. **CPU-only latency.** Batching speed-ups on a CPU are smaller than on a GPU, and padding
   to the longest document in a batch can cancel them out. We measure this and do not
   extrapolate to GPUs. Thread count and max sequence length are reported. A GPU run is
   optional future work.
5. **Small corpus and graded but sparse qrels.** Unjudged documents count as non-relevant.
   Queries with no relevant documents in the pool make nCG undefined: we exclude them and
   report how many.
6. **Split hygiene (checked).** No query IDs overlap between train, dev and test. 1 dev/test
   and 4 train/test query texts are identical. No component is trained, so this cannot
   leak labels, and dev-based tuning is affected negligibly.
   Measured split sizes (queries with ≥ 1 positive): train 2590, dev 324, test 323.
   Diffuse (≥ 20 rel): dev 157, test 145. Median relevant per query: dev 18.5, test 16.
7. **Prior/CE scale mismatch (discovered, see FINDINGS F3).** The paper's percentile prior
   breaks with a CE whose scores are skewed toward 0. We add a *calibrated rank prior*
   (extension, `src/pbar/calibration.py`), cross-fitted over query folds on dev and fit on
   dev for test runs. The paper prior is always kept as the faithful reference condition.
8. **Compute logistics.** This machine is a CPU-only laptop. Full-pool CE scores
   (N=1000, 512 tokens) are precomputed once on a free Kaggle T4
   (`scripts/make_kaggle_kernel.py`, `scripts/precompute.py`). Caches are keyed by
   (qid, corpus doc index), so pools rebuilt locally line up exactly. On dev, 3 of 324
   queries had a near-tie at rank 1000 flip between GPU and CPU float arithmetic. The
   doc-keyed cache handles this; a position-keyed one would have silently misaligned scores.
   Latency is still measured locally on CPU.
9. **Single dataset.** Conclusions are about NFCorpus. Generalising would need a second
   BEIR set (e.g. TREC-COVID or SciFact), which is a stretch goal.

## 7. Milestones

| # | Deliverable | Done when |
|---|---|---|
| M0 | data loader, metrics, oracle + tests | `pytest` passes; split statistics printed |
| M1 | dense pools, CE cache, MIPS / SingleStage / Full-CE baselines on dev + test | summary table generated end-to-end from config |
| M2 | sequential BoundaryMORPH + GP tests (posterior vs. a naive numpy reference, interpolation) | beats SingleStage on CE-nCG at B < k on dev (sanity check, not assumed) |
| M3 | batch variants + ablations, tuned on dev only | grid on test with CIs |
| M4 | live latency harness (CE calls, rounds, GP overhead) | latency-vs-quality Pareto plots |
| M5 | analysis write-up: when batching helps/hurts and why | README results section from generated tables only |
