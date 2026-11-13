# Findings log (dev split only, so far)

All numbers come from result directories under `results/`. Nothing here was computed by hand.
Every result so far is **dev**. Test has not been evaluated.

Setup: BEIR NFCorpus dev (324 queries with ≥1 positive). Pool: top-1000 by
`intfloat/e5-base-v2` MIPS. CE: `cross-encoder/ms-marco-MiniLM-L-6-v2` (sigmoid,
max_length 512), scored once on a Kaggle T4 (fp32) and replayed through a budget-enforcing oracle.

## F1. This CE is weaker than the retriever on NFCorpus (`results/baselines_dev_33f4df77b5`)

| k | MIPS nCG | Full-CE nCG (rank whole pool by CE) |
|---|---|---|
| 20 | 34.3 | 33.7 |
| 50 | 39.0 | 36.6 |
| 100 | 46.2 | 43.0 |

Any method that approximates the CE's top-k is therefore capped on human labels. This is
why CE-nCG (selection fidelity w.r.t. the CE) and nCG (end quality) are reported separately.
A pilot with stronger CEs is running (model selection on dev).

## F2. SingleStage does nothing once B ≤ k

It reranks the MIPS top-B, which already lie inside the top-k, so the selected *set* is
identical to MIPS (k=50 and k=100, all B: same CE-nCG and nCG as MIPS). This reproduces
the motivation of BoundaryMORPH.

## F3. The paper's percentile prior is not invariant to CE score scale (`results/batch_dev_32f2998ec0`)

This CE's sigmoid scores are extremely skewed (40 dev queries: median 0.0000, p99 0.026,
0.4% > 0.5). The paper's prior μ0 = (N−1−pos)/(N−1) lies far above these values.
* Seq. MORPH at k=20, B=10 has CE-nCG **73.4 < 74.6 for MIPS with zero CE calls**.
* The max-σ ablation **degrades as budget grows** (k=20: 61.8 → 47.8 → 35.9 for B = 10/25/50).
* Diagnosis (40 queries): in the max-σ run, unscored docs in the final set had posterior
  mean − true CE = **+0.66** on average. Scored docs (shown at their true, low CE)
  lose to unscored docs with inflated μ, so even genuinely relevant scored docs get ejected.

## F4. Calibrated rank prior (our extension) fixes F3 (`results/batch_calib_dev_04a2d7e8d4`)

μ0(pos) = E[CE | MIPS rank = pos], isotonic (PAV), **cross-fitted over 2 query folds**
(each query's prior is fit on the other fold, which a unit test checks). It uses only CE
scores and ranks, never qrels.

| CE-nCG % | MIPS | SingleStage | MORPH, paper prior | MORPH, calibrated |
|---|---|---|---|---|
| k=20, B=10 | 74.6 | 74.6 | 73.4 | 76.0 |
| k=20, B=25 | 74.6 | 77.2 | 78.6 | 79.9 |
| k=50, B=25 | 80.0 | 80.0 | 82.0 | 81.5 |
| k=100, B=50 | 84.1 | 84.1 | 86.1 | 86.2 |

* Max-σ with the calibrated prior is now flat (74.7 / 74.9 / 75.2 at k=20). It no longer collapses.
* Human nCG at k=20: calibrated MORPH 34.9 / 35.8 / 36.1 (B = 10/25/50) vs MIPS 34.3 and
  Full-CE 33.7 on dev. **This did not replicate on test (see F8): treat it as dev noise.**
  At k=100, all CE-based methods stay ≤ MIPS (F1).

## F5. Batch selection costs no measurable quality (same run)

Paired against calibrated sequential MORPH, over all 63 (variant ∈ {top-b, KB-mean}, b ∈ {2,4,8}, B, k) cells:
* CE-nCG: |Δ| ≤ **0.87 pp** (widest bootstrap 95% CI [0.08, 1.83]).
* Human nCG: |Δ| ≤ **0.80 pp**, **0 of 63** significant after Holm.
* Control, random sampling in the boundary band (b=4): **−3 to −9 pp**, significant in every
  cell. The boundary-uncertainty rule, not "spending calls near the cut", produces the gain.
* Fantasy ablation: the pessimistic kriging liar is slightly worse than the mean fantasy (−0.1 to −0.6 pp).
* Statistics note: Wilcoxon flags some 0.03–0.4 pp CE-nCG deltas because of many tied
  per-query differences. We treat bootstrap CIs and effect sizes as the primary evidence.

## F6. On this CPU, batching makes CE inference slower per document (`results/latency_dev_663e55587d/microbench.csv`)

Setup: MiniLM-L-6 CE, max_length 512, 4 torch threads, Ryzen 7 5825U laptop. The same 32
random pool docs × 3 queries are split into calls of size bs; medians over 5 repeats.

| batch size | s / call | s / doc | per-doc speed vs bs=1 |
|---|---|---|---|
| 1 | 0.057 | 0.057 | 1.00× |
| 2 | 0.134 | 0.067 | 0.86× |
| 4 | 0.305 | 0.076 | 0.74× |
| 8 | 0.708 | 0.089 | 0.68× |
| 16 | 1.418 | 0.089 | 0.64× |
| 32 | 2.809 | 0.088 | 0.65× |

A single sequence already saturates the threads, and batched sequences are padded to the
longest doc in the call. **Fewer rounds therefore do not translate into lower latency on
this hardware.** The latency benefit of batch selection depends on an accelerator (or a
remote CE service with per-request overhead), where batch throughput scales. This has to
be measured there, not assumed. The end-to-end per-query timings are in the same run directory.

## F7. End-to-end CPU latency: sequential is fastest here (`results/latency_dev_663e55587d/summary.csv`)

15 dev queries, k=50, live CE, 4 threads, 2 warm-up queries discarded, randomized job order,
median over 2 repeats. Per-query median wall-clock (s):

| method | B=25 | B=50 | rounds (B=50) | GP overhead (B=50) |
|---|---|---|---|---|
| SingleStage, b=1 | 1.36 | 2.91 | 50 | ~0 |
| MORPH seq. (b=1) | 1.55 | 2.93 | 50 | 0.16 |
| MORPH-KB b=2 | 1.83 | 3.37 | 25 | 0.14 |
| MORPH-KB b=4 | 2.02 | 3.85 | 13 | 0.14 |
| MORPH-KB b=8 | 2.13 | 4.10 | 7 | 0.11 |
| SingleStage, one batch | 2.22 | 4.70 | 1 | 0 |

* On CPU, wall-clock is dominated by CE compute, which batching *increases* (F6). Cutting
  rounds 7× (b=8) makes queries ~40% *slower*.
* BoundaryMORPH's GP overhead is small: ~0.1–0.16 s/query, ≈5% of total at N=1000, B=50.
* Live CPU scores match the GPU-computed cache (max abs diff rounds to 0.000).
* Conclusion so far: batch selection keeps quality (F5) but pays off in latency only where
  batched CE throughput scales (GPU or a remote CE service). Next step: rerun this harness
  on a Kaggle T4.

## F8. TEST split, frozen config, run once (`results/final_test_8b63c00e50`)

Config `configs/final_test.yaml`, chosen on dev. 323 test queries. Calibrated prior fit on
**dev only** (`calibration: {mode: split, fit_split: dev}`). Reference: calibrated sequential MORPH.

**Replicated from dev:**
* **Batching is quality-neutral.** Top-b and KB at b ∈ {2,4,8} vs sequential: CE-nCG |Δ| ≤ 0.89 pp
  (widest CI [−1.59, 1.25]). Human nCG |Δ| ≤ 0.87 pp, **0 of 54 significant** (Holm).
* **The acquisition rule matters.** Random-boundary control: CE-nCG −4.0 to −9.2 pp (9/9 significant).
* **Boundary selection improves CE-set fidelity over MIPS** (calibrated MORPH − MIPS, CE-nCG):
  k=20: +3.4 / +7.6 / +10.0 pp (B = 10/25/50); k=50: +0.4 / +2.5 / +4.7; k=100: −0.2 / +0.2 / +1.5.
  The CIs exclude 0 except at k=50, B=10 and at k=100, B ≤ 25.
* **Calibrated vs paper prior:** calibrated is better at k=20 with small B (paper prior −2.5 pp
  at B=10, −1.1 pp at B=25, both significant). Elsewhere the two are within ~±1.8 pp; the paper
  prior is nominally ahead at k=20, B=50 (+1.8, not significant).

**Not replicated (corrects F4):** on human labels, calibrated MORPH vs MIPS is
**−0.7 to +0.7 pp, with every 95% CI including 0** (k ∈ {20,50,100}, B ∈ {10,25,50}). The dev
gain at k=20 (+1.8 pp) did not hold up on test. With this CE (F1), better CE-set fidelity does
not turn into better human-judged relevance on NFCorpus.

Statistics note (again): Wilcoxon gives p < 1e-10 for some cells whose bootstrap CI of the mean
includes 0 (e.g. CE-nCG k=50, B=10). The per-query differences are heavily tied and skewed,
so we report the bootstrap CI of the mean as the primary evidence.

## F9. CE pilot: no available CE beats the dense retriever on a deep NFCorpus pool (dev)

Selection on DEV only (`scripts/compare_ce.py`). CEs were scored on a Kaggle T4 (MiniLM-L-6 in fp32,
the others in fp16, max_length 512, the same e5 pools).

Full-CE (rank the entire 1000-doc pool by CE) minus MIPS, human nCG, paired 95% CI:

| CE | k=20 | k=50 | k=100 | CE score median / share > 0.5 |
|---|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 | −0.6 [−2.5, +1.4] | −2.4 [−4.3, −0.5] | −3.3 [−5.5, −1.0] | 3e-5 / 0.3% |
| ms-marco-MiniLM-L-12-v2 | −6.2 [−8.6, −3.9] | −6.8 [−9.1, −4.5] | −6.7 [−9.1, −4.3] | 4e-5 / 1.8% |
| bge-reranker-base | −6.1 [−8.4, −3.8] | −6.0 [−8.4, −3.4] | −5.5 [−8.4, −2.6] | 4e-4 / 0.5% |
| bge-reranker-v2-m3 | −4.6 [−6.7, −2.5] | −4.6 [−6.7, −2.6] | −4.7 [−7.1, −2.5] | 7e-4 / 0.9% |

Sanity check that the scores are valid: rerank only the MIPS top-`depth`, nDCG@10 vs MIPS (32.8):

| CE | depth 20 | depth 100 | depth 1000 |
|---|---|---|---|
| MiniLM-L-6 | +0.8 [−0.4, +2.0] | +0.4 [−1.1, +1.8] | −0.1 [−1.6, +1.5] |
| MiniLM-L-12 | +0.3 [−0.9, +1.6] | −1.4 [−3.1, +0.1] | −6.0 [−8.1, −4.1] |
| bge-reranker-base | −0.2 [−1.5, +1.1] | −3.0 [−4.7, −1.3] | −6.2 [−8.4, −4.1] |
| bge-reranker-v2-m3 | **+1.4 [+0.1, +2.8]** | −0.3 [−1.9, +1.3] | −3.0 [−4.8, −1.2] |

Interpretation: shallow reranking behaves as expected (small or no gains, as is typical on
NFCorpus with a strong dense retriever). Deeper pools let CE false positives in. On a
1000-doc pool every CE is a *worse* relevance signal than e5. Every CE's scores are also
heavily skewed (median ≪ 0.01), so the calibration issue (F3) is general, not specific to MiniLM.
**Decision (made on dev): keep MiniLM-L-6.** It is the least harmful on deep pools and the cheapest,
so the test results (F8) stand as the main condition. The open question "do CE-fidelity gains
become human-label gains when the CE is better than the retriever?" cannot be answered on
NFCorpus. It needs a dataset where reranking clearly helps (e.g. SciFact or TREC-COVID).

## Resume here (next session)

```bash
# 1. CE pilot (private Kaggle kernel; dev only, fp16). Fetch and unzip into the repo root:
.venv-pbar/Scripts/kaggle kernels status bojanabiga/pbar-ce-pilot
.venv-pbar/Scripts/kaggle kernels output bojanabiga/pbar-ce-pilot -p dl/ && python -I -c "import zipfile; zipfile.ZipFile('dl/cache.zip').extractall('.')"
#    Compare Full-CE vs MIPS nCG on dev per CE (configs/baselines.yaml with cross_encoder.model
#    and fp16: true). Pick a CE on DEV only. If one is chosen, its test scores need another Kaggle run.
# 2. Frozen test run: DONE 2026-11-12 (F8). Do not re-run with changed settings.
.venv-pbar/Scripts/python.exe scripts/run.py configs/final_test.yaml --workers 4
.venv-pbar/Scripts/python.exe scripts/analyze.py results/final_test_<hash> --metric ce_ncg --ref 'morph|{"batch_size": 1, "prior": "calibrated"}'
```

## Open
* Latency: does cutting rounds by b× actually reduce wall-clock time on CPU? (M4, running)
* Does F5 hold with a stronger, better-calibrated CE? (pilot running on Kaggle)
* Test-split confirmation with frozen configs (calibration fit on dev).
