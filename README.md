# Parallel Boundary-Aware Reranking

Can the boundary-based CE budget allocation of **BoundaryMORPH** (Caplan et al., 2026,
arXiv:2609.27213) pick *several* documents per round without losing selection quality?
And does that actually reduce wall-clock latency?

Methodology, paper-faithfulness notes, limitations and milestones: [docs/PLAN.md](docs/PLAN.md).

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate        # Windows; use bin/activate on Linux/macOS
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .[dev]
pytest
```

## Run

```bash
# smoke test (20 dev queries)
python scripts/run.py configs/baselines.yaml --max-queries 20
# full dev split; test is run only for frozen configurations
python scripts/run.py configs/baselines.yaml
```

The first run downloads BEIR NFCorpus to `data/raw/`, embeds the corpus, and CE-scores
every (query, pool doc) pair once into `cache/`. All quality experiments then replay
these cached scores through a budget-enforcing oracle. Outputs go to
`results/<name>_<split>_<config-hash>/`: `per_query.csv`, `summary.csv` (means with
bootstrap 95% CIs, all queries and the ≥20-relevant "diffuse" subset), and `meta.json`.

## Status

- [x] M0: data, metrics, oracle, tests
- [x] M1: MIPS / SingleStage / Full-CE baselines (dev)
- [x] M2: sequential BoundaryMORPH (paper Alg. 1/2), plus the paper's ablations
- [x] M3: batch variants (top-b, kriging believer/liar, random-boundary control) and a calibrated-prior extension (dev)
- [ ] M4: live CPU latency (running)
- [ ] M5: test-split confirmation and analysis

## Results

Dev-split findings so far are in [docs/FINDINGS.md](docs/FINDINGS.md). They are copied from
generated result files. Test-split results will appear here only after configurations are frozen.

## Heavy precompute on a free GPU (optional)

```bash
python scripts/make_kaggle_kernel.py OUT --kernel-id <user>/<slug> --splits dev test \
    --gpu0 cross-encoder/ms-marco-MiniLM-L-6-v2 --pool-size 1010
kaggle kernels push -p OUT && kaggle kernels output <user>/<slug> -p dl/   # unzip dl/cache.zip here
```
