"""Live latency experiment (CPU, real cross-encoder).

    python scripts/latency.py configs/latency.yaml [--max-queries 5] [--threads 4] [--repeats 1]

This requires the CE score cache for the configured split/pool (create it with scripts/run.py);
otherwise `prepare` scores the pools first.
"""
import argparse
import logging

import pandas as pd

from pbar.experiment import load_config
from pbar.latency import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--max-queries", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--repeats", type=int)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(a.config)
    if a.max_queries:
        cfg["max_queries"] = a.max_queries
    if a.threads:
        cfg["latency"]["num_threads"] = a.threads
    if a.repeats:
        cfg["latency"]["repeats"] = a.repeats
    summary = run(cfg)
    with pd.option_context("display.width", 220, "display.max_rows", 200, "display.precision", 4):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
