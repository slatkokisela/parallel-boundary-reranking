"""Run a quality experiment from a config.

    python scripts/run.py configs/baselines.yaml [--split test] [--max-queries 20]
"""
import argparse
import logging

import pandas as pd

from pbar.experiment import load_config, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split")
    ap.add_argument("--max-queries", type=int)
    ap.add_argument("--pool-size", type=int)
    ap.add_argument("--workers", type=int, default=1, help="parallel processes over (method,B,k) jobs")
    ap.add_argument("--ce-model", help="override cross_encoder.model")
    ap.add_argument("--ce-fp16", action="store_true", help="use the fp16-scored CE cache")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(a.config)
    if a.split:
        cfg["split"] = a.split
    if a.max_queries:
        cfg["max_queries"] = a.max_queries
    if a.pool_size:
        cfg["pool_size"] = a.pool_size
    if a.ce_model:
        cfg["cross_encoder"]["model"] = a.ce_model
        cfg["name"] += "_" + a.ce_model.split("/")[-1]
    if a.ce_fp16:
        cfg["cross_encoder"]["fp16"] = True
    cfg["name"] = f"{cfg['name']}_{cfg['split']}"
    summary = run(cfg, workers=a.workers)
    cols = ["subset", "method", "B", "k", "n_q", "ncg_pool", "ce_ncg", f"ndcg@{cfg['eval'].get('ndcg_k', 10)}", "frac_calls_in_set"]
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.precision", 3):
        print(summary[cols].sort_values(["subset", "k", "method", "B"]).to_string(index=False))


if __name__ == "__main__":
    main()
