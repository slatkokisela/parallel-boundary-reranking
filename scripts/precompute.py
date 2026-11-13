"""Precompute corpus/query embeddings and CE score caches (run on GPU if available).

    python scripts/precompute.py configs/baselines.yaml --splits dev test --pool-size 1000

Scores only pairs not already cached. Caches are keyed by corpus doc index, so an
N=1000 cache also serves N=500 runs. Copy `cache/` between machines freely.
"""
import argparse
import logging
import time

from pbar.experiment import load_config, prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--pool-size", type=int)
    ap.add_argument("--max-length", type=int, help="override CE max_length")
    ap.add_argument("--ce-model", help="override cross-encoder model")
    ap.add_argument("--fp16", action="store_true", help="CE under fp16 autocast (GPU only)")
    ap.add_argument("--ce-batch-size", type=int)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for split in a.splits:
        cfg = load_config(a.config)
        cfg["split"] = split
        if a.pool_size:
            cfg["pool_size"] = a.pool_size
        if a.max_length:
            cfg["cross_encoder"]["max_length"] = a.max_length
        if a.ce_model:
            cfg["cross_encoder"]["model"] = a.ce_model
        if a.fp16:
            cfg["cross_encoder"]["fp16"] = True
        if a.ce_batch_size:
            cfg["cross_encoder"]["batch_size"] = a.ce_batch_size
        t = time.time()
        _, ctxs, _ = prepare(cfg)
        logging.info("%s: %d queries x N=%d done in %.0fs", split, len(ctxs), cfg["pool_size"],
                     time.time() - t)


if __name__ == "__main__":
    main()
