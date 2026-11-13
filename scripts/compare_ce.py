"""Compare cross-encoders on one split (model selection: use DEV only).

    python scripts/compare_ce.py configs/baselines.yaml --split dev \
        --ce cross-encoder/ms-marco-MiniLM-L-6-v2 --ce-fp16 BAAI/bge-reranker-base ...

For each CE: Full-CE (rank the whole pool by CE) vs MIPS on human nCG@k (paired bootstrap CI),
SingleStage at the configured budgets, and the CE score distribution (relevant to the
calibration issue in docs/FINDINGS.md F3).
"""
import argparse
import copy
import logging

import numpy as np
import pandas as pd

from pbar.experiment import load_config, run
from pbar.stats import paired_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--ce", nargs="*", default=[], help="fp32-cached CE models")
    ap.add_argument("--ce-fp16", nargs="*", default=[], help="fp16-cached CE models")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    base = load_config(a.config)
    base["split"] = a.split
    base["methods"] = [{"name": "mips"}, {"name": "single_stage"}, {"name": "full_ce"}]
    rows = []
    for model, fp16 in [(m, False) for m in a.ce] + [(m, True) for m in a.ce_fp16]:
        cfg = copy.deepcopy(base)
        cfg["cross_encoder"].update(model=model, fp16=fp16)
        cfg["name"] = f"cecmp_{a.split}_{model.split('/')[-1]}"
        run(cfg)
        from pathlib import Path
        from pbar.experiment import config_hash
        df = pd.read_csv(Path(cfg["results_dir"]) / f"{cfg['name']}_{config_hash(cfg)}" / "per_query.csv")
        from pbar.experiment import prepare
        _, ctxs, ce = prepare(cfg)
        allce = np.concatenate([ce[c.qid] for c in ctxs])
        for k in cfg["eval"]["k"]:
            mips = df[(df.method == "mips") & (df.k == k)][["qid", "ncg_pool"]]
            full = df[(df.method == "full_ce") & (df.k == k)][["qid", "ncg_pool"]]
            m = full.merge(mips, on="qid", suffixes=("_ce", "_mips"))
            t = paired_test(m.ncg_pool_ce.values, m.ncg_pool_mips.values)
            rows.append({"ce": model.split("/")[-1], "fp16": fp16, "k": k,
                         "mips_ncg": 100 * m.ncg_pool_mips.mean(), "fullce_ncg": 100 * m.ncg_pool_ce.mean(),
                         "delta_pp": 100 * t["delta"], "ci": f"[{100*t['ci_lo']:+.2f}, {100*t['ci_hi']:+.2f}]",
                         "ce_median": float(np.median(allce)), "ce_p90": float(np.quantile(allce, .9)),
                         "ce_frac_gt_0.5": float((allce > 0.5).mean())})
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.precision", 3):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
