"""Paired analysis of a quality run against a reference method (default: sequential MORPH).

    python scripts/analyze.py results/batch_dev_<hash> [--metric ce_ncg] [--subset all|diffuse]

For every (method, params, B, k), it reports the mean metric and the paired delta vs the
reference at the same (B, k), with a bootstrap 95% CI, a Wilcoxon p-value and a
Holm-adjusted p-value across all comparisons in the table. Writes analysis_<metric>_<subset>.csv/.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pbar.stats import holm, paired_test


def label(row) -> str:
    p = json.loads(row["params"])
    b = p.pop("batch_size", None)
    extra = ",".join(f"{k}={v}" for k, v in sorted(p.items()))
    return row["method"] + (f"[b={b}]" if b else "") + (f"[{extra}]" if extra else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--metric", default="ce_ncg")
    ap.add_argument("--subset", default="all", choices=["all", "diffuse"])
    ap.add_argument("--ref", default='morph|{"batch_size": 1}')
    ap.add_argument("--diffuse-min-rel", type=int, default=20)
    a = ap.parse_args()
    run = Path(a.run_dir)
    df = pd.read_csv(run / "per_query.csv")
    if a.subset == "diffuse":
        df = df[df.n_rel >= a.diffuse_min_rel]
    df["label"] = df.apply(label, axis=1)
    ref_m, ref_p = a.ref.split("|")
    ref = df[(df.method == ref_m) & (df.params == ref_p)]
    rows = []
    for (lab, B, k), g in df.groupby(["label", "B", "k"]):
        r = ref[(ref.k == k) & (ref.B == B)] if not np.isnan(B) else ref.iloc[0:0]
        row = {"method": lab, "B": B, "k": k, "n_q": g.qid.nunique(), "mean": g[a.metric].mean(),
               "rounds": g.rounds.mean()}
        if len(r) and lab != ref.iloc[0]["label"]:
            m = g.merge(r[["qid", a.metric]], on="qid", suffixes=("", "_ref"))
            t = paired_test(m[a.metric].values, m[a.metric + "_ref"].values)
            row.update(delta=t["delta"], ci_lo=t["ci_lo"], ci_hi=t["ci_hi"], p=t["wilcoxon_p"])
        rows.append(row)
    out = pd.DataFrame(rows)
    mask = out.get("p", pd.Series(dtype=float)).notna()
    if mask.any():
        out.loc[mask, "p_holm"] = holm(out.loc[mask, "p"].tolist())
    out = out.sort_values(["k", "B", "method"])
    stem = run / f"analysis_{a.metric}_{a.subset}"
    out.to_csv(stem.with_suffix(".csv"), index=False)
    fmt = out.copy()
    for c in ["mean", "delta", "ci_lo", "ci_hi"]:
        if c in fmt:
            fmt[c] = (fmt[c] * 100).round(2)          # percentage points
    stem.with_suffix(".md").write_text(fmt.to_markdown(index=False, floatfmt=".4g"))
    with pd.option_context("display.width", 220, "display.max_rows", 500):
        print(fmt.to_string(index=False))


if __name__ == "__main__":
    main()
