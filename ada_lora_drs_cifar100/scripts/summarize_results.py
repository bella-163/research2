import argparse
from pathlib import Path
import pandas as pd


def parse_run_name(name: str):
    if name.startswith("s") and "_" in name:
        first, rest = name.split("_", 1)
        try:
            return int(first[1:]), rest
        except ValueError:
            return None, name
    return None, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for p in sorted(root.glob("*/metrics.csv")):
        df = pd.read_csv(p)
        if df.empty:
            continue
        last = df.tail(1).copy()
        seed, method = parse_run_name(p.parent.name)
        last.insert(0, "seed", seed)
        last.insert(1, "method", method)
        rows.append(last)

    if not rows:
        raise SystemExit(f"No metrics.csv found under {root}")

    all_df = pd.concat(rows, ignore_index=True)
    cols = [c for c in ["average_accuracy", "final_accuracy", "forgetting", "feature_drift", "train_loss", "ce_loss", "atl_loss"] if c in all_df.columns]
    summary = all_df.groupby("method")[cols].agg(["mean", "std", "count"])

    print("\n=== Final-stage results ===")
    print(all_df.to_string(index=False))
    print("\n=== Mean / Std / Count ===")
    print(summary)

    all_df.to_csv(root / "all_results.csv", index=False)
    summary.to_csv(root / "summary_mean_std.csv")
    print(f"\nSaved: {root / 'all_results.csv'}")
    print(f"Saved: {root / 'summary_mean_std.csv'}")


if __name__ == "__main__":
    main()
