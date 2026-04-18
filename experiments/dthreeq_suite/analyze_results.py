from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / "results_raw.csv")
    grouped = (
        df.groupby(["variant", "weight_lr"])["best_test_error"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean")
    )
    grouped.to_csv(RESULTS_DIR / "best_error_by_variant_lr.csv", index=False)
    grouped_md = grouped.copy()
    grouped_md["weight_lr"] = grouped_md["weight_lr"].map(lambda v: f"{v:.0e}")
    with (RESULTS_DIR / "best_error_by_variant_lr.md").open("w", encoding="utf-8") as f:
        f.write(grouped_md.round(4).to_markdown(index=False))
        f.write("\n")

    by_variant = (
        df.groupby("variant")["best_test_error"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("mean")
    )
    by_variant.to_csv(RESULTS_DIR / "best_error_by_variant.csv", index=False)
    with (RESULTS_DIR / "best_error_by_variant.md").open("w", encoding="utf-8") as f:
        f.write(by_variant.round(4).to_markdown(index=False))
        f.write("\n")

    plt.figure(figsize=(11, 5))
    for variant, sub in grouped.groupby("variant"):
        sub = sub.sort_values("weight_lr")
        plt.plot(sub["weight_lr"], sub["mean"], marker="o", label=variant)
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("weight_lr")
    plt.ylabel("mean best test error")
    plt.title("DThreeQ Two Moons Screen")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "best_error_vs_lr.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.barh(by_variant["variant"], by_variant["mean"], xerr=by_variant["std"])
    plt.xlabel("mean best test error")
    plt.title("Best Error by Variant")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "best_error_by_variant.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
