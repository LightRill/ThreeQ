from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mnist_dthreeq_activation_boost.results_utils import write_summaries
from experiments.mnist_dthreeq_activation_boost.suite_config import EXPERIMENT_NAME


SCRIPT_DIR = Path(__file__).resolve().parent


def _candidate_dirs() -> list[Path]:
    home = Path.home()
    roots = [
        home / "ray_results",
        home / "radas_results",
        Path("/mnt/data/nfs/ray-results/yuhang"),
    ]
    out = []
    for root in roots:
        if root.exists():
            out.extend(root.glob(f"**/{EXPERIMENT_NAME}*"))
    return sorted(set(out), key=lambda path: path.stat().st_mtime, reverse=True)


def _find_progress_csv() -> Path:
    override = os.environ.get("RADAS_RESULT_DIR")
    candidates = [Path(override)] if override else _candidate_dirs()
    for directory in candidates:
        files = sorted(directory.glob("**/progress.csv"))
        if files:
            return files[0]
    raise FileNotFoundError(f"Could not find progress.csv for {EXPERIMENT_NAME}.")


def main() -> None:
    progress = _find_progress_csv()
    df = pd.read_csv(progress)
    write_summaries(df, SCRIPT_DIR / "results")
    print(f"loaded: {progress}")
    print(df)


if __name__ == "__main__":
    main()
