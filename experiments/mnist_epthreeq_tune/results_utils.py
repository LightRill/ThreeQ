from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.mnist_legacy_repro.results_utils import write_summaries as _write_summaries


def write_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    _write_summaries(df, output_dir)
