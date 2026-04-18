from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.mnist_dthreeq_supervision_budget.results_utils import write_summaries


def write_fullbudget_summaries(df: pd.DataFrame, output_dir: Path) -> None:
    write_summaries(df, output_dir)
