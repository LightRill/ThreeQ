from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import radas


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mechanism_diagnostic.results_utils import write_summaries
from experiments.mechanism_diagnostic.suite_config import EXPERIMENT_NAME, param_space


SCRIPT_DIR = Path(__file__).resolve().parent


async def main() -> None:
    os.chdir(ROOT)
    output_dir = SCRIPT_DIR / "results"
    results = await radas.restore_and_materialize(
        experiment_name=EXPERIMENT_NAME,
        param_space=param_space(),
        output_dir=output_dir,
    )
    write_summaries(results["df"], output_dir)
    print("restore_source:", results.get("restore_source"))
    print(results["df"])


if __name__ == "__main__":
    asyncio.run(main())
