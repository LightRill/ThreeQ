from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threeq_common.mechanism import run_mechanism_diagnostic


def trainable(config):
    return run_mechanism_diagnostic(dict(config))
