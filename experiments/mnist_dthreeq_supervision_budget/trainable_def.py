from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from threeq_common.mnist_dthreeq_focus import train_one_mnist_dthreeq_focus


def trainable(config):
    return train_one_mnist_dthreeq_focus(dict(config))
