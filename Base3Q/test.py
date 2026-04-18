import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
