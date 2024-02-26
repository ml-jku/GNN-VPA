from typing import List

from hydra.utils import instantiate, log
from omegaconf import DictConfig
from pytorch_lightning import Callback

import numpy as np
import torch
import os
import random


def init_lightning_callbacks(cfg: DictConfig) -> List[Callback]:
    """Initialize callbacks for pytorch lightning ⚡.

    Args:
        cfg (DictConfig): The configuation for the callbacks.

    Returns:
        List[Callback]: The callbacks
    """
    callbacks: List[Callback] = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(instantiate(cb_conf))
    return callbacks


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
