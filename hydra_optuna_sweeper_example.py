from typing import Tuple

import hydra
import numpy as np
from omegaconf import DictConfig


# x, y を通る原点が中心の円の面積
@hydra.main(config_path='conf', config_name='HOS_config')
def circle_area(cfg: DictConfig) -> float:
    x: float = cfg.x
    y: float = cfg.y
    return (x**2+y**2) * np.pi


@hydra.main(config_path='conf', config_name='HOS_multi_objective_config')
def binh_and_korn(cfg: DictConfig) -> Tuple[float, float]:
    x: float = cfg.x
    y: float = cfg.y

    v0 = 4 * x ** 2 + 4 * y ** 2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


if __name__ == '__main__':
    # circle_area()
    binh_and_korn()
