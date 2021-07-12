import hydra
import numpy as np
from omegaconf import DictConfig


# x, y を通る原点が中心の円の面積
@hydra.main(config_path='conf', config_name='HOS_config')
def circle_area(cfg: DictConfig) -> float:
    x: float = cfg.x
    y: float = cfg.y
    return (x**2+y**2) * np.pi


if __name__ == '__main__':
    circle_area()
