import hydra
from omegaconf import DictConfig


@hydra.main(config_path='conf', config_name='hydra_example')
def hydra_example(cfg: DictConfig) -> None:
    print(cfg.model.node1)
    print(cfg.model.node2)


if __name__ == '__main__':
    hydra_example()
