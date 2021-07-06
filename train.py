import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='conf', config_name='config_example')
def hydra_sample(cfg: DictConfig) -> None:
    print(cfg.model.node1)
    print(cfg.model.node2)


@hydra.main(config_path='conf', config_name='config')
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == '__main__':
    my_app()
