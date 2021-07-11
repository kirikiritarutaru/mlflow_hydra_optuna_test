import warnings

import hydra
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from hydra import utils
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from model import SAMPLE_DNN

warnings.simplefilter('ignore', UserWarning)


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


@hydra.main(config_path='conf', config_name='config_example')
def main(cfg):
    train = MNIST(
        utils.get_original_cwd(),
        download=True, train=True, transform=transforms.ToTensor()
    )
    print(utils.get_original_cwd())

    val = MNIST(
        utils.get_original_cwd(),
        download=True, train=False, transform=transforms.ToTensor()
    )

    trainloader = DataLoader(
        train, batch_size=cfg.train.batch_size, shuffle=True
    )
    testloader = DataLoader(val, batch_size=cfg.test.batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    model = SAMPLE_DNN(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.optimizer.lr,
        momentum=cfg.optimizer.momentum
    )

    # start new run
    mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.mlflow.runname)
    with mlflow.start_run():
        for epoch in range(cfg.train.epoch):
            # log param
            log_params_from_omegaconf_dict(cfg)
            for i, data in enumerate(trainloader):
                x, y = data
                x = x.to(device)
                y = y.to(device)
                steps = epoch * len(trainloader) + i

                outputs = model(x).to(device)
                loss = criterion(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log metric
                mlflow.log_metric("loss", loss.item(), step=steps)

            correct = 0
            total = 0
            with torch.no_grad():
                for (x, y) in testloader:
                    x = x.to(device)
                    y = y.to(device)
                    outputs = model(x).to(device)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            accuracy = float(correct / total)
            mlflow.log_metric('acc', accuracy, step=epoch)

    return accuracy


if __name__ == '__main__':
    main()
    print('Finished!')
