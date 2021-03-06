# mlflow_hydra_optuna_test
[mlflow](https://www.mlflow.org/docs/latest/index.html), [hydra](https://hydra.cc/docs/intro), [optuna](https://optuna.readthedocs.io/en/stable/)を用いたハイパーパラメータ管理

hydra >= 1.0に対応

# 準備
```bash
pip install hydra-core hydra-optuna-sweeper mlflow optuna torch torchvision
mkdir ~/src
cd ~/src
```
src下にクローン


## hydra

### コマンドラインからの値の変更

```bash
python hydra_example.py model.node1=256 model.node2=128
```

### ハイパーパラメータのグリッドサーチ

```bash
python mlflow_hydra_example.py --multirun model.node1=128,256 train.epoch=5,10
```

## mlflow

```bash
python mlflow_example.py
mlflow ui
```
[localhost](http://localhost:5000) を開いて結果を確認

## optuna

### optunaを用いたハイパーパラメータの探索

`mlflow_hydra_example.py`内で`config_example` → `MHO_config`に変更

```bash
python mlflow_hydra_example.py --multirun 'optimizer.lr=choice(0.1, 0.01, 0.001, 0.0001)' 'model.node1=range(10, 500)'
```

## hydra_optuna_sweeper

### hydra_optuna_sweeperを用いた parameter 探索
- HOS_config.yaml に探索範囲を設定
```bash
python hydra_optuna_sweeper_example.py --multirun
```

- コマンドラインから探索範囲の設定
```bash
python hydra_optuna_sweeper_example.py --multirun y=range(-10, 10)
```

### hydra_optuna_sweeperを用いた multi objective parameter 探索

- HOS_multi_objective_config.yaml に探索範囲を設定
- `binh_and_korn` を呼ぶ
```bash
python hydra_optuna_sweeper_example.py --multirun
```

## 参考

- [Hydra, MLflow, Optunaの組み合わせで手軽に始めるハイパーパラメータ管理](https://supikiti22.medium.com/hydra-mlflow-optuna%E3%81%AE%E7%B5%84%E3%81%BF%E5%90%88%E3%82%8F%E3%81%9B%E3%81%A7%E6%89%8B%E8%BB%BD%E3%81%AB%E5%A7%8B%E3%82%81%E3%82%8B%E3%83%8F%E3%82%A4%E3%83%91%E3%83%BC%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF%E7%AE%A1%E7%90%86-6b8e6d41b3da )
- [Hydraで始めるハイパラ管理](https://speakerdeck.com/supikiti/hydra-mlflow-optuna?slide=21])
- [ハイパーパラメータ管理：Mlflow と Hydra](https://udnp.hatenablog.com/entry/2021/03/06/164516)

