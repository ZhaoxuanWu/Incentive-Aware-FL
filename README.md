# Incentive-Aware Federated Learning with Training-Time Model Rewards

This repository is the official implementation of the following paper accepted by the Thirty-ninth International Conference on Machine Learning (ICML) 2022:
> [***Incentive-Aware Federated Learning with Training-Time Model Rewards***](https://openreview.net/forum?id=FlY7WQ2hWS&noteId=FlY7WQ2hWS).
>
> Zhaoxuan Wu, Mohammad Mohammadi Amiri, Ramesh Raskar, Bryan Kian Hsiang Low



## Requirements

To install requirements:
```setup
conda env create -f environment.yml
```

## Preparing Datasets and Partitions

We use *MNIST, FMNIST, SVHN, CIFAR-10, CIFAR-100, STT* as benchmark datasets. The downloading and processing of the datasets will be handled by the code automatically.

We need to create data partitions for the heterogenous clients first before running federated learning algorithms on them.

The types of partitions that we support are
- Distribution-based label distribution skew: ``noniid-labeldir``
- Quantity-based label distribution skew: ``noniid-#label3`` (the number of labels can be modified)
- Noise-based feature distribution skew: ``gaussian_0.1`` (the sigma=0.1 can be modifed)
- Quantity skew: ``iid-diff-quantity``
- Homogeneous partition: ``homo``

We give one example here:
```bash
python partition.py -D mnist -N 50 -split noniid-labeldir -seeds 10
```

## Run IAFL experiments

At the beginning of the `main_IAFL.py` file, there are descriptions for the options required to run the code.

We give one example here:
```bash
python main_IAFL.py -D mnist -model cnn -split noniid-labeldir -N 50 -seed 0 -gpu 0
```
## (Optional) Standalone Accuracies

The standalone accuracies are used as a contribution measure in the baseline experiments. To obtain them, run something like
```bash
python standalone.py -D mnist -model cnn -split noniid-labeldir -N 50 -seed 0 -gpu 0
```
Then, we have already provided a commented code snippet in ``main_IAFL.py`` to use standalone accuracies as contributions. Uncomment that section of code and run the ``main_IAFL.py`` file.

## Other Baseline Methods
We implemented FedAvg finetune (FedAvg-FT), local global FedAvg (LG-FedAvg), cosine gradient Shapley value (CGSV) and rank as baseline methods for comparison.

They can be run similarly as the ``main_IAFL.py`` file. The implementations are in ``main_fedavgft.py``, ``main_lgfedavg.py``, ``main_cgsv.py`` and ``main_rank.py``, respectively.