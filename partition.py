import torch

import os
import random
import argparse
import numpy as np
from torchvision.datasets import MNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100
from torch.utils.data import Dataset
from torchtext.data import Field, LabelField
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='Process which dataset to run')
parser.add_argument('-D', '--dataset', help='Training dataset', type=str, required=True)
parser.add_argument('-N', '--n_clients', help='Number of clients', type=int, required=True)
parser.add_argument('-split', '--split', dest='split', help='The type of data split', 
                    type=str, required=True)
parser.add_argument('-seeds', help='seed for reproducibility', type=int, default=10)
args = parser.parse_args()

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def partition_data(dataset, partition, n_parties, beta=0.5, datadir=''):

    allowed_datasets = ['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100', 'sst']
    if dataset in allowed_datasets:
        X_train, y_train, X_test, y_test = load_dataset(dataset)
    else:
        raise NotImplementedError()

    y_train = np.array(y_train)
    n_train = y_train.shape[0]

    # Homogenerous partition & Noise-based feature distribution skew
    if partition == "homo" or "gaussian" in partition:
        # Perform random and equal sampling
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    # Distribution-based label distribution skew
    elif "noniid-labeldir" in partition:
        if len(partition) > len("noniid-labeldir"):
            beta = float(partition.split('-')[-1])
            print('Beta for noniid-labeldir', beta)
        min_size = 0
        min_require_size = 10

        # K is the number of classes in the dataset
        K = 10
        if dataset == 'cifar100':
            K = 100
        elif dataset == 'tinyimagenet':
            K = 200
        elif dataset == 'sst':
            K = 5

        N = y_train.shape[0]
        net_dataidx_map = {}
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                # Sampling proportions from a Dirichlet distribution
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                # Balance
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        # Shuffle before returning
        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # Quantity-based label distribution skew
    elif partition > "noniid-#label0" and partition <= "noniid-#label9":
        num = eval(partition[13:])

        # K is the number of classes in the dataset
        K = 10

        # if dataset in []: # For binary classification dasets
        #     assert num == 1
        #     num = 1
        #     K = 2

        if dataset == "cifar100":
            K = 100
        elif dataset == "tinyimagenet":
            K = 200
        elif dataset == 'sst':
            K = 5

        if num == 10:
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(10):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,n_parties)
                for j in range(n_parties):
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[j])
        else:
            times=[0 for i in range(K)]
            contain=[]
            for i in range(n_parties):
                if num==1 and K==2:
                    # Add stochasticity in this case
                    selected=random.randint(0,1)
                    current=[selected]
                    times[selected]+=1
                else:
                    current=[i%K]
                    times[i%K]+=1
                j=1
                while (j<num):
                    ind=random.randint(0,K-1)
                    if (ind not in current):
                        j=j+1
                        current.append(ind)
                        times[ind]+=1
                contain.append(current)
            net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
            for i in range(K):
                idx_k = np.where(y_train==i)[0]
                np.random.shuffle(idx_k)
                split = np.array_split(idx_k,times[i])
                ids=0
                for j in range(n_parties):
                    if i in contain[j]:
                        net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                        ids+=1

    # Quantity skew
    elif partition == "iid-diff-quantity":
        idxs = np.random.permutation(n_train)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}
        

    # Print partition statistics
    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

def record_net_data_stats(y_train, net_dataidx_map):

    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    print('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts


##### DATASET UTILS #####
def load_dataset(name='mnist'):
    if name == 'mnist':
        train = MNIST('.data', train=True, download=True)
        test = MNIST('.data', train=False, download=True)
        return train.data, train.targets, test.data, test.targets
    elif name == 'fmnist':
        train = FashionMNIST('.data', train=True, download=True)
        test = FashionMNIST('.data', train=False, download=True)
        return train.data, train.targets, test.data, test.targets
    elif name == 'svhn':
        train = SVHN('.data', 'train', download=True)
        test = SVHN('.data', 'test', download=True)
        return train.data, train.labels, test.data, test.labels
    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        train = CIFAR10('.data', train=True, download=True, transform=transform_train)
        test = CIFAR10('.data', train=False, download=True)
        return train.data, train.targets, test.data, test.targets
    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
        train = CIFAR100('.data', train=True, download=True, transform=transform_train)
        test = CIFAR100('.data', train=False, download=True)
        return train.data, train.targets, test.data, test.targets
    elif name == 'sst':
        from torchtext.datasets import SST
        text_field = Field(lower=True)
        label_field = LabelField(dtype=torch.long, sequential=False)
        train_data, validation_data, test_data = SST.splits(text_field, label_field, root='.data', fine_grained=True)

        label_to_num = {'very negative': 0, 'negative': 1, 'neutral': 2, 'positive': 3, 'very positive': 4}
        X_train = [text for text in train_data.text]
        y_train = [label_to_num[label] for label in train_data.label]
        X_test = [text for text in test_data.text]
        y_test = [label_to_num[label] for label in test_data.label]

        return X_train, y_train, X_test, y_test
    else:
        raise NotImplementedError()

def set_seed(seed):
    # Reporducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    path = 'results/partitions/{}'.format(args.dataset)
    mkdirs(path)
    n_party = 50
    for i in range(args.seeds):
        set_seed(i)
        (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts) = partition_data(args.dataset, args.split, args.n_clients)
        print('{}/{}-{}-parties{}-seed{}.npz'.format(path, args.dataset, args.split, args.n_clients, i))
        np.savez('{}/{}-{}-parties{}-seed{}.npz'.format(path, args.dataset, args.split, args.n_clients, i),
                 net_dataidx_map=net_dataidx_map,
                 traindata_cls_counts=traindata_cls_counts,
                 )
