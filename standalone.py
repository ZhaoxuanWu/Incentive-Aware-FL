import torch

import os
import copy
import time
import random
import datetime
import argparse
import numpy as np
import logging

from collections import defaultdict
from tqdm import tqdm

from utils.configs import mnist_args, fmnist_args, cifar_cnn_args, cifar_resnet_args, svhn_args, cifar100_resnet_args, cifar100_cnn_args, sst_args, sst_lstm_args
from utils.Data_Prepper import Data_Prepper
from utils.utils import train_model, evaluate, mkdirs

parser = argparse.ArgumentParser(description='Process which dataset to run')
parser.add_argument('-D', '--dataset', help='Training dataset', type=str, required=True)
parser.add_argument('-N', '--n_clients', help='Number of clients', type=int, default=5)
parser.add_argument('-model', help='Base model', type=str, default='cnn', choices=['cnn', 'resnet18', 'lstm'])
parser.add_argument('-nocuda', dest='cuda', help='Use CPU', action='store_false')
parser.add_argument('-cuda', dest='cuda', help='Use GPU', action='store_true')
parser.add_argument('-gpu', help='GPU id', type=int, default=2)
parser.add_argument('-seed', help='seed for reproducibility', type=int, default=0)
parser.add_argument('-split', '--split', dest='split', help='The type of data split', 
                    type=str, default='iid-diff-quantity')

cmd_args = parser.parse_args()

print(cmd_args)

# Specify GPU ID
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]=str(cmd_args.gpu)

def set_seed(seed):
    # Reporducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if torch.cuda.is_available() and cmd_args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Dataset options
if cmd_args.dataset == 'mnist':
    if cmd_args.model == 'cnn':
        args = copy.deepcopy(mnist_args)
    else:
        raise NotImplementedError()
    
    args['iterations'] = 50
    args['num_classes'] = 10

elif cmd_args.dataset == 'fmnist':
    if cmd_args.model == 'cnn':
        args = copy.deepcopy(fmnist_args)
    else:
        raise NotImplementedError()

    args['iterations'] = 50
    args['num_classes'] = 10

elif cmd_args.dataset == 'svhn':
    if cmd_args.model == 'cnn':
        args = copy.deepcopy(svhn_args)
    else:
        raise NotImplementedError()

    args['iterations'] = 50
    args['num_classes'] = 10

elif cmd_args.dataset == 'cifar10':
    if cmd_args.model == 'cnn':
        args = copy.deepcopy(cifar_cnn_args)
    elif cmd_args.model in ['resnet18']:
        args = copy.deepcopy(cifar_resnet_args)
    else:
        raise NotImplementedError()

    args['iterations'] = 100
    args['num_classes'] = 10

elif cmd_args.dataset == 'cifar100':
    if cmd_args.model == 'cnn':
        args = copy.deepcopy(cifar100_cnn_args)
    elif cmd_args.model in ['resnet18']:
        args = copy.deepcopy(cifar100_resnet_args)
    else:
        raise NotImplementedError()

    args['iterations'] = 1000 if cmd_args.model == 'resnet18' else 100
    args['num_classes'] = 10

elif cmd_args.dataset == 'sst':
    if cmd_args.model == 'cnn':
        args = copy.deepcopy(sst_args)
    elif cmd_args.model == 'lstm':
        args = copy.deepcopy(sst_lstm_args)
    else:
        raise NotImplementedError()

    args['iterations'] = 100
    args['num_classes'] = 5

else:
    raise NotImplementedError()

args['local_epochs'] = 1
args['join_rate'] = np.linspace(1, 1, cmd_args.n_clients + 1)[1:]
args.update(vars(cmd_args))
print (args)

def main(args):
    # Set seeding
    set_seed(args['seed'])

    # ========== Processing ==========
    n_clients = args['n_clients']
    data_prepper = Data_Prepper(
                    args['dataset'], train_batch_size=args['batch_size'], n_clients=args['n_clients'], 
                    train_val_split_ratio=args['train_val_split_ratio'], device=device, args_dict=args)

    # Loading the data partitions indices for clients
    partition_file = 'results/partitions/{}/{}-{}-parties{}-seed{}.npz'.format(args['dataset'], args['dataset'], args['split'], args['n_clients'], args['seed'])
    net_dataidx_map = np.load(partition_file, allow_pickle=True)['net_dataidx_map'].item()
    indices_list = [net_dataidx_map[i] for i in range(n_clients)]

    # Loader    
    if 'gaussian' in args['split']:
        noise = float(args['split'].split('_')[1])
        train_loaders = data_prepper.get_train_loaders_from_indices(indices_list, noise=noise)
    else:
        train_loaders = data_prepper.get_train_loaders_from_indices(indices_list)
    test_loader = data_prepper.get_test_loader()
    shard_sizes = data_prepper.shard_sizes

    print('Total FL iterations:', args['iterations'])

    if args['dataset'] in ['sst']:
        # Extra arguments needed for LSTM model and language dataset
        server_model = args['model_fn'](args=data_prepper.args).to(device)
    else:
        server_model = args['model_fn'].to(device)
    loss_fn = args['loss_fn']

    # ========== Training the clinet models ==========
    losses = []
    accuracies = []
    progress_bar = tqdm(range(n_clients), position=0, leave=True)
    for i in progress_bar:
        model = copy.deepcopy(server_model)
        optimizer = args['optimizer_fn'](model.parameters(), lr=args['lr'], weight_decay=1e-5)
        loader = train_loaders[i]

        model.train()
        model = model.to(device)
        
        model, loss = train_model(model, loader, loss_fn, optimizer, 
                                  device=device, local_epochs=args['iterations'])
        
        model.eval()
        loss, accuracy = evaluate(model, test_loader, loss_fn=loss_fn, device=device)
        
        print(i, shard_sizes[i], loss, accuracy)
        losses.append(loss.item())
        accuracies.append(accuracy.item())

    path = 'results/standalone/{}/'.format(args['dataset'])
    mkdirs(path)
    np.savez('results/standalone/{}/{}-{}-{}-parties{}-seed{}.npz'.format(args['dataset'], args['dataset'], args['model'], args['split'], args['n_clients'], args['seed']), 
                losses=losses, accuracies=accuracies, shard_sizes=shard_sizes)

if __name__ == '__main__':
    main(args)