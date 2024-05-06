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
from utils.utils import train_model, compute_grad_update, compute_grad_update_clip, evaluate, add_gradients_to_model, add_gradients_to_model_batch, mkdirs

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
parser.add_argument('-q', help='Stochasticity parameter', type=float, default=0)
parser.add_argument('-kappa', help='Sharing parameter', type=float, default=0)

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
print(args)

def main(args):
    # Set seeding
    set_seed(args['seed'])
    
    # Logging information
    save_filename = '{}_{}_{}_epoch{}_lr{}_{}_share_{}_stochastic_{}_seed{}'.format(args['dataset'], args['n_clients'], args['split'], args['iterations'], args['lr'], args['model'], args['kappa'], args['q'], args['seed'])
    print('Filename', save_filename)
    mkdirs('results/logs')
    logging.basicConfig(level=logging.INFO, filename='results/logs/{}.log'.format(save_filename))
    
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

    # Reward weights
    # Option 1: naively use the shared/data sizes as the contribution measure, this does not work with homo and gaussian where shard sizes are all the same
    shard_sizes = data_prepper.shard_sizes
    p_i = shard_sizes = torch.tensor(shard_sizes).float()
    p_ceil = torch.max(shard_sizes)
    reward_weights_gamma = torch.clip(p_i/p_ceil, max=1)

    # # Option 2: use the standalone accuracies (need to run standalone.py and load it here)
    # loaded_p_i = np.load('results/standalone/{}/{}-{}-{}-parties{}-seed{}.npz'.format(args['dataset'], args['dataset'], args['model'], args['split'], args['n_clients'], args['seed']))['accuracies']
    # p_i = torch.from_numpy(loaded_p_i)
    # p_ceil = torch.max(p_i)
    # reward_weights_gamma = torch.clip(p_i/p_ceil, max=1)

    # # Option 3: any other contribution measure, here shows an example using the participation rate as the contribution measure
    # # To use this, modify the args['join_rate'] definition above
    # p_i = args['join_rate']
    # p_i = torch.from_numpy(p_i)
    # p_ceil = torch.max(p_i)
    # reward_weights_gamma = torch.clip(p_i/p_ceil, max=1)

    if args['dataset'] in ['sst']:
        # Extra arguments needed for LSTM model and language dataset
        server_model = args['model_fn'](args=data_prepper.args).to(device)
    else:
        server_model = args['model_fn'].to(device)
    loss_fn = args['loss_fn']

    # ========== Initalize the clinets ==========
    agent_models, agent_optimizers, agent_schedulers = [], [], []

    for i in range(n_clients):
        model = copy.deepcopy(server_model)

        optimizer = args['optimizer_fn'](model.parameters(), lr=args['lr'], weight_decay=1e-5)

        # Exponential LR decay
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = args['lr_decay'])

        agent_models.append(model)
        agent_optimizers.append(optimizer)
        agent_schedulers.append(scheduler)
    
    # ========== Logging ==========
    # For performance analysis
    valid_perfs, local_perfs, fed_perfs = defaultdict(list), defaultdict(list), defaultdict(list)
    # For checking
    join_indicators = []
    
    # ========== FL training ==========
    progress_bar = tqdm(range(args['iterations']), position=0, leave=True)
    for iteration in progress_bar:

        # ---------- Stochasticity ----------            

        # Added stochasticity
        for i in range(n_clients):
            if np.random.rand() < args['q']:
                agent_models[i] = copy.deepcopy(server_model)

        # -----------------------------------

        join_indicator = [np.random.rand() < args['join_rate'][i] for i in range(n_clients)]
        join_indicator = torch.tensor(join_indicator).int()
        join_indicators.append(join_indicator)
        
        # --------- Client updates ----------
        gradients = []
        for i in range(n_clients):
            if join_indicator[i].item() == 0:
                # Clinet not available, 1 indicates join 
                gradient = {}
                global_dict = server_model.state_dict()
                for k in global_dict.keys():
                    gradient[k] = torch.zeros(global_dict[k].shape).to(device)
                gradients.append(gradient)
                agent_schedulers[i].step()
                continue 
                
            loader = train_loaders[i]
            model = agent_models[i]
            scheduler = agent_schedulers[i]
            optimizer = args['optimizer_fn'](model.parameters(), lr=scheduler.get_last_lr()[-1])
            loss_fn = args['loss_fn']

            model.train()
            model = model.to(device)
            backup = copy.deepcopy(model)
            
            model, loss = train_model(model, loader, loss_fn, optimizer, 
                                        device=device, local_epochs=args['local_epochs'], 
                                        scheduler=scheduler)
                            
            gradient = compute_grad_update(old_model=backup, new_model=model, device=device)
            # # Optionally to clip
            # gradient = compute_grad_update_clip(old_model=backup, new_model=model, device=device)

            gradients.append(gradient)
            
            # Revert to before local update
            model.load_state_dict(backup.state_dict())    
                    
        # ---------- Client Rewards ----------
        r_weights = []
        reward_proportions = reward_weights_gamma
        reward_proportions = reward_proportions ** (1 - args['kappa'])
        reward_size = torch.ceil(reward_proportions * (torch.sum(join_indicator)-1))
        print("Join_indicator:", join_indicator)
        print("Reward_size:", reward_size)

        for i in range(n_clients):
            tmp_join_indicator = join_indicator.clone()
            tmp_join_indicator[i] = 0
            included_clients = np.random.choice(torch.where(tmp_join_indicator==1)[0], 
                                                int(reward_size[i]),
                                                replace=False)
            inclusion_indicator = torch.zeros(n_clients).int()
            for idx in included_clients:
                inclusion_indicator[idx] = 1
            if join_indicator[i] == 1:
                # Use a client's own gradient if it joins this iteration
                inclusion_indicator[i] = 1
                      
            if torch.sum(inclusion_indicator) == 0:
                # No gradient will be rewarded
                reward_weights = inclusion_indicator
            else:
                reward_weights = torch.div(inclusion_indicator, torch.sum(inclusion_indicator))
            
            r_weights.append(reward_weights)
                    
        add_gradients_to_model_batch(agent_models, gradients, r_weights)

        # ---------- Server updates ----------
        
        server_reward_weights = r_weights[torch.argmax(reward_proportions)] # Max reference
        # server_reward_weights = r_weights[reward_proportions.sort()[1][n_clients//2]] # Median reference
                    
        add_gradients_to_model(server_model, gradients, server_reward_weights)
            
        # ---------- Validation & Testing ----------            
        if iteration % 1 == 0:    
            for i, model in enumerate(agent_models + [server_model]):
                
                loss, accuracy = evaluate(model, test_loader, loss_fn=loss_fn, device=device)
                if i == len(agent_models):
                    print('Loss:{}, Accu:{}'.format(loss, accuracy))
                    logging.info('Loss:{}, Accu:{}'.format(loss, accuracy))
                elif i % 1 == 0:
                    print('Agent {}. Loss:{}, Accu:{}'.format(i, loss, accuracy))
                    logging.info('Agent {}. Loss:{}, Accu:{}'.format(i, loss, accuracy))

                valid_perfs[str(i)+'_loss'].append(loss.item())
                valid_perfs[str(i)+'_accu'].append(accuracy.item())

        progress_bar.set_description("{:2}".format(iteration))
    
    mkdirs('results/IAFL')
    np.savez('results/IAFL/{}.npz'.format(save_filename),
             valid_perfs=valid_perfs, local_perfs=local_perfs, fed_perfs=fed_perfs,
             join_indicators=join_indicators, args=args)

if __name__ == '__main__':
    main(args)