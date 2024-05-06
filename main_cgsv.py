import torch

import os
import copy
import math
import time
import random
import datetime
import argparse
import numpy as np
import logging
import torch.nn.functional as F

from collections import defaultdict
from tqdm import tqdm
from torch.linalg import norm

from utils.configs import mnist_args, fmnist_args, cifar_cnn_args, cifar_resnet_args, svhn_args, cifar100_cnn_args, cifar100_resnet_args, sst_args, sst_lstm_args
from utils.Data_Prepper import Data_Prepper
from utils.utils import train_model, compute_grad_update, compute_grad_update_clip, add_gradient_updates, add_update_to_model, evaluate, mkdirs

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

if cmd_args.dataset == 'mnist':
    if cmd_args.model == 'cnn':
        args = copy.deepcopy(mnist_args)
    else:
        raise NotImplementedError()
    
    args['iterations'] = 50
    args['num_classes'] = 10
    args['Gamma'] = 0.5

elif cmd_args.dataset == 'fmnist':
    if cmd_args.model == 'cnn':
        args = copy.deepcopy(fmnist_args)
    else:
        raise NotImplementedError()

    args['iterations'] = 50
    args['num_classes'] = 10
    args['Gamma'] = 0.5

elif cmd_args.dataset == 'svhn':
    if cmd_args.model == 'cnn':
        args = copy.deepcopy(svhn_args)
    else:
        raise NotImplementedError()

    args['iterations'] = 50
    args['num_classes'] = 10
    args['Gamma'] = 0.5

elif cmd_args.dataset == 'cifar10':
    if cmd_args.model == 'cnn':
        args = copy.deepcopy(cifar_cnn_args)
    elif cmd_args.model in ['resnet18']:
        args = copy.deepcopy(cifar_resnet_args)
    else:
        raise NotImplementedError()

    args['iterations'] = 100
    args['num_classes'] = 10
    args['Gamma'] = 0.15

elif cmd_args.dataset == 'cifar100':
    if cmd_args.model == 'cnn':
        args = copy.deepcopy(cifar100_cnn_args)
    elif cmd_args.model in ['resnet18']:
        args = copy.deepcopy(cifar100_resnet_args)
    else:
        raise NotImplementedError()

    args['iterations'] = 1000 if cmd_args.model == 'resnet18' else 100
    args['num_classes'] = 10
    args['Gamma'] = 0.15

elif cmd_args.dataset == 'sst':
    if cmd_args.model == 'cnn':
        args = copy.deepcopy(sst_args)
    elif cmd_args.model == 'lstm':
        args = copy.deepcopy(sst_lstm_args)
    else:
        raise NotImplementedError()

    args['iterations'] = 100
    args['num_classes'] = 5
    args['Gamma'] = 1

else:
    raise NotImplementedError()

args['local_epochs'] = 1
args['join_rate'] = np.linspace(1, 1, cmd_args.n_clients + 1)[1:]
args['alpha'] = 0.95
args['beta'] = 1

args.update(vars(cmd_args))
print (args)

def main(args):
    # Set seeding
    set_seed(args['seed'])

    # Logging information
    save_filename = '{}_{}_{}_epoch{}_lr{}_{}_seed{}'.format(args['dataset'], args['n_clients'], args['split'], args['iterations'], args['lr'], args['model'], args['seed'])
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

    # Shard sizes refer to the sizes of the local data of each agent
    shard_sizes = data_prepper.shard_sizes
    shard_sizes = torch.tensor(shard_sizes).float()
    print("Shard sizes are: ", shard_sizes.tolist())

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

    rs_dict, qs_dict = [], []
    rs = torch.zeros(n_clients, device=device)
    past_phis = []
    
    # ========== FL training ==========
    progress_bar = tqdm(range(args['iterations']), position=0, leave=True)
    for iteration in progress_bar:

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

            # append the normalzied gradient
            flattened = flatten(gradient)
            norm_value = norm(flattened) + 1e-7 # to prevent division by zero
                    
            gradient = unflatten(torch.multiply(torch.tensor(args['Gamma']), torch.div(flattened,  norm_value)), gradient)
            
            gradients.append(gradient)
            
            # Revert to before local update
            model.load_state_dict(backup.state_dict())


        # ---------- Server Aggregate ---------
        aggregated_gradient = {}
        server_global_dict = server_model.state_dict()
        for k in server_global_dict.keys():
            aggregated_gradient[k] = torch.zeros(server_global_dict[k].shape).to(device)

        if iteration == 0:
            # First iteration use FedAvg
            weights = torch.div(shard_sizes, torch.sum(shard_sizes))
        else:
            weights = rs

        for gradient, weight in zip(gradients, weights):
            add_gradient_updates(aggregated_gradient, gradient, weight=weight)

        add_update_to_model(server_model, aggregated_gradient)


        # Update reputation and calculate reward gradients
        flat_aggre_grad = flatten(aggregated_gradient)
        phis = torch.tensor([F.cosine_similarity(flatten(gradient), flat_aggre_grad, 0, 1e-10) for gradient in gradients], device=device)
        past_phis.append(phis)

        rs = args['alpha'] * rs + (1 - args['alpha']) * phis
        rs = torch.clamp(rs, min=1e-3) # Make sure the rs do not go negative
        rs = torch.div(rs, rs.sum()) # Normalize the weights to 1 
        
        # Altruistic degree function
        q_ratios = torch.tanh(args['beta'] * rs) 
        q_ratios = torch.div(q_ratios, torch.max(q_ratios))
        
        qs_dict.append(q_ratios)
        rs_dict.append(rs)       
                    
        # ---------- Client Rewards ----------
        for i in range(n_clients):
            reward_gradient = mask_grad_update_by_order(aggregated_gradient, mask_percentile=q_ratios[i], mode='layer')

            add_update_to_model(agent_models[i], reward_gradient)

        weights = torch.div(shard_sizes, torch.sum(shard_sizes)) if iteration == 0 else rs
            
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

    mkdirs('results/cgsv')
    np.savez('results/cgsv/{}.npz'.format(save_filename),
                valid_perfs=valid_perfs, local_perfs=local_perfs, fed_perfs=fed_perfs,
                join_indicators=join_indicators, args=args)


def flatten(grad_update):
    return torch.cat([grad_update[name].data.view(-1) for name in grad_update])

def unflatten(flattened, normal_shape):
    grad_update = {}
    for name in normal_shape:
        param = normal_shape[name]
        n_params = len(param.view(-1))
        grad_update[name] = torch.as_tensor(flattened[:n_params]).reshape(param.size())
        flattened = flattened[n_params:]
    return grad_update

def cosine_similarity(grad1, grad2, normalized=False):
	"""
	Input: two sets of gradients of the same shape
	Output range: [-1, 1]
	"""

	cos_sim = F.cosine_similarity(flatten(grad1), flatten(grad2), 0, 1e-10) 
	if normalized:
		return (cos_sim + 1) / 2.0
	else:
		return cos_sim

def mask_grad_update_by_order(grad_update, mask_order=None, mask_percentile=None, mode='all'):
	if mode == 'all':
		# mask all but the largest <mask_order> updates (by magnitude) to zero
		all_update_mod = torch.cat([update.data.view(-1).abs()
									for update in grad_update])
		if not mask_order and mask_percentile is not None:
			mask_order = int(len(all_update_mod) * mask_percentile)
		
		if mask_order == 0:
			return mask_grad_update_by_magnitude(grad_update, float('inf'))
		else:
			topk, indices = torch.topk(all_update_mod, mask_order)
			return mask_grad_update_by_magnitude(grad_update, topk[-1])

	elif mode == 'layer': # layer wise largest-values criterion
		grad_update = copy.deepcopy(grad_update)

		mask_percentile = max(0, mask_percentile)
		for i, layer in enumerate(grad_update):
			layer_mod = grad_update[layer].data.view(-1).abs()
			if mask_percentile is not None:
				mask_order = math.ceil(len(layer_mod) * mask_percentile)

			if mask_order == 0:
				grad_update[layer].data = torch.zeros(grad_update[layer].data.shape, device=grad_update[layer].device)
			else:
				topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))																																												
				grad_update[layer].data[grad_update[layer].data.abs() < topk[-1]] = 0
		return grad_update

def mask_grad_update_by_magnitude(grad_update, mask_constant):
	# mask all but the updates with larger magnitude than <mask_constant> to zero
	# print('Masking all gradient updates with magnitude smaller than ', mask_constant)
	grad_update = copy.deepcopy(grad_update)
	for i, update in enumerate(grad_update):
		grad_update[i].data[update.data.abs() < mask_constant] = 0
	return grad_update

if __name__ == '__main__':
    main(args)