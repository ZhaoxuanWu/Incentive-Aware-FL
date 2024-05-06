import torch
from torch import nn, optim
from utils.models import ResNet18, ResNet18GroupNorm, ResNet34, ResNet50, SimpleCNN, SimpleCNNMNIST, CNN_Text, LSTM_Text

mnist_args = {
	# setting parameters
	'dataset': 'mnist',
	'batch_size' : 64, 
	'train_val_split_ratio': 1, 

	# model parameters
	'model_fn': SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10),
	'optimizer_fn': optim.Adam,
	'loss_fn': nn.CrossEntropyLoss(), 
	'lr': 0.01,
	'lr_decay':0.977,  #0.977**100 ~= 0.1
}

fmnist_args = {
	# setting parameters
	'dataset': 'fmnist',
	'batch_size' : 64, 
	'train_val_split_ratio': 1, 

	# model parameters
	'model_fn': SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10),
	'optimizer_fn': optim.Adam,
	'loss_fn': nn.CrossEntropyLoss(), 
	'lr': 0.01,
	'lr_decay':0.977,  #0.977**100 ~= 0.1
}

svhn_args = {
	# setting parameters
	'dataset': 'svhn',
	'batch_size' : 64, 
	'train_val_split_ratio': 1,

	# model parameters
	'model_fn': SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10),
	'optimizer_fn': optim.Adam,
	'loss_fn': nn.CrossEntropyLoss(), 
	'lr': 0.001,
	'lr_decay':0.977,  #0.977**100 ~= 0.1
}

cifar_cnn_args = {
	# setting parameters
	'dataset': 'cifar10',
	'batch_size' : 64, 
	'train_val_split_ratio': 1,

	# model parameters
	'model_fn': SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10),
	'optimizer_fn': optim.Adam,
	'loss_fn': nn.CrossEntropyLoss(),
	'lr': 0.001,
	'lr_decay': 0.977,  #0.977**100 ~= 0.1
}

cifar_resnet_args = {
	# setting parameters
	'dataset': 'cifar10',
	'batch_size' : 64, 
	'train_val_split_ratio': 1,

	# model parameters
	'model_fn': ResNet18GroupNorm(in_channels=3, num_classes=10),
	'optimizer_fn': optim.Adam,
	'loss_fn': nn.CrossEntropyLoss(),
	'lr': 0.001,
	'lr_decay': 0.977,  #0.977**100 ~= 0.1
}


cifar100_cnn_args = {
	# setting parameters
	'dataset': 'cifar100',
	'batch_size' : 64, 
	'train_val_split_ratio': 1,

	# model parameters
	'model_fn': SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=100),
	'optimizer_fn': optim.Adam,
	'loss_fn': nn.CrossEntropyLoss(),
	'lr': 0.001,
	'lr_decay': 0.977,  #0.977**100 ~= 0.1

}

cifar100_resnet_args = {
	# setting parameters
	'dataset': 'cifar100',
	'batch_size' : 64, 
	'train_val_split_ratio': 1,

	# model parameters
	'model_fn': ResNet18GroupNorm(in_channels=3, num_classes=100),
	'optimizer_fn': optim.Adam,
	'loss_fn': nn.CrossEntropyLoss(),
	'lr': 0.005,
	'lr_decay': 0.988,  #0.988**200 ~= 0.1
}

sst_args = {
	# setting parameters
	'dataset': 'sst',
	'batch_size' : 64, 
	'train_val_split_ratio': 1,

	# model parameters
    'model_fn': CNN_Text,
	'embed_num': 20000,
	'embed_dim': 300,
	'class_num': 5,
	'kernel_num': 128,
	'kernel_sizes': [3,3,3],
	'static':False,

	'optimizer_fn': optim.Adam,
	'loss_fn': nn.CrossEntropyLoss(), 
	'lr': 0.001,
	'lr_decay':0.977,  #0.977**100 ~= 0.1
}

sst_lstm_args = {
	# setting parameters
	'dataset': 'sst',
	'batch_size' : 64, 
	'train_val_split_ratio': 1,

	# model parameters
    'model_fn': LSTM_Text,
	'embed_num': 20000,
	'embed_dim': 300,
	'class_num': 5,
	'static':False,

	'optimizer_fn': optim.Adam,
	'loss_fn': nn.CrossEntropyLoss(), 
	'lr': 0.001,
	'lr_decay':0.977,  #0.977**100 ~= 0.1
}
