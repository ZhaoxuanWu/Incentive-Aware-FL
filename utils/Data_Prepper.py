import copy
import math
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchtext.data import Field, LabelField, BucketIterator
from torchtext.data import Dataset as TextDataset


class Data_Prepper:
    def __init__(self, name, train_batch_size, n_clients,
                 test_batch_size=1000, valid_batch_size=None, 
                 train_val_split_ratio=1, device=None, args_dict=None):
        self.name = name
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_batch_size = valid_batch_size if valid_batch_size else test_batch_size
        self.n_clients = n_clients
        self.train_val_split_ratio = train_val_split_ratio
        self.device = device
        self.args = None
        self.args_dict = args_dict
        self.allowed_datasets = ['mnist', 'fmnist', 'svhn', 'cifar10', 'cifar100', 'sst']

        if name in ['mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn']:
            # Vision datastes
            self.train_dataset, self.validation_dataset, self.test_dataset, _, _, _ = self.prepare_dataset(name)

            print('------')
            print("Train to split size: {}. Validation size: {}. Test size: {}".format(len(self.train_dataset), len(self.validation_dataset), len(self.test_dataset)))
            print('------')

            self.valid_loader = DataLoader(self.validation_dataset, batch_size=self.test_batch_size)
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size)
        elif name in ['sst']:
            # Language datasets
            self.args  = {}
            self.train_dataset, self.validation_dataset, self.test_dataset = self.prepare_dataset(name)
            
            if name == 'sst':
                self.valid_loader = BucketIterator(self.validation_dataset, batch_size = self.test_batch_size, sort_key=lambda x: len(x.text), device=self.device)
            self.test_loader = BucketIterator(self.test_dataset, batch_size = self.test_batch_size, sort_key=lambda x: len(x.text), device=self.device)
            
            keys_to_copy = ['embed_dim', 'kernel_num', 'kernel_sizes', 'static']
            for key in keys_to_copy:
                if key in self.args_dict.keys():
                    self.args[key] = self.args_dict[key]

            print("Model embedding arguments:", self.args)
            print('------')
            print("Train to split size: {}. Validation size: {}. Test size: {}".format(len(self.train_dataset), len(self.validation_dataset) if self.validation_dataset is not None else 0, len(self.test_dataset)))
            print('------')
        else:
            raise NotImplementedError()
        
    def prepare_dataset(self, name='mnist'):
        if name == 'sst':
            from torchtext.datasets import SST
            if 'kernel_sizes' in self.args_dict.keys():
                # Padding to prevent min length smaller than the kernel size
                min_len_padding = get_pad_to_min_len_fn(min_length=max(self.args_dict['kernel_sizes']))
                text_field = Field(lower=True, postprocessing=min_len_padding, include_lengths=True, batch_first=True)
            else:
                text_field = Field(lower=True, include_lengths=True, batch_first=True)
            label_field = LabelField(dtype=torch.long, sequential=False)
            train_data, validation_data, test_data = SST.splits(text_field, label_field, root='.data', fine_grained=True)

            text_field.build_vocab( *([train_data, validation_data, test_data]))
            label_field.build_vocab( *([train_data, validation_data, test_data]))
            self.args['embed_num'] = len(text_field.vocab)
            self.args['class_num'] = len(label_field.vocab)
            self.args['fields'] = train_data.fields

            return train_data, validation_data, test_data
        elif name == 'mnist':
            transform_train_list = []
            train = FastMNIST('.data', train=True, download=True)
            test = FastMNIST('.data', train=False, download=True)
        elif name == 'fmnist':
            transform_train_list = []
            train = FastFMNIST('.data', train=True, download=True)
            test = FastFMNIST('.data', train=False, download=True)
        elif name == 'svhn':
            transform_train_list = []
            train = FastSVHN('.data', 'train', download=True)
            test = FastSVHN('.data', 'test', download=True)
        elif name == 'cifar10':
            transform_train_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            transform_train = transforms.Compose(transform_train_list)

            train = FastCIFAR10('.data', train=True, download=True)
            test = FastCIFAR10('.data', train=False, download=True)
        elif name == 'cifar100':
            transform_train_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15)
            ]
            transform_train = transforms.Compose(transform_train_list)

            train = FastCIFAR100('.data', train=True, download=True)
            test = FastCIFAR100('.data', train=False, download=True)
        else:
            raise NotImplementedError()
        
        train_indices, valid_indices = get_train_valid_indices(len(train), self.train_val_split_ratio)
        
        if len(transform_train_list) == 0:
            train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
        else:
            train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device, transform=transform_train)
        validation_set = Custom_Dataset(train.data[valid_indices],train.targets[valid_indices] , device=self.device)
        test_set = Custom_Dataset(test.data, test.targets, device=self.device)

        return train_set, validation_set, test_set, train, test, transform_train_list

    def get_valid_loader(self):
        return self.valid_loader

    def get_test_loader(self):
        return self.test_loader

    def get_train_loaders_from_indices(self, indices_list, batch_size=None, noise=None):
        if not batch_size:
            batch_size = self.train_batch_size
        
        if self.name not in self.allowed_datasets: 
            raise NotImplementedError()

        self.shard_sizes = [len(indices) for indices in indices_list]

        if self.name in ['sst']:
            if noise is not None:
                raise NotImplementedError('Gaussian feature noise does not make sense for text data.')
            datasets = []
            for indices in indices_list:
                examples = []
                for i in indices:
                    examples.append(self.train_dataset[i])
                datasets.append(TextDataset(examples, self.args['fields']))

            client_train_loaders = [BucketIterator(train_dataset, batch_size=self.train_batch_size, device=self.device, sort_key=lambda x: len(x.text), train=True) for train_dataset in datasets]
        else:
            if noise is not None:
                n_clients = len(indices_list)
                
                client_train_loaders = []
                _, _, _, train, test, transform_train_list = self.prepare_dataset(self.name)
                for i in range(n_clients):
                    noise_level = noise / n_clients * (i + 1)
                    print('Gaussian noise level:', noise_level)
                    transform_train_list_copy = copy.deepcopy(transform_train_list)
                    transform_train_list_copy.append(AddGaussianNoise(0., noise_level))
                    transform_train = transforms.Compose(transform_train_list_copy)
                    train_set = Custom_Dataset(train.data[indices_list[i]], train.targets[indices_list[i]], device=self.device, transform=transform_train)
                    client_train_loaders.append(DataLoader(train_set, batch_size=batch_size))
            else:
                client_train_loaders = [DataLoader(self.train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(indices)) for indices in indices_list]

        self.train_loaders = client_train_loaders

        return client_train_loaders
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(tensor.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

from torchvision.datasets import MNIST

class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)		
        
        self.data = self.data.unsqueeze(1).float().div(255)
        # Uncomment if for 32*32 MNIST
        # from torch.nn import ZeroPad2d
        # pad = ZeroPad2d(2)
        # self.data = torch.stack([pad(sample.data) for sample in self.data])

        self.targets = self.targets.long()

        # Normalize
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, self.targets
        print('MNIST data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


from torchvision.datasets import FashionMNIST
class FastFMNIST(FashionMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)		
        
        self.data = self.data.unsqueeze(1).float().div(255)
        
        self.targets = self.targets.long()

        # Normalize
        self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, self.targets
        print('FMNIST data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target
    
from torchvision.datasets import SVHN
class FastSVHN(SVHN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = torch.from_numpy(self.data)
        self.data = self.data.float().div(255)
        # self.data = self.data.permute(0, 3, 1, 2)
        self.targets = torch.Tensor(self.labels).long()

        # Normalize
        for i in range(self.data.shape[1]):
            mean = self.data[:,i].mean()
            std = self.data[:,i].std()
            self.data[:,i].sub_(mean).div_(std)

        print('SVHN data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target

from torchvision.datasets import CIFAR10
class FastCIFAR10(CIFAR10):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Scale data to [0,1]
        from torch import from_numpy
        self.data = from_numpy(self.data)
        self.data = self.data.float().div(255)
        self.data = self.data.permute(0, 3, 1, 2)

        self.targets = torch.Tensor(self.targets).long()

        # Normalize
        for i, (mean, std) in enumerate(zip((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))):
            self.data[:,i].sub_(mean).div_(std)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, self.targets
        print('CIFAR10 data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target
    
from torchvision.datasets import CIFAR100
class FastCIFAR100(CIFAR100):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Scale data to [0,1]
        from torch import from_numpy
        self.data = from_numpy(self.data)
        self.data = self.data.float().div(255)
        self.data = self.data.permute(0, 3, 1, 2)

        self.targets = torch.Tensor(self.targets).long()

        # Normalize
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        for i, (mean, std) in enumerate(zip(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)):
            self.data[:,i].sub_(mean).div_(std)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, self.targets
        print('CIFAR100 data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class Custom_Dataset(Dataset):

    def __init__(self, X, y, device=None, transform=None):
        self.data = X.to(device)
        self.targets = y.to(device)
        self.count = len(X)
        self.device = device
        self.transform = transform

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx]), self.targets[idx]

        return self.data[idx], self.targets[idx]


def get_train_valid_indices(n_samples, train_val_split_ratio):
    indices = list(range(n_samples))
    # random.Random(1111).shuffle(indices)
    split_point = int(n_samples * train_val_split_ratio)
    train_indices, valid_indices = indices[:split_point], indices[split_point:]
    return  train_indices, valid_indices 

def random_split(sample_indices, m_bins, equal=True):
    np.random.seed(1111)
    sample_indices = np.asarray(sample_indices)
    if equal:
        np.random.shuffle(sample_indices)
        indices_list = np.array_split(sample_indices, m_bins)
    else:
        split_points = np.random.choice(
            len(sample_indices) - 2, m_bins - 1, replace=False) + 1
        split_points.sort()
        indices_list = np.split(sample_indices, split_points)

    return indices_list

def get_pad_to_min_len_fn(min_length):
    def pad_to_min_len(batch, vocab, min_length=min_length):
        pad_idx = vocab.stoi['<pad>']
        for idx, ex in enumerate(batch):
            if len(ex) < min_length:
                batch[idx] = ex + [pad_idx] * (min_length - len(ex))
        return batch
    return pad_to_min_len