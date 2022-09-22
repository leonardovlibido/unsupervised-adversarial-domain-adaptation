from pathlib import Path
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def initialize_datasets(dataset_name, batch_size, input_size, train_subset_size = None, val_subset_size = None):
    dataset = {}
    dataloaders = {}
    dataset_sizes = {}

    if dataset_name == 'mnist':
        root = '/home/rdjordjevic/master/repos/domain-adaptation-codebase/datasets/mnist'
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.1307, 0.1307, 0.1307,), (0.3081, 0.3081, 0.3081,)),
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.1307, 0.1307, 0.1307,), (0.3081, 0.3081, 0.3081,)),
            ])
        }

        dataset['train'] = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=data_transforms['train'],
            # target_transform=transforms.Compose([transforms.ToTensor()]),
        )
        dataset['val'] = datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=data_transforms['val'],
            # target_transform=transforms.Compose([transforms.ToTensor()]),
        )
    elif dataset_name == 'svhn':
        root = '/home/rdjordjevic/master/repos/domain-adaptation-codebase/datasets/svhn'

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.43768206, 0.44376972, 0.47280434], [0.19803014, 0.20101564, 0.19703615])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.43768206, 0.44376972, 0.47280434], [0.19803014, 0.20101564, 0.19703615])
            ])
        }

        dataset['train'] = datasets.SVHN(
            root=root,
            split='train',
            download=True,
            transform=data_transforms['train'],
            # target_transform=transforms.Compose([transforms.ToTensor()]),
        )
        dataset['val'] = datasets.SVHN(
            root=root,
            split='test',
            download=True,
            transform=data_transforms['val'],
            # target_transform=transforms.Compose([transforms.ToTensor()]),
        )
    elif dataset_name == 'mnistm':
        root = Path('/home/rdjordjevic/master/repos/domain-adaptation-codebase/datasets/mnistm/image_folder')
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.4579, 0.4620, 0.4082], [0.2519, 0.2367, 0.2587])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.4579, 0.4620, 0.4082], [0.2519, 0.2367, 0.2587])
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        dataset['train'] = datasets.ImageFolder(str(root / 'train'), data_transforms['train'])
        dataset['val'] = datasets.ImageFolder(str(root / 'test'), data_transforms['val'])

    if train_subset_size is not None:
        dataset_sizes['train'] = len(dataset['train'])
        print(f'---{dataset_sizes["train"]}, {train_subset_size}')
        indices = random.sample(range(0, dataset_sizes['train']), train_subset_size)
        dataset['train'] = Subset(dataset['train'], indices)

    if val_subset_size is not None:
        dataset_sizes['val'] = len(dataset['val'])
        print(f'---{dataset_sizes["val"]}, {val_subset_size}')
        indices = random.sample(range(0, dataset_sizes['val']), val_subset_size)
        dataset['val'] = Subset(dataset['val'], indices)

    dataset_sizes['train'] = len(dataset['train'])
    dataset_sizes['val'] = len(dataset['val'])

    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 1,
        'pin_memory': True,
        'shuffle': True
    }

    dataloaders['train'] = DataLoader(dataset['train'], **dataloader_kwargs)
    dataloaders['val'] = DataLoader(dataset['val'], **dataloader_kwargs)

    return dataloaders, dataset_sizes


def get_mean_and_std(dataloader):
    import torch
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def display_dataset_info():
    dataloaders, dataset_sizes = initialize_datasets('svhn', 1024, 32)
    print(dataset_sizes)

    inputs, labels = next(iter(dataloaders['train']))
    print(f'X shape: {inputs.shape}')
    print(f'label shape: {labels.shape}')

    # mean, std = get_mean_and_std(dataloaders['train'])
    # print(f'dataset mean : {mean.tolist()}')
    # print(f'dataset std : {std.tolist()}')

    import torch
    counts = torch.zeros(10)
    for data, _ in dataloaders['train']:
        counts += labels.unique(return_counts=True)[1]

    print(counts.tolist())
    print((counts / counts.sum()).tolist())


if __name__ == "__main__":
    display_dataset_info()
