import torch
from torchvision import datasets, transforms

from .eurosat import SimpleDataManager
from torch.utils.data import DataLoader, random_split

def load_data(root, dataset, batch_size):
    
    if dataset == 'pathmnist':
        
        import medmnist
        from medmnist import INFO
        data_flag = 'pathmnist'
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        
        data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[.5], std=[.5])
                ])

        train_dataset = DataClass(split='train', transform=data_transform, download=True)
        test_dataset = DataClass(split='test', transform=data_transform, download=True)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=2*batch_size, shuffle=False)
        return train_loader, test_loader
        
    if dataset == 'eurosat':
        image_size = 64
        data_manager = SimpleDataManager(root, image_size, batch_size)
        train_dataloader = data_manager.get_data_loader(aug=True, normalise=True)
        test_dataloader = data_manager.get_data_loader(aug=False, normalise=True)
        return train_dataloader, test_dataloader

    if dataset in ['mnist', 'cifar10', 'fashion_mnist', 'svhn', 'stl10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset == 'flowers102':
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to a common size for other datasets
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    if dataset == 'mnist':
        ds = datasets.MNIST
    elif dataset == 'cifar10':
        ds = datasets.CIFAR10
    elif dataset == 'fashion_mnist':
        ds = datasets.FashionMNIST
    elif dataset == 'svhn':
        ds = datasets.SVHN
    elif dataset == 'stl10':
        ds = datasets.STL10
    elif dataset == 'caltech101':
        ds = datasets.Caltech101
    elif dataset == 'oxford_iiit_pet':
        ds = datasets.OxfordIIITPet
    elif dataset == 'flowers102':
        ds = datasets.Flowers102
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # Adjust parameters based on the dataset
    if dataset in ['mnist', 'cifar10', 'fashion_mnist']:
        train_data = ds(root=root, train=True, download=True, transform=transform)
        test_data = ds(root=root, train=False, download=True, transform=transform)
    elif dataset in ['svhn']:
        train_data = ds(root=root, split='train', download=True, transform=transform)
        test_data = ds(root=root, split='test', download=True, transform=transform)
    elif dataset == 'caltech101':
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(dataset, [train_size, test_size])
    elif dataset == 'stl10':
        train_data = ds(root=root, split='train', download=True, transform=transform)
        test_data = ds(root=root, split='test', download=True, transform=transform)
    elif dataset in ['oxford_iiit_pet']:
        train_data = ds(root=root, split='trainval', download=True, transform=transform)
        test_data = ds(root=root, split='test', download=True, transform=transform)
    elif dataset in ['flowers102']:
        train_data = ds(root=root, split='train', download=True, transform=transform)
        test_data = ds(root=root, split='test', download=True, transform=transform)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
