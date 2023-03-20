from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import typing
import numpy as np
import pathlib
np.random.seed(0)

mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)


def get_data_dir():
    server_dir = pathlib.Path("/work/datasets/cifar10")
    if server_dir.is_dir():
        return str(server_dir)
    return "data/cifar10"


def load_cifar10(resize:int, new_mean_std: bool, transform_name: str, batch_size: int, validation_fraction: float = 0.1) -> typing.List[torch.utils.data.DataLoader]:
    
    # Note that transform train will apply the same transform for
    # validation!
    if new_mean_std:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (.25, .25, .25)
        
        
    transform_dict = {
        'default': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]),

        'flip': transforms.Compose([    #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomHorizontalFlip()
            ]),

        'flip_crop': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    }
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),])
    

    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),])
     
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    



    #transform_train =  transform_dict.get(transform_name)
    #transform_test = transform_dict['default']
    


        
    # Get the transform object based on the name passed in
    #transform = transform_dict.get(transform_name, transform_dict['default'])
    
    if resize: #Resizes to 224x224
        transform_train.transforms.insert(0,transforms.Resize(resize))
        transform_test.transforms.insert(0,transforms.Resize(resize))

    
    data_train = datasets.CIFAR10(get_data_dir(),
                                  train=True,
                                  download=True,
                                  transform=transform_train,
                                 )

    augment_train = datasets.CIFAR10(get_data_dir(),
                                  train=True,
                                  download=True,
                                  transform=augment,
                                 )

    data_test = datasets.CIFAR10(get_data_dir(),
                                 train=False,
                                 download=True,
                                 transform=transform_test,
                                )

    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    
    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)
    
    
    augmented_train = torch.utils.data.DataLoader(augment_train, #data_train, 
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)
    
    dataloader_train = torch.utils.data.DataLoader(
         	        dataset=torch.utils.data.ConcatDataset([dataloader_train.dataset, augmented_train.dataset]),
                    batch_size=dataloader_train.batch_size,
                    num_workers=dataloader_train.num_workers)
    

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_val, dataloader_test
