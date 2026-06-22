# =============================================================
# ml_pipeline/src/dataset.py  —  DataLoaders with augmentation
# =============================================================

from collections import Counter

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

import config


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])
    return {'train': train_tf, 'val': eval_tf, 'test': eval_tf}


def get_datasets(root=config.DATA_ROOT):
    tfs = get_transforms()
    return {
        split: datasets.ImageFolder(f'{root}/{split}', transform=tfs[split])
        for split in config.SPLITS
    }


def get_dataloaders(ds_dict):
    # WeightedRandomSampler for class imbalance
    train_labels  = [s[1] for s in ds_dict['train'].samples]
    class_counts  = Counter(train_labels)
    total         = len(train_labels)
    cw = {c: total / (len(class_counts) * n) for c, n in class_counts.items()}
    sw = [cw[l] for l in train_labels]
    sampler = WeightedRandomSampler(sw, len(sw), replacement=True)

    loaders = {
        'train': DataLoader(ds_dict['train'], batch_size=config.BATCH_SIZE,
                            sampler=sampler, num_workers=config.NUM_WORKERS,
                            pin_memory=True),
        'val':   DataLoader(ds_dict['val'],   batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=True),
        'test':  DataLoader(ds_dict['test'],  batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=config.NUM_WORKERS,
                            pin_memory=True),
    }
    print('✅  DataLoaders ready')
    for s in config.SPLITS:
        print(f'    {s:5s}: {len(ds_dict[s]):,} images')
    print(f'    Class weights → {cw}')
    return loaders
