from itertools import islice
import os

import numpy as np
import pandas as pd
import pretrainedmodels as ptm
from sacred import Experiment
from sacred.observers import FileStorageObserver, TelegramObserver
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm

from auglib.augmentation import Augmentations, set_seeds
from auglib.meters import AverageMeter
from auglib.dataset_loader import CSVDatasetWithName

np.set_printoptions(precision=4, suppress=True)

ex = Experiment()
fs_observer = FileStorageObserver.create('results')
ex.observers.append(fs_observer)
telegram_file = 'telegram.json'
if os.path.isfile(telegram_file):
    telegram_obs = TelegramObserver.from_config(telegram_file)
    ex.observers.append(telegram_obs)


@ex.config
def cfg():
    train_root = None
    train_csv = None
    train_split = None
    val_root = None
    val_csv = None
    val_split = None
    n_classes = 7
    epochs = 200  # maximum number of epochs
    batch_size = 32  # batch size
    num_workers = 8  # parallel jobs for data loading and augmentation
    model_name = None  # model: inceptionv4, densenet161, resnet152, senet154
    val_samples = 8  # number of samples per image in validation
    early_stopping_patience = 22  # patience for early stopping
    weighted_loss = False  # use weighted loss based on class imbalance
    balanced_loader = False  # balance classes in data loader
    # augmentations
    aug = {
        'hflip': False,  # Random Horizontal Flip
        'vflip': False,  # Random Vertical Flip
        'rotation': 0,  # Rotation (in degrees)
        'shear': 0,  # Shear (in degrees)
        'scale': 1.0,  # Scale (tuple (min, max))
        'color_contrast': 0,  # Color Jitter: Contrast
        'color_saturation': 0,  # Color Jitter: Saturation
        'color_brightness': 0,  # Color Jitter: Brightness
        'color_hue': 0,  # Color Jitter: Hue
        'random_crop': False,  # Random Crops
        'random_erasing': False,  # Random Erasing
        'piecewise_affine': False,  # Piecewise Affine
        'tps': False,  # TPS Affine
    }


def train_epoch(device, model, dataloaders, criterion, optimizer, phase,
                batches_per_epoch=None):
    losses = AverageMeter()
    accuracies = AverageMeter()
    all_preds = []
    all_labels = []
    model.train()

    if batches_per_epoch:
        tqdm_loader = tqdm(
            islice(dataloaders['train'], 0, batches_per_epoch),
            total=batches_per_epoch)
    else:
        tqdm_loader = tqdm(dataloaders[phase])
    for data in tqdm_loader:
        (inputs, labels), name = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        if phase == 'train':
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        acc = torch.sum(preds == labels.data).item() / preds.shape[0]
        accuracies.update(acc)
        all_preds += list(F.softmax(outputs, dim=1).cpu().data.numpy())
        all_labels += list(labels.cpu().data.numpy())
        tqdm_loader.set_postfix(loss=losses.avg, acc=accuracies.avg)

    # Calculate multiclass AUC
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Confusion Matrix
    print('Confusion matrix')
    cm = confusion_matrix(all_labels, all_preds.argmax(axis=1))
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    print(cmn)
    acc = np.trace(cmn) / cmn.shape[0]

    return {'loss': losses.avg, 'acc': acc}


def save_images(dataset, to, n=32):
    for i in range(n):
        img_path = os.path.join(to, 'img_{}.png'.format(i))
        save_image(dataset[i][0], img_path)


@ex.automain
def main(train_root, train_csv, train_split, val_root, val_csv, val_split,
         epochs, aug, model_name, batch_size, num_workers, val_samples,
         early_stopping_patience,
         n_classes, weighted_loss, balanced_loader, _run):
    assert(model_name in
           ('inceptionv4', 'resnet152', 'densenet161', 'senet154'))

    AUGMENTED_IMAGES_DIR = os.path.join(fs_observer.dir, 'images')
    CHECKPOINTS_DIR = os.path.join(fs_observer.dir, 'checkpoints')
    BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'model_best.pth')
    LAST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'model_last.pth')
    for directory in (AUGMENTED_IMAGES_DIR, CHECKPOINTS_DIR):
        os.makedirs(directory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'inceptionv4':
        model = ptm.inceptionv4(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, n_classes)
        aug['size'] = 299
        aug['mean'] = model.mean
        aug['std'] = model.std
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        aug['size'] = 224
        aug['mean'] = [0.485, 0.456, 0.406]
        aug['std'] = [0.229, 0.224, 0.225]
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
        aug['size'] = 224
        aug['mean'] = [0.485, 0.456, 0.406]
        aug['std'] = [0.229, 0.224, 0.225]
    elif model_name == 'senet154':
        model = ptm.senet154(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, n_classes)
        aug['size'] = model.input_size[1]
        aug['mean'] = model.mean
        aug['std'] = model.std
    model.to(device)

    augs = Augmentations(**aug)
    model.aug_params = aug

    train_ds = CSVDatasetWithName(
        train_root, train_csv, 'image', 'label',
        transform=augs.tf_transform, add_extension='.jpg', split=train_split)
    val_ds = CSVDatasetWithName(
        val_root, val_csv, 'image', 'label',
        transform=augs.tf_transform, add_extension='.jpg', split=val_split)

    datasets = {
        'train': train_ds,
        'val': val_ds
    }

    if balanced_loader:
        data_sampler = sampler.WeightedRandomSampler(
            train_ds.sampler_weights, len(train_ds))
        shuffle = False
    else:
        data_sampler = None
        shuffle = True

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            sampler=data_sampler, worker_init_fn=set_seeds),
        'val': DataLoader(datasets['val'], batch_size=batch_size,
                          shuffle=False, num_workers=num_workers,
                          worker_init_fn=set_seeds),
    }

    if weighted_loss:
        criterion = nn.CrossEntropyLoss(
            weight=torch.Tensor(datasets['train'].class_weights_list).cuda())
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001,
                          momentum=0.9, weight_decay=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                     min_lr=1e-5, patience=10)

    metrics = {
        'train': pd.DataFrame(columns=['epoch', 'loss', 'acc']),
        'val': pd.DataFrame(columns=['epoch', 'loss', 'acc'])
    }

    best_val_loss = 1000.0
    epochs_without_improvement = 0
    batches_per_epoch = None

    for epoch in range(epochs):
        print('train epoch {}/{}'.format(epoch+1, epochs))
        epoch_train_result = train_epoch(
            device, model, dataloaders, criterion, optimizer, 'train',
            batches_per_epoch)

        metrics['train'] = metrics['train'].append(
            {**epoch_train_result, 'epoch': epoch}, ignore_index=True)
        print('train', epoch_train_result)

        epoch_val_result = train_epoch(
            device, model, dataloaders, criterion, optimizer, 'val',
            batches_per_epoch)

        metrics['val'] = metrics['val'].append(
            {**epoch_val_result, 'epoch': epoch}, ignore_index=True)
        print('val', epoch_val_result)

        scheduler.step(epoch_val_result['loss'])

        if epoch_val_result['loss'] < best_val_loss:
            best_val_loss = epoch_val_result['loss']
            epochs_without_improvement = 0
            torch.save(model, BEST_MODEL_PATH)
            print('Best loss at epoch {}'.format(epoch))
        else:
            epochs_without_improvement += 1

        print('-' * 40)

        if epochs_without_improvement > early_stopping_patience:
            torch.save(model, LAST_MODEL_PATH)
            break

        if epoch == (epochs-1):
            torch.save(model, LAST_MODEL_PATH)

    for phase in ['train', 'val']:
        metrics[phase].epoch = metrics[phase].epoch.astype(int)
        metrics[phase].to_csv(os.path.join(fs_observer.dir, phase + '.csv'),
                              index=False)

    print('Best validation loss: {}'.format(best_val_loss))

    # TODO: return more metrics
    return {'max_val_acc': metrics['val']['acc'].max()}
