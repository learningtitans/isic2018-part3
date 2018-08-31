import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from .augmentation import set_seeds
from .meters import AverageMeter


class AugmentOnTest:
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n * len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i // self.n]


def test_with_augmentation(model, dataset, device, num_workers, n,
                           save_images=False):
    assert n >= 1, "n must be larger than 1"

    model.eval()
    criterion = nn.CrossEntropyLoss()

    if n != 1:
        dataset = AugmentOnTest(dataset, n)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=n, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        worker_init_fn=set_seeds)

    losses = AverageMeter()
    predictions = pd.DataFrame(
        columns=['image', 'label', 'MEL', 'NV', 'BCC',
                 'AKIEC', 'BKL', 'DF', 'VASC'])
    all_scores = []

    for i, data in enumerate(tqdm(dataloader)):
        (inputs, labels), name = data

        # If the inputs tensor has 5 dimensions, it means
        # that TenCrop or DenseCrop was used.
        if inputs.dim() == 5:
            inputs = inputs.squeeze(0)
            labels = labels.repeat(inputs.shape[0])

        inputs = inputs.to(device)
        labels = labels.to(device)

        if save_images:
            if i <= 10:
                save_image(make_grid(inputs, padding=0),
                           'grid_{}.jpg'.format(i))

        with torch.no_grad():
            outputs = model(inputs)
            scores = F.softmax(outputs, dim=1).cpu().data.numpy()
            loss = criterion(outputs, labels)

        losses.update(loss.item(), inputs.size(0))

        scores = scores.mean(axis=0)
        all_scores.append(scores)
        predictions = predictions.append(
            {'image': name[0],
             'label': labels.data[0].item(),
             'MEL': scores[0],
             'NV': scores[1],
             'BCC': scores[2],
             'AKIEC': scores[3],
             'BKL': scores[4],
             'DF': scores[5],
             'VASC': scores[6]},
            ignore_index=True)

    labels_array = predictions['label'].values.astype(int)
    scores_array = np.array(all_scores)
    predicted_labels = scores_array.argmax(axis=1)

    acc = accuracy_score(labels_array, predicted_labels)

    # Confusion Matrix
    print('Confusion matrix')
    cm = confusion_matrix(labels_array, predicted_labels)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    print(cmn)
    acc = np.trace(cmn) / cmn.shape[0]

    return ({'loss': losses.avg, 'acc': acc}, predictions)
