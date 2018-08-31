import argparse

import numpy as np
import torch

from auglib.augmentation import Augmentations
from auglib.dataset_loader import CSVDatasetWithName
from auglib.test import test_with_augmentation

np.set_printoptions(precision=4, suppress=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to the model')
    parser.add_argument('dataset_root', help='Path to dataset root')
    parser.add_argument('dataset_csv', help='Path to dataset csv')
    parser.add_argument('--dataset-split', help='Path to dataset split')
    parser.add_argument('-n', type=int, default=1,
                        help='Number of image copies')
    parser.add_argument('--print-predictions', '-p', action='store_true',
                        help='Print the predicted value for each image')
    parser.add_argument('--output', '-o',
                        help='Path to output CSV file')
    return parser.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model)
    model.eval()
    model.to(device)

    print(model.aug_params)
    augs = Augmentations(**model.aug_params)
    dataset = CSVDatasetWithName(args.dataset_root,
                                 args.dataset_csv,
                                 'image',
                                 'label',
                                 transform=augs.tf_transform,
                                 add_extension='.jpg',
                                 split=args.dataset_split)

    score, preds = test_with_augmentation(model, dataset, device, 8, args.n)
    print(score)

    if args.print_predictions:
        for _, row in preds.iterrows():
            print("{},{}".format(row['image'], row['score']))

    if args.output:
        preds.to_csv(args.output, index=False,
                     columns=['image', 'MEL', 'NV', 'BCC', 'AKIEC',
                              'BKL', 'DF', 'VASC'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
