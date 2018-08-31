import os
import os.path

import pandas as pd
import torch.utils.data as data
from torchvision.datasets.folder import default_loader


# TODO: Make target_field optional for unannotated datasets.
class CSVDataset(data.Dataset):
    def __init__(self, root, csv_file, image_field, target_field,
                 loader=default_loader, transform=None,
                 target_transform=None, add_extension=None,
                 limit=None, random_subset_size=None,
                 split=None):
        self.root = root
        self.loader = loader
        self.image_field = image_field
        self.target_field = target_field
        self.transform = transform
        self.target_transform = target_transform
        self.add_extension = add_extension

        self.data = pd.read_csv(csv_file, sep=None)

        # Split
        if split is not None:
            with open(split, 'r') as f:
                selected_images = f.read().splitlines()
            self.data = self.data[self.data[image_field].isin(selected_images)]
            self.data = self.data.reset_index()

        # Calculate class weights for WeightedRandomSampler
        self.class_counts = dict(self.data['label'].value_counts())
        self.class_weights = {label: max(self.class_counts.values()) / count
                              for label, count in self.class_counts.items()}
        self.sampler_weights = [self.class_weights[cls]
                                for cls in self.data['label']]
        self.class_weights_list = [self.class_weights[k]
                                   for k in sorted(self.class_weights)]

        if random_subset_size:
            self.data = self.data.sample(n=random_subset_size)
            self.data = self.data.reset_index()

        if type(limit) == int:
            limit = (0, limit)
        if type(limit) == tuple:
            self.data = self.data[limit[0]:limit[1]]
            self.data = self.data.reset_index()

        classes = list(self.data[self.target_field].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes

        print('Found {} images from {} classes.'.format(len(self.data),
                                                        len(classes)))
        for class_name, idx in self.class_to_idx.items():
            n_images = dict(self.data[self.target_field].value_counts())
            print("    Class '{}' ({}): {} images.".format(
                class_name, idx, n_images[class_name]))

    def __getitem__(self, index):
        path = os.path.join(self.root,
                            self.data.loc[index, self.image_field])
        if self.add_extension:
            path = path + self.add_extension
        sample = self.loader(path)
        target = self.class_to_idx[self.data.loc[index, self.target_field]]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.data)


class CSVDatasetWithName(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        return super().__getitem__(i), name
