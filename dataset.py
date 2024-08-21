import os
import numpy
import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset


class magic_wand_dataset(Dataset):
    def __init__(self, root_dir, transform=torch.tensor):
        super(magic_wand_dataset, self).__init__()

        self.transform = transform
        self.images, self.labels = [], []
        self.classes = [file.split(".")[0] for file in os.listdir(root_dir)]

        for file in os.listdir(root_dir):
            name, ext = os.path.splitext(file)
            images = numpy.load(os.path.join(root_dir, file))

            self.images.extend(images)
            class_index = self.classes.index(name)
            self.labels.extend([class_index for _ in range(len(images))])

        self.labels = torch.tensor(self.labels)
        self.images = self.transform(self.images)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class train_set(Dataset):
    def __init__(self, images, labels):
        super(train_set, self).__init__()
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx].reshape(28, 28), self.labels[idx]

    def to_one_hot(self, labels, num_classes):
        return one_hot(labels.long(), num_classes).type(torch.float32)


class val_set(Dataset):
    def __init__(self, images, labels):
        super(val_set, self).__init__()
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx].reshape(28, 28), self.labels[idx]

    def to_one_hot(self, labels, num_classes):
        return one_hot(labels.long(), num_classes).type(torch.float32)

    def one_hot_to_label(self, one_hot):
        return torch.argmax(one_hot, dim=1)


if __name__ == "__main__":
    dataset = magic_wand_dataset(root_dir="data/.npy")
