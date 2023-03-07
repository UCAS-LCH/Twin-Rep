import numpy as np
import pickle

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset


class CustomDatasetFromNumpy(Dataset):
    def __init__(self, img, label, transform):
        self.img = img
        self.label = label
        self.transform = transform
        self.len = len(self.img)

    def __getitem__(self, index):
        img_tensor = transforms.ToPILImage()(self.img[index])
        img_tensor = self.transform(img_tensor)
        label_tensor = self.label[index]
        return (img_tensor, label_tensor)

    def __len__(self):
        return self.len


def get_semisup_dataloader(batch_size):
    with open("../data/ti_500K_pseudo_labeled.pickle", "rb",) as f:
        data = pickle.load(f)
    img, label = data["data"], data["extrapolated_targets"]
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    # select random subset
    index = np.random.permutation(np.arange(len(label)))[0 : int(1.0 * len(label))]

    sm_loader = torch.utils.data.DataLoader(
        CustomDatasetFromNumpy(img[index], label[index], train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    return sm_loader