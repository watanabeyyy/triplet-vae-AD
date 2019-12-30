import glob
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from PIL import Image
import random


def create_npy(target_dir, output_name):
    paths = glob.glob(target_dir + "/*")

    os.makedirs("dataset", exist_ok=True)

    img_list = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img_h = img.shape[0]
        img_w = img.shape[1]
        h = 0
        num = 0
        while True:
            h = np.random.randint(h, min(h + 100, img_h))
            if h + 64 >= img_h - 1 or num >= 500:
                break
            w = 0
            while True:
                w = np.random.randint(w, min(w + 100, img_w))
                if w + 64 >= img_w or num >= 500:
                    break
                mini_img = img[h:h + 64, w:w + 64]
                m = np.mean(mini_img)
                # plt.plot(m,".")
                # plt.pause(0.01)
                if m < 30:
                    # plt.clf()
                    # plt.imshow(mini_img)
                    # plt.pause(0.01)
                    None
                else:
                    img_list.append(np.copy(mini_img))
                    num += 1

        print(num)
    print(np.array(img_list).shape)
    np.save("dataset/" + output_name, np.array(img_list))


class MDDataset(Dataset):

    def __init__(self, data_npy, transform=None):
        self.data_npy = data_npy
        self.transform = transform
        self.make_triplet()

    def __len__(self):
        return self.data_npy.shape[0] // 3

    def __getitem__(self, idx):
        anchor = self.anchor[idx]
        positive = self.positive[idx]
        negative = self.negative_aug(self.negative[idx])

        anchor = Image.fromarray(anchor)
        positive = Image.fromarray(positive)
        negative = Image.fromarray(negative)

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def make_triplet(self):
        idx = [i for i in range(self.data_npy.shape[0])]
        random.shuffle(idx)
        size = len(idx) // 3
        self.anchor = self.data_npy[idx[:size]]
        self.positive = self.data_npy[idx[size:size * 2]]
        self.negative = self.data_npy[idx[-size:]]

    def negative_aug(self, data):
        data = np.copy(data)
        aug = np.random.randint(0, 2)
        if aug == 0:
            h = np.random.randint(4, data.shape[0] // 8)
            w = np.random.randint(data.shape[0] // 8, data.shape[1])
            x1 = np.random.randint(0, data.shape[0] - h)
            y1 = np.random.randint(0, data.shape[1] - w)
            data[x1:x1 + h, y1:y1 + w] = np.random.randint(0, 255)
        elif aug == 1:
            h = np.random.randint(data.shape[0] // 4, data.shape[0] // 4 * 3)
            w = np.random.randint(data.shape[1] // 4, data.shape[1] // 4 * 3)
            x1 = np.random.randint(0, data.shape[0] - h)
            y1 = np.random.randint(0, data.shape[1] - w)
            data[x1:x1 + h, y1:y1 + w] = np.random.randint(0, 255)
        return data


if __name__ == "__main__":
    from config import config

    target_dir = config.dataset_path + "dataset"
    create_npy(target_dir, "data")
