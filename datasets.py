import glob
import os
import numpy as np

from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, mode="train", transforms_label=None, transforms_img=None):
        super().__init__()
        self.transforms_label = transform.Compose(transforms_label)
        self.transforms_img = transform.Compose(transforms_img)
        self.mode = mode
        if self.mode == 'train':
            self.files = sorted(
                glob.glob(os.path.join(root, mode, "imgs") + "/*.*"))
        self.labels = sorted(
            glob.glob(os.path.join(root, mode, "labels") + "/*.*"))
        self.set_attrs(total_len=len(self.labels))
        print(f"from {mode} split load {self.total_len} images.")

    def __getitem__(self, index):
        label_path = self.labels[index % len(self.labels)]
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        img_B = self.transforms_label(img_B)

        if self.mode == "train":
            img_A = Image.open(self.files[index % len(self.files)])
            img_A = img_A.convert('RGB')
            img_A = self.transforms_img(img_A)
        else:
            img_A = np.empty([1])

        input_dict = {'label': img_B,
                      'instance': 0,
                      'image': img_A,
                      'path': label_path,
                      }

        return input_dict
