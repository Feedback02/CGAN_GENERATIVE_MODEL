import torch
import torch.nn as nn
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from kornia.color import RgbToLab, LabToRgb

class NormalizeLab:
    def __call__(self, img):
        l, a, b = torch.chunk(img, chunks=3, dim=0)
        l = (l - 50) / 50
        a = (a + 128) / 255 * 2 - 1
        b = (b + 128) / 255 * 2 - 1
        return torch.cat((l, a, b), dim=0)

    def denormalize(self, img):
        l, a, b = torch.chunk(img, chunks=3, dim=0)
        l = l * 50 + 50
        a = (a + 1) / 2 * 255 - 128
        b = (b + 1) / 2 * 255 - 128
        return torch.cat((l, a, b), dim=0)
    
    def denormalize_batch(self, lab_images):
        denorm_images = [] 
        for lab_image in lab_images: 
            denorm_images.append(self.denormalize(lab_image)) 
        return torch.stack(denorm_images)



class CustomDatasetImage(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.rgb2lab = RgbToLab()
        self.lab2rgb = LabToRgb()
        self.normalize_lab = NormalizeLab()

        for i, dir_name in enumerate(os.listdir(root_dir)):
            if i == 1000:
                break
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                json_file = os.path.join(dir_path, 'prompt.json')
                if os.path.exists(json_file):
                    with open(json_file, 'r') as file:
                        data = json.load(file)
                        input_text = data['input']
                        edit_text = data['edit']
                    for filename in os.listdir(dir_path):
                        if filename.endswith('_0.jpg'):
                            image_before_name_cut = filename[:-6]
                            image_before_name = f"{image_before_name_cut}_0.jpg"
                            image_after_name = f"{image_before_name_cut}_1.jpg"
                            self.samples.append((image_before_name, image_after_name, input_text, edit_text))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.tensor_to_image = transforms.ToPILImage()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_before_name, image_after_name, input_text, edit_text = self.samples[idx]

        image_before_path = os.path.join(self.root_dir, image_before_name)
        image_before = Image.open(image_before_path).convert('RGB')
        image_before = self.transform(image_before)
        image_before = self.rgb2lab(image_before)
        image_before = self.normalize_lab(image_before)

        return image_before

    def print_image_from_lab(self, image):
        image = self.normalize_lab.denormalize(image)
        image = to_pil_image(self.lab2rgb(image.unsqueeze(0)).squeeze(0))
        return image

    def print_l_channel(self, image):
        l_channel = image[0, :, :]
        plt.imshow(l_channel, cmap='gray')
        plt.title("L Channel - Grayscale")
        plt.axis('off')
        plt.show()

class CustomDatasetManga(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.rgb2lab = RgbToLab()
        self.lab2rgb = LabToRgb()
        self.normalize_lab = NormalizeLab()

        image_files = os.listdir(root_dir)
        print('Loading Manga dataset...')
        for i, image_name in enumerate(image_files):
            if i == 5000:
                break
            if self.check_image_integrity(image_name):
                self.samples.append(image_name)

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        self.tensor_to_image = transforms.ToPILImage()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_name = self.samples[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert('RGB')

        image_tensor = self.transform(image)
        image_lab = self.rgb2lab(image_tensor)
        image_lab = self.normalize_lab(image_lab)

        return image_lab

    def print_image_from_lab(self, image_lab):
        image_lab = self.normalize_lab.denormalize(image_lab)
        image_rgb = self.lab2rgb(image_lab.unsqueeze(0))
        image_pil = to_pil_image(image_rgb.squeeze(0))
        image_pil.show()

    def print_l_channel(self, image_lab):
        l_channel = image_lab[0, :, :]
        plt.imshow(l_channel.numpy(), cmap='gray')
        plt.title("L Channel - Grayscale")
        plt.axis('off')
        plt.show()

    def check_image_integrity(self, image_name):
        image_path = os.path.join(self.root_dir, image_name)
        try:
            with Image.open(image_path) as img:
                img.verify()
                return True
        except (IOError, SyntaxError):
            return False

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=1, manga=True):
        super().__init__()
        self.manga = manga
        self.root_dir = root_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        if self.manga:
            self.dataset = CustomDatasetManga(self.root_dir)
        else:
            self.dataset = CustomDatasetImage(self.root_dir)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
