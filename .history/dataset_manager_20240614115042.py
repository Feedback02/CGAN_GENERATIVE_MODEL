import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from kornia.color import RgbToLab
from kornia.color import LabToRgb 

import pytorch_lightning as pl

class CustomDatasetImage(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.rgb2lab = RgbToLab()
        self.lab2rgb = LabToRgb()
        counter = 0

        for i,dir_name in enumerate(os.listdir(root_dir)):
            if i==1000:
                break
                
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                json_file = os.path.join(dir_path, 'prompt.json')
                if os.path.exists(json_file):
                    with open(json_file, 'r') as file:
                        data = json.load(file)
                        input_text = data['input']
                        edit_text = data['edit']
                    #print(os.listdir(dir_path))
                    for filename in os.listdir(dir_path):
                        #lets extract for each _0 image, its corresponding _1
                        filename = filename[:-4] # delete  '.jpg'
                        #print(filename)
                        if filename.endswith('_0'):
                            # original image\
                            image_before_name_cut = filename[:-2] # is the 'image before' without _0

                            image_before_name = image_before_name_cut + '_0'
                            image_after_name = image_before_name_cut + '_1'

                            image_before_name = os.path.join(dir_name, f"{image_before_name}.jpg")
                            image_after_name = os.path.join(dir_name, f"{image_after_name}.jpg")
                            #print(image_before_name)
                            #print(image_after_name)


                            self.samples.append((image_before_name, image_after_name, input_text, edit_text))
                else:
                    counter +=1

        #calculate the mean and std of the entire dataset ?
        #typicalli is done to classification, in pix2pix is used -1,1
       # mean = np.mean(self.samples)
        #print(len(self.samples))
        self.transform = transforms.Compose([
            #transforms.Resize((256, 256)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.tensor_to_image = transforms.ToPILImage()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_before_name, image_after_name, input_text, edit_text = self.samples[idx]

        image_before_path = os.path.join(self.root_dir, image_before_name)
        image_after_path = os.path.join(self.root_dir, image_after_name)

        image_before = Image.open(image_before_path).convert('RGB')
        image_before = self.transform(image_before)
        image_before = self.rgb2lab(image_before)

        #image_after = Image.open(image_after_path).convert('RGB')
        #image_after = self.transform(image_after)
        #image_after = self.rgb2lab(image_before)

        return image_before

    def print_image_from_lab(self, image):
        #mean = np.array([0.5, 0.5, 0.5])
       # std = np.array([0.5, 0.5, 0.5])
        #image = image * std[:, None, None] + mean[:, None, None]
       # print(image.shape)
        image = to_pil_image(self.lab2rgb(image).squeeze(0))
       
        #image.show()
        return image

    def print_l_channel(self, image):
        l_channel = image[0,:,:]
        plt.imshow(l_channel, cmap='gray')
        plt.title("L Channel - Grayscale")
        plt.axis('off')
        plt.show()
        #image.show()
        #return image


class CustomDatasetManga(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = []
        self.rgb2lab = RgbToLab()
        self.lab2rgb = LabToRgb()

        # List images and limit to first 1000
        image_files = os.listdir(root_dir)
        print('Loading Manga dataset...')
        #print(image_files)
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

        return image_lab

    def print_image_from_lab(self, image_lab):
        image_rgb = self.lab2rgb(image_lab)
        image_pil = to_pil_image(image_rgb)
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
                img.verify()  # Verify the image
                # If no exception is raised, the image is okay
                return True
        except (IOError, SyntaxError) as e:
            # Handle exceptions that are raised
            #print(f"Error opening or verifying image: {e}")
            return False



class CustomDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=1, manga=True):
        super().__init__()
        self.manga = manga
        self.root_dir = root_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        if self.manga==True:
            self.dataset = CustomDatasetManga(self.root_dir)
        else:
            self.dataset = CustomDatasetImage(self.root_dir)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
