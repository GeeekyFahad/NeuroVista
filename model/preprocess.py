import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import random



def load_mri(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data

def normalize(img):
    img = (img - np.mean(img)) / (np.std(img) + 1e-5)
    return img

def resize(img, new_shape=(128, 128, 128)):
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    img = F.interpolate(img, size=new_shape, mode="trilinear", align_corners=False)
    return img.squeeze()

def preprocess(path, augment=False):
    img = load_mri(path)
    img = normalize(img)
    img = resize(img)
    img = img.float()

    if augment:
        img = random_flip(img)
        img = add_noise(img)

    return img


def random_flip(img):
    if random.random() > 0.5:
        img = torch.flip(img, dims=[0])
    if random.random() > 0.5:
        img = torch.flip(img, dims=[1])
    if random.random() > 0.5:
        img = torch.flip(img, dims=[2])
    return img

def add_noise(img, noise_level=0.05):
    noise = torch.randn_like(img) * noise_level
    return img + noise

