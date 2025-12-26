import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

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

def preprocess(path):
    img = load_mri(path)
    img = normalize(img)
    img = resize(img)
    return img.float()
