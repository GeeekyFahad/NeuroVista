from torch.utils.data import Dataset
import os
import torch
from preprocess import preprocess

class OASISDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.paths = []
        self.labels = []
        classes = {
            "Non Demented":0,
            "Very mild Dementia":1,
            "Mild Dementia":2,
            "Moderate Dementia":3
        }

        for cls in classes:
            folder = os.path.join(root_dir, cls)
            for file in os.listdir(folder):
                if file.endswith(".nii") or file.endswith(".nii.gz"):
                    self.paths.append(os.path.join(folder, file))
                    self.labels.append(classes[cls])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        x = preprocess(path)
        x = x.unsqueeze(0)
        y = torch.tensor(self.labels[idx]).long()
        return x, y
