import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pandas as pd
from PIL import Image

class CXR8Dataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path (str): Path to the CSV file.
            image_dir (str): Directory containing the images.
            transform (callable, optional): Transform to apply to images.
        """
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        # Filter rows where the 'id' is not in the available images
        available_images = set(os.listdir(self.image_dir))
        self.data = self.data[self.data['id'].isin(available_images)]

        # Extract labels (columns 2–21)
        self.labels = self.data.drop(columns=['id', 'subject_id']).values.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image ID and construct its full path
        img_id = self.data.iloc[idx]['id']
        img_path = os.path.join(self.image_dir, img_id)

        # Load the image and convert to grayscale
        image = Image.open(img_path).convert("L")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        # Get the corresponding labels
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.float32)  # Convert labels to tensor
        
        return image, label
    