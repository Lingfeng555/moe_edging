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

class NEUDataset(Dataset):
    base_path = "metal_dataset/"

    def __init__(self, set:str, transform=None, seed:int = None):
        '''set can be train, test, or valid'''
        super().__init__()
        self.base_path = self.base_path+set+'/'
        self.categories = os.listdir(self.base_path)
        self.transform = transform
        paths = self._get_labels()
        self.data = {
            "Path": paths
        }

        for categ in self.categories: self.data[categ] = [0 for i in range(len(paths))]

        for i in range(len(paths)): self.data[paths[i].split("/")[2]][i] = 1

        self.data = pd.DataFrame(self.data)
        
        if seed != None :
            self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)



    def _get_labels(self):
        return [
            os.path.join(self.base_path, categ, image)
            for categ in self.categories
            for image in os.listdir(os.path.join(self.base_path, categ))
        ]
    
    def __len__(self):
        paths = dataset._get_labels()
        return len(paths)
    
    def __getitem__(self, index):
        img_path = self.data.iloc[index]["Path"]
        image = Image.open(img_path).convert("L")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        label = dataset.data.drop(columns="Path").iloc[index].values.astype(int)
        label = torch.tensor(label, dtype=torch.int8)

        return image, label
        
        

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Crear el dataset
    dataset = NEUDataset(set="train", transform=transform, seed=1)

    # Probar con un DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Intentar iterar sobre el dataloader
    for images, labels in dataloader:
        print("Iteración exitosa.")
        print("Imagen shape:", images.shape)
        print("Etiqueta shape:", labels.shape)
        break