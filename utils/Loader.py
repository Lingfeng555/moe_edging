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

    def __init__(self, set:str, transform=None, seed:int = None, scale: float = 1 ):
        '''set can be train, test, or valid'''
        super().__init__()
        self.base_path = self.base_path+set+'/'
        self.categories = os.listdir(self.base_path)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.scale = scale
        self.data = {
            "Path": self._get_labels()
        }

        for categ in self.categories: self.data[categ] = [0 for i in range(len(self.data["Path"]))]

        for i in range(len(self.data["Path"])): self.data[self.data["Path"][i].split("/")[2]][i] = 1

        self.data = pd.DataFrame(self.data)
        
        if seed != None :
            self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True)

        self.data = pd.merge(self.data, pd.read_csv("output.csv"), on='Path', how='inner')

    def _get_labels(self):
        return [
            os.path.join(self.base_path, categ, image)
            for categ in self.categories
            for image in os.listdir(os.path.join(self.base_path, categ))
        ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = self.data.iloc[index]["Path"]
        image = Image.open(img_path)

        original_width, original_height = image.size
        image = image.resize((max(1, int(original_width * self.scale)), max(1, int(original_height * self.scale))))

        if self.transform:
            image = self.transform(image)

        label = self.data.drop(columns="Path").iloc[index].values.astype(int)
        label = torch.tensor(label, dtype=torch.int8)

        # Extraer sp y sr como float y convertir a tensor
        sp = float(self.data.iloc[index]["sp"])
        sr = float(self.data.iloc[index]["sr"])
        best_parameters = torch.tensor([sp, sr], dtype=torch.float)
        
        return image, label, best_parameters
    
    def check_image_sizes(self):
        sizes = set()
        for path in self.data["Path"]:
            with Image.open(path) as img:
                sizes.add(img.size)
                if len(sizes) > 1:
                    print(f"Diferentes tamaños encontrados: {sizes}")
                    return False
        print(f"Todas las imágenes tienen el mismo tamaño: {sizes}")
    
    def calculate_batch_size_in_gb(self, batch_size: int):
        # Obtener el tamaño de una imagen procesada
        img_path = self.data.iloc[0]["Path"]
        with Image.open(img_path).convert("L") as img:
            original_width, original_height = img.size
            new_width = max(1, int(original_width * self.scale))
            new_height = max(1, int(original_height * self.scale))

        # Tamaño de imagen: (Canales x Alto x Ancho) * tamaño de cada pixel (float32 = 4 bytes)
        image_size_bytes = (1 * new_height * new_width) * 4  # Escala de grises, 1 canal, float32

        # Tamaño de la etiqueta: cantidad de categorías * 1 byte (int8)
        num_labels = len(self.categories)
        label_size_bytes = num_labels * 1  # int8 = 1 byte por categoría

        # Calcular tamaño total por batch
        total_bytes_per_batch = (image_size_bytes + label_size_bytes) * batch_size

        # Convertir bytes a gigabytes (1 GB = 1024^3 bytes)
        total_gb_per_batch = total_bytes_per_batch / (1024 ** 3)

        print(f"Tamaño de cada batch ({batch_size} muestras): {total_gb_per_batch:.6f} GB")
        return total_gb_per_batch
    
    def get_sp_sr (self, path: str):
        return dataset.data["sp"][dataset.data["Path"] == path], dataset.data["sr"][dataset.data["Path"] == path]
    
if __name__ == '__main__':

    # Crear el dataset
    dataset = NEUDataset(set="train", transform=None, seed=1, scale=0.5)

    # Probar con un DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images, labels, best_parameters in dataloader:
        print(images.shape, labels.shape, best_parameters)
        break
