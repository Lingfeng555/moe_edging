import sys
sys.path.insert(1, '../') 
from utils.Loader import CXR8Dataset
from utils.evaluator import Evaluator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import os
from pathlib import Path
import pandas as pd
from PIL import Image

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class HardGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2):
        """
        Args:
            input_dim (int): Dimension of input features.
            num_experts (int): Total number of experts.
            top_k (int): Maximum number of experts to activate (hard gating).
        """
        super(HardGatingNetwork, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        Forward pass of the gating network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            selected_experts (torch.Tensor): Binary mask of activated experts (batch_size, num_experts).
            expert_weights (torch.Tensor): Normalized weights for the selected experts (batch_size, num_experts).
        """
        logits = self.gate(x)
        top_k_values, top_k_indices = torch.topk(logits, self.top_k, dim=1)

        mask = torch.zeros_like(logits)
        mask.scatter_(1, top_k_indices, 1.0)

        sparse_logits = mask * logits 
        expert_weights = F.softmax(sparse_logits, dim=1) 

        return mask, expert_weights
    
class ExpertCNN(nn.Module):
    def __init__(self):
        super(ExpertCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        
        # Placeholder for dynamically calculated in_features
        self.fc = None  # Dynamically initialized in forward pass
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        if self.fc is None:
            num_features = x.view(x.size(0), -1).size(1)
            self.fc = nn.Linear(num_features, 20).to(x.device)  # Output size is 20
        
        x = x.view(x.size(0), -1)
        return self.fc(x)

class MoEModel(nn.Module):
    def __init__(self, num_experts):
        super(MoEModel, self).__init__()
        self.num_experts = num_experts
        
        # Experts
        self.experts = nn.ModuleList([ExpertCNN() for _ in range(num_experts)])
        
        # Feature extractor for gating
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Changed in_channels to 1
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 128),  # Adjust size based on your input image dimensions
            nn.ReLU()
        )
        
        # Gating network
        self.gating = HardGatingNetwork(input_dim=128, num_experts=num_experts)

    def forward(self, x):
        # Extract features from the image for gating
        features = self.feature_extractor(x)
        
        # Get gating scores
        mask, gating_scores = self.gating(features)
        
        # Top-2 sparsity
        topk_values, topk_indices = torch.topk(gating_scores, k=self.num_experts, dim=-1)
        
        # Compute outputs for all experts
        outputs = torch.stack([self.experts[i](x) for i in range(self.num_experts)], dim=1)
        
        # Select the outputs of the top-k experts
        selected_outputs = outputs.gather(
            1, topk_indices.unsqueeze(-1).expand(-1, -1, outputs.size(-1))
        )
        
        # Combine the outputs of the selected experts
        combined_output = (selected_outputs * topk_values.unsqueeze(-1)).sum(dim=1)
        
        return combined_output, gating_scores
    
class AverageROCAUC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            """
            Calcula el ROC AUC por columna y devuelve el promedio.
            Si una columna solo tiene una clase, devuelve un tensor de ceros 
            del mismo tamaño que el output previsto para esa columna.
            """
            device = y_pred.device

            # Pasamos a numpy para calcular ROC AUC
            y_pred_np = y_pred.detach().cpu().numpy()
            y_true_np = y_true.detach().cpu().numpy()

            rocs = []
            for col_idx in range(y_pred_np.shape[1]):
                col_preds = y_pred_np[:, col_idx]
                col_trues = y_true_np[:, col_idx]

                # Si la predicción de esa columna es de una sola clase
                if len(np.unique(np.round(col_preds))) < 2:
                    # Retornamos 0 en esa columna (en numpy)
                    rocs.append(0.0)
                else:
                    rocs.append(roc_auc_score(col_trues, col_preds))

            # Promedio de ROC AUC (número puro en numpy)
            avg_roc = np.mean(rocs)

            # Creamos un tensor escalar para retornar
            # requires_grad=True no hará que sea entrenable, dado que no es diferenciable.
            return torch.tensor(1-avg_roc, device=device, dtype=torch.float, requires_grad=True)

train_csv_path = os.path.expanduser("~/datasets/CXR8/LongTailCXR/nih-cxr-lt_single-label_train.csv")
train_image_dir = os.path.expanduser("~/datasets/CXR8/images/images_001/images/")
test_csv_path = os.path.expanduser("~/datasets/CXR8/LongTailCXR/nih-cxr-lt_single-label_test.csv")
train_image_dir = os.path.expanduser("~/datasets/CXR8/images/images_001/images/")
batch_size = 20
image_scale = (224, 224)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale (1 channel)
    transforms.Resize(image_scale),               # Resize images to a uniform size
    transforms.ToTensor(),                       # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for grayscale (mean and std for a single channel)
])

train_dataset = CXR8Dataset(csv_path=train_csv_path, image_dir=train_image_dir, transform=transform)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = CXR8Dataset(csv_path=test_csv_path, image_dir=train_image_dir, transform=transform)
test_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

print(f"Number of images in the filtered dataset: {len(train_dataset)}")

# Test the DataLoader
for images, labels in train_data_loader:
    print(f"Batch of images: {images.shape}")
    print(f"Batch of labels: {labels.shape}")
    break

num_experts = 4
model = MoEModel(num_experts=num_experts).to('cuda')
num_epochs = 15

criterion = AverageROCAUC()
optimizer = Adam(model.parameters(), lr=0.001)

categs = test_data_loader.dataset.data.columns.to_list()
categs.remove("id")
categs.remove("subject_id")
len(categs)

def addnewrow(dataframe, row_list):
    """
    Agrega una nueva fila a un DataFrame dado a partir de una lista.
    
    Parámetros:
    - dataframe: pd.DataFrame. El DataFrame al que se agregará la nueva fila.
    - row_list: list. Una lista que contiene los valores de la nueva fila.

    Retorna:
    - pd.DataFrame: El DataFrame actualizado con la nueva fila.
    """
    if len(row_list) != len(dataframe.columns):
        raise ValueError("La longitud de la lista no coincide con el número de columnas del DataFrame")
    
    # Convertimos la lista en un DataFrame temporal
    new_row = pd.DataFrame([row_list], columns=dataframe.columns)
    
    # Concatenamos el DataFrame temporal con el original
    dataframe = pd.concat([dataframe, new_row], ignore_index=True)
    return dataframe

def average_roc_auc(df_pred, df_test):
    ret = 0
    for col in df_pred.columns.to_list():
        metrics = Evaluator.eval_classification(y_pred=df_pred[col], y_true=df_test[col], binary_classification=True)
        ret += metrics["roc_auc"]
    return ret/len(df_pred.columns.to_list())

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_sparsity = 0
    num_batches = 0

    for images, labels in train_data_loader:
        images, labels = images.to('cuda'), labels.to('cuda')

        # Forward pass
        outputs, gating_scores = model(images)

        # Compute loss
        loss = criterion(outputs, labels)  # Multi-label loss
        total_loss += loss.item()

        # Calculate sparsity
        sparsity = 1 - (gating_scores.sum(dim=1) / model.num_experts).mean().item()
        total_sparsity += sparsity

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_batches += 1

    avg_loss = total_loss / len(train_data_loader)
    avg_sparsity = total_sparsity / num_batches
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Sparsity: {avg_sparsity:.4f}")


model.eval()

df_test = pd.DataFrame(columns=categs)
df_pred = pd.DataFrame(columns=categs)

with torch.no_grad():
    for images, labels in test_data_loader:
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs, _ = model(images)
        
        # Apply sigmoid to outputs and threshold at 0.5
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        #print(len(predictions.detach().cpu().numpy().tolist()))
        print(len(labels.detach().cpu().numpy().tolist()[0]))
        # Collect predictions and labels

        for x in predictions:
            df_pred = addnewrow(df_pred, x.cpu().numpy().tolist())
        for x in labels:
            df_test = addnewrow(df_test, x.cpu().numpy().tolist())