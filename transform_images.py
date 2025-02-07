import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
import optuna
from joblib import Parallel, delayed
from utils.Loader import NEUDataset
from utils.Perspectiver import Perspectiver
from source.Prototype1 import Prototype1

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import multiprocessing
import torch
from mpl_toolkits.mplot3d import Axes3D
from skimage.restoration import denoise_wavelet
import random
import math
from PIL import Image
from collections import deque
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, label, generate_binary_structure
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.ndimage import label as ndi_label, binary_dilation

loaded_model = Prototype1(num_attention_heads=16)
loaded_model.load_state_dict(torch.load("h1.pth", map_location=torch.device('cpu')))
loaded_model.to("cuda")
loaded_model.eval()

def save_image(path, image):
    """
    Guarda la imagen en la ruta especificada, creando la carpeta si no existe.
    
    Args:
        path (str): Ruta completa donde se guardará la imagen.
        image (numpy.ndarray): Imagen a guardar.
    """
    # Extraer el directorio de la ruta
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)  # Crear carpeta si no existe

    # Guardar la imagen en la ruta indicada
    cv2.imwrite(path, image)

dataset = NEUDataset(set="train", seed=555, scale=0.5, best_param=True, output_path="outputs_k10")

for i in range(len(dataset)):
    image, label, best_params = dataset.__getitem__(index=i)

    original_image = Perspectiver.grayscale_to_rgb(Perspectiver.normalize_to_uint8(image.detach().cpu().numpy()[0]))

    path = "clustered_metal_dataset" + dataset.data["Path"][i]
    sp = dataset.data["sp"][i]
    sr = dataset.data["sr"][i]
    k = dataset.data["k"][i]
    clustered_image = Perspectiver.kmeansClustering(Perspectiver.meanShift(original_image, sp, sr), k=k)

    save_image(path, clustered_image)

