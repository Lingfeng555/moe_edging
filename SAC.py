# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import math
import numpy as np
from sklearn.metrics import davies_bouldin_score
import gym
from gym import spaces
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from source.Prototype1 import Prototype1
from utils.Loader import NEUDataset
from utils.Perspectiver import Perspectiver

def calculate_reward(image, sp, sr):
    image = Perspectiver.grayscale_to_rgb(Perspectiver.normalize_to_uint8(image.detach().cpu().numpy()[0]))
    after = Perspectiver.meanShift(image, sp, sr)
    original_gray = Perspectiver.rgb_to_grayscale(image).flatten()
    clustered_gray = Perspectiver.rgb_to_grayscale(after).flatten()

    n_clusters = len(np.unique(after))
    if n_clusters < 20:
        return 10000/(n_clusters+1)
        
    score = davies_bouldin_score(original_gray.reshape(-1, 1), clustered_gray)

    # Metric to maximize: Silhouette Score per cluster
    return (math.sqrt(score)+n_clusters)

def reward_function(output, image): 
    """ Reward function based or the distance between the predicted values and correct values """

    output = output.detach().cpu().numpy()

    sp = output[0]
    sr = output[1]
    penalty_sp = (1000 * sp - 100) if sp <= 0 else 0
    penalty_sr = (1000 * sr - 100) if sr <= 0 else 0

    penalty_sp += -(50 * sp) if sp >= 100 else 0
    penalty_sr += -(50 * sr) if sr >= 100 else 0

    return (calculate_reward(image, 1 if sp <= 1 else sp,  1 if sr <= 1 else sr) + penalty_sp + penalty_sr)


if __name__ == '__main__':
   model = Prototype1(num_attention_heads=16) #Receives a [BatchSize, 1, 100, 100] image and returns [batch_size, 2]
   dataset = NEUDataset(set="train",scale=0.5)


