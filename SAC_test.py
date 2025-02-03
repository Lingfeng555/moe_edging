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
    image = Perspectiver.grayscale_to_rgb(Perspectiver.normalize_to_uint8(image.detach().cpu().numpy()[0][0]))
    after = Perspectiver.meanShift(image, sp, sr)
    original_gray = Perspectiver.rgb_to_grayscale(image).flatten()
    clustered_gray = Perspectiver.rgb_to_grayscale(after).flatten()

    n_clusters = len(np.unique(after))
    if n_clusters < 20:
        return -10000/(n_clusters+1)
        
    score = davies_bouldin_score(original_gray.reshape(-1, 1), clustered_gray)

    # Metric to maximize: Silhouette Score per cluster
    return (math.sqrt(score)/2+n_clusters)

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

# Entorno personalizado: observa imágenes, acciones continuas [sp, sr]
class ImageEnv(gym.Env):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,100,100), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        self.index = 0
        self.state = None

    def reset(self):
        self.index = np.random.randint(0, len(self.dataset))
        image, _ = self.dataset[self.index]       # image: [1, 100, 100]
        self.state = image.unsqueeze(0).numpy()   # state: [1, 1, 100, 100]
        #self.index = (self.index+1) % len(self.dataset)
        return self.state

    def step(self, action):
        # Reward y nueva observación
        img_tensor = torch.from_numpy(self.state)
        r = reward_function(torch.FloatTensor(action), img_tensor)
        done = True    # Ep. corto (single-step) o ajusta según tu lógica
        info = {}
        next_obs = self.reset()  # Nuevamente, o mantén la misma para multi-step
        return next_obs, r, done, info

# Extractor de características usando tu Prototype1 (para Stable-Baselines3)
class Prototype1Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, num_attention_heads=16):
        # 'features_dim' = 2, porque Prototype1 retorna [batch, 2]
        super().__init__(observation_space, features_dim=2)
        self.model = Prototype1(num_attention_heads=num_attention_heads)
        print("Inicilizacion correcta")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs llega en forma [batch, canales, alto, ancho], ej: [N,1,100,100]
        return self.model(obs)  # Salida [N,2]
    
    def getModel(self):
        return self.model

# Ejecución
if __name__ == "__main__":
    # Carga tu dataset
    dataset = NEUDataset(set="train", scale=0.5)
    env = ImageEnv(dataset)

    # Indica en policy_kwargs tu extractor personalizado
    policy_kwargs = dict(
        features_extractor_class=Prototype1Extractor,
        features_extractor_kwargs=dict(num_attention_heads=16),
    )

    # Ajusta el tamaño del buffer para no agotar memoria
    model = SAC(
        #"MlpPolicy",
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=5e-5,
        batch_size=64,
        buffer_size=10_000  # reduce para evitar un uso excesivo de RAM
    )

    model.learn(total_timesteps=40000)

    model.save(path="SAC_TEST")

    # Asegurarse de que el extractor de características está correctamente inicializado
    _ = model.policy  # Forzar la inicialización del policy extractor

    # Verificar si el extractor se ha inicializado correctamente
    print("Features Extractor:", model.policy.features_extractor)

    if model.policy.features_extractor is None:
        print("⚠️  Warning: Stable-Baselines3 no creó el extractor. Asignándolo manualmente...")
        model.policy.features_extractor = Prototype1Extractor(env.observation_space)


    # Extraer la red neuronal entrenada desde el modelo SAC
    trained_model = model.policy.features_extractor.model  # Prototype1 dentro de Prototype1Extractor

    # Verificar que el modelo se extrajo correctamente
    print("Tipo de modelo extraído:", type(trained_model))
    print(trained_model)

    # Guardar solo la red neuronal en un archivo .pth
    torch.save(trained_model.state_dict(), "h1.pth")
    print("Modelo guardado exitosamente en h1.pth")


