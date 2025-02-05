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
from fosas import optimized_get_mean_pit_area

def calculate_reward(image, sp, sr):
    image = Perspectiver.grayscale_to_rgb(Perspectiver.normalize_to_uint8(image.detach().cpu().numpy()[0][0]))
    after = Perspectiver.kmeansClustering(Perspectiver.meanShift(image, sp, sr), k = 5)
    try:
        return optimized_get_mean_pit_area(after)
    except: 
        return -100

def reward_function(output, image): 
    """ Reward function based or the distance between the predicted values and correct values """

    output = output.detach().cpu().numpy()

    sp = output[0]
    sr = output[1]
    
    if sp <= 0 : return sp * 50 - 5 
    if sr <= 0 : return sr * 50 - 5

    if sp >= 100 : return (-50)*(abs(50-sp)) - 5 
    if sr >= 100 : return (-50)*(abs(50-sr)) - 5

    print(f"sp: {sp}, sr: {sr}")

    return calculate_reward(image, 1 if sp <= 1 else sp,  1 if sr <= 1 else sr)

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
        batch_size=512,
        buffer_size=10_000  # reduce para evitar un uso excesivo de RAM
    )

    model.learn(total_timesteps=1000, log_interval=100, progress_bar=True)

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
    torch.save({
    'state_dict': trained_model.state_dict(),
    'model_class': 'Prototype1',  # You might not need this if it's imported elsewhere
    'num_attention_heads': 16  # Save architecture parameters
    }, "h1.pth")
    print("Modelo guardado exitosamente en h1.pth")


