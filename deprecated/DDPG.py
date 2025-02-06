# Imports
import torch
import torch.nn as nn
import gym
import numpy as np
from gym import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from source.Prototype1 import Prototype1
from utils.Loader import NEUDataset
from utils.Perspectiver import Perspectiver
from fosas import optimized_get_mean_pit_area

# Función para calcular recompensa
def calculate_reward(image, sp, sr):
    image = Perspectiver.grayscale_to_rgb(
        Perspectiver.normalize_to_uint8(image.detach().cpu().numpy()[0][0])
    )
    after = Perspectiver.kmeansClustering(
        Perspectiver.meanShift(image, sp, sr), k=5
    )
    try:
        return optimized_get_mean_pit_area(after)
    except:
        return -100

def reward_function(output, image):
    """Calcula la recompensa basada en la salida del modelo y la imagen."""
    output = output.detach().cpu().numpy()
    sp, sr = output[0], output[1]
    if sp <= 1:
        return sp * 50 - 5 
    if sr <= 1:
        return sr * 50 - 5
    if sp >= 100:
        return (-50) * (abs(50-sp)) - 5 
    if sr >= 100:
        return (-50) * (abs(50-sr)) - 5
    print(f"sp: {sp}, sr: {sr}")
    return calculate_reward(image, 1 if sp <= 1 else sp, 1 if sr <= 1 else sr)

# Entorno personalizado: observa imágenes y acciones continuas (2 valores)
class ImageEnv(gym.Env):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 100, 100), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        self.index = 0
        self.state = None

    def reset(self):
        self.index = np.random.randint(0, len(self.dataset))
        image, _ = self.dataset[self.index]  # image: [1, 100, 100]
        self.state = image.unsqueeze(0).numpy()  # state: [1, 1, 100, 100]
        return self.state

    def step(self, action):
        img_tensor = torch.from_numpy(self.state)
        r = reward_function(torch.FloatTensor(action), img_tensor)
        done = True  # Episodio corto (single-step)
        info = {}
        next_obs = self.reset()
        return next_obs, r, done, info

# Extractor de características usando Prototype1
class Prototype1Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, num_attention_heads=16):
        # features_dim debe coincidir con la salida de Prototype1 (2)
        super().__init__(observation_space, features_dim=2)
        self.model = Prototype1(num_attention_heads=num_attention_heads)
        print("Inicialización correcta del extractor Prototype1")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [batch, canales, alto, ancho]
        return self.model(obs)
    
    def getModel(self):
        return self.model

# Wrapper que une extractor y actor para obtener la acción a partir de la imagen
class ActorWrapper(nn.Module):
    def __init__(self, features_extractor: nn.Module, actor: nn.Module):
        super().__init__()
        self.features_extractor = features_extractor
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        action = self.actor(features)
        return action

# Ejecución principal
if __name__ == "__main__":
    # Cargar dataset y crear entorno
    dataset = NEUDataset(set="train", scale=0.5)
    env = ImageEnv(dataset)

    # Configuración del extractor personalizado en policy_kwargs
    policy_kwargs = dict(
        features_extractor_class=Prototype1Extractor,
        features_extractor_kwargs=dict(num_attention_heads=16),
    )

    # Crear ruido para las acciones (requerido por DDPG)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Crear y entrenar el modelo DDPG
    model = DDPG(
        "CnnPolicy",  # Política basada en CNN para imágenes
        env,
        policy_kwargs=policy_kwargs,
        action_noise=action_noise,
        verbose=1,
        learning_rate=5e-5,
        batch_size=512,
        buffer_size=10_000,
        gradient_steps=55,
    )

    model.learn(total_timesteps=10000, log_interval=100, progress_bar=True)
    model.save("DDPG_TEST")

    # Forzar la inicialización del policy
    _ = model.policy
    print("Features Extractor:", model.policy.features_extractor)
    if model.policy.features_extractor is None:
        print("⚠️ Warning: Stable-Baselines3 no creó el extractor. Asignándolo manualmente...")
        model.policy.features_extractor = Prototype1Extractor(env.observation_space)

    # Extraer el actor entrenado (la parte que genera acciones)
    actor_network = model.policy.actor

    # Crear el wrapper que une el extractor y el actor
    full_actor = ActorWrapper(model.policy.features_extractor, actor_network)
    full_actor.eval()

    # Guardar únicamente el actor (wrapper) y los parámetros necesarios
    torch.save({
        'state_dict': full_actor.state_dict(),
        'model_class': 'ActorWrapper',
        'num_attention_heads': 16
    }, "actor_ddpg.pth")
    print("Modelo del actor guardado exitosamente en actor_ddpg.pth")
