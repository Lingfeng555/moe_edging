import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from sklearn.metrics import davies_bouldin_score
import numpy as np
from torch.utils.data import DataLoader
import itertools
import concurrent.futures

from utils.Loader import NEUDataset
from source.Prototype1 import Prototype1
from utils.Perspectiver import Perspectiver

MAX_WORKERS = 256

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        """ Save a transition (state, action, reward, next_state, done) """
        self.memory.append(transition)

    def sample(self, batch_size):
        """ Sample a batch of experiences """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def clear(self):
        self.memory.clear()
    
class RL_Agent:
    def __init__(self, gamma=0.99, lr=1e-3, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Neural Network
        self.model = Prototype1(num_attention_heads=16)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Experience Replay Memory
        self.memory = ReplayMemory(capacity=1000000)

    def predict(self, state): 
        """ Predicts two floating numbers instead of discrete action selection, state is a image"""
        with torch.no_grad():
            output = self.model(state)
        return output  # Output: Two continuous numbers

    def store_experience(self, experience):
        """ Save an experience tuple (state, output, reward, next_state, done) """
        #print("Memorando")
        self.memory.push(experience)

    def train_step(self, batch_size=32):
        batch = self.memory.sample(batch_size=batch_size)
        states, outputs, rewards, next_states, dones = zip(*batch)
        #print(states[0].shape)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        outputs = torch.stack(outputs).squeeze(1)
        rewards = torch.tensor(rewards).float().unsqueeze(1)
        dones = torch.tensor(dones).float().unsqueeze(1)


        # Esta parte esta mal

        # Compute target using Bellman equation
        next_outputs = self.model(next_states)
        rewards_expanded = rewards.expand(-1, 2)
        #rewards_expanded = rewards.expand_as(next_outputs)

        target_values = (rewards_expanded  + ((1 - dones) * self.gamma * next_outputs)) 
        
        # Compute loss
        loss = F.mse_loss(outputs[:, 0], target_values[:, 0]) + F.mse_loss(outputs[:, 1], target_values[:, 1])
        #loss = self.loss_fn(outputs, target_values)

        #for i in range(len(rewards)):
            #print(f"\tExample {i} -> Current output: {outputs[i]} Reward: {rewards[i]}, Target value: {target_values[i]}, dones: {dones[i]}, loss: {loss}")

        # 0--------------------------------------------------------------

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        print(f"\tAvg guess: {outputs.mean(dim=0)}, avg=target: {target_values.mean(dim=0)}")

class NEUEnvironment:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.index = 0  # Track current index
        self.batch_size = batch_size
        self.num_iterations_per_epoch = len(dataset) // batch_size

    def reset(self):
        """ Reset environment to initial state """
        self.index = 0
    
    def cicle(self, images, outputs):
        rewards = [None] * len(images)

        def compute_reward(i):
            #print(f"Tamaño de output: {images[i].shape}")
            return self.reward_function(outputs[i], images[i])
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(compute_reward, range(len(images))))
        rewards[:] = results

        self.index = (self.index + self.batch_size)%len(self.dataset)
        done = [(self.index == 0)] * len(images)
        return rewards, done
    
    def calculate_reward(self, image, sp, sr):
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

    def reward_function(self, output, image): 
        """ Reward function based or the distance between the predicted values and correct values """

        output = output.detach().cpu().numpy()
        sp = output[0]
        sr = output[1]

        penalty_sp = (1000 * sp - 100) if sp <= 0 else 0
        penalty_sr = (1000 * sr - 100) if sr <= 0 else 0

        penalty_sp += -(50 * sp) if sp >= 100 else 0
        penalty_sr += -(50 * sr) if sr >= 100 else 0

        return (self.calculate_reward(image, 1 if sp <= 1 else sp,  1 if sr <= 1 else sr) + penalty_sp + penalty_sr)
    
class Gym:
    def __init__(self, agent, enviroment):
        self.agent = agent
        self.enviroment = enviroment

    def train(self,num_epochs = 1000):

        reward_per_epoch = []

        for epoch in range(num_epochs):

            dataloader_iterator = itertools.cycle(self.enviroment.dataloader)

            for i in range(self.enviroment.num_iterations_per_epoch):
                images, labels = next(dataloader_iterator)
                outputs = self.agent.predict(images)

                rewards, dones = self.enviroment.cicle(images, outputs)
                next_images, labels = next(dataloader_iterator)

                #print(f"Reward: {np.array(rewards).shape}")
                def store_experience(experience):
                    self.agent.store_experience((
                        images[experience], 
                        outputs[experience], 
                        rewards[experience], 
                        next_images[experience], 
                        dones[experience]
                    ))

                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    executor.map(store_experience, range(self.enviroment.batch_size))
                
                self.agent.train_step(self.enviroment.batch_size)

                total_reward = np.sum(rewards)
            
            reward_per_epoch.append(total_reward)

            print(f"Episode {epoch}/{num_epochs}, "
              f"Total Reward: {total_reward:.3f}, AVG reward: {np.average(rewards)}, Epsilon: {self.agent.epsilon:.5f}, postive_rate: {np.sum(np.array(rewards) > 0) / len(rewards)}")

        return reward_per_epoch, self.agent.model

if __name__ == '__main__':
    agent = RL_Agent(gamma=0.99, lr=5e-3, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01)
    dataset = NEUDataset(set="train", scale=0.5)
    enviroment = NEUEnvironment(dataset, batch_size=256)
    gym = Gym(agent=agent, enviroment=enviroment)
    gym.train()
