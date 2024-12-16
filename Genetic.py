import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
from collections import deque
from Common import preprocess_observation
import signal

torch.set_default_dtype(torch.float32)

class GeneticNet(nn.Module):
    def __init__(self, input_channels, output_size):
        super(GeneticNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc_input_size = 64 * 7 * 7  
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class GeneticAgent:
    def __init__(self, input_channels, output_size=5, learning_rate=0.3, gamma=0.9, memory_size=100, batch_size=16, target_update_interval=10):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = gamma
        
        self.memory_size = memory_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, input_channels, 84, 84), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, input_channels, 84, 84), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool)

        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.update_count = 0  

        self.model = GeneticNet(input_channels, output_size).to(torch.float32).to(self.device)
        self.target_model = GeneticNet(input_channels, output_size).to(torch.float32).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() 
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

    def choose_action(self, state, epsilon=0.1, deterministic=False):
        if np.random.rand() < epsilon and not deterministic:  
            return np.random.randint(0, self.model.fc2.out_features)
        else:  
            state = preprocess_observation(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device).to(self.model.fc1.weight.dtype)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
        
    def save_transition(self, state, action, reward, new_state, done):
        index = self.memory_counter % self.memory_size
        state = preprocess_observation(state)
        
        new_state = preprocess_observation(new_state)
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done
    
        self.memory_counter += 1

    def train_from_memory(self, force_train=False):
        """Entrena la red usando un batch del buffer de memoria."""
        if self.memory_counter < self.batch_size:  
            return

        max_mem = min(self.memory_counter, self.memory_size)
        random_batch = np.random.choice(max_mem, self.batch_size, replace=False)
    
        states = torch.tensor(self.state_memory[random_batch]).to(self.device)
        actions = torch.tensor(self.action_memory[random_batch], dtype=torch.int64).to(self.device)
        rewards = torch.tensor(self.reward_memory[random_batch]).to(self.device)
        next_states = torch.tensor(self.new_state_memory[random_batch]).to(self.device)
        dones = torch.tensor(self.terminal_memory[random_batch]).to(self.device)

        assert states.shape[1:] == (self.model.conv1.in_channels, 84, 84), f"Expected state input shape {(self.model.conv1.in_channels, 84, 84)}, but got {states.shape[1:]}"
        assert next_states.shape[1:] == (self.model.conv1.in_channels, 84, 84), f"Expected next_state input shape {(self.model.conv1.in_channels, 84, 84)}, but got {next_states.shape[1:]}"

        q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones * 1)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_interval == 0:
            self.update_target_network()

    def reset_memory(self):
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, self.model.conv1.in_channels, 84, 84), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, self.model.conv1.in_channels, 84, 84), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool)

    def update_target_network(self):
        """Sincroniza los pesos de la red principal con la red objetivo."""
        self.target_model.load_state_dict(self.model.state_dict())
    def get_genome(self):
        return [param.data.numpy() for param in self.model.parameters()]

    def set_genome(self, genome):
        for param, new_weights in zip(self.model.parameters(), genome):
            param.data = torch.tensor(new_weights)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path))

def initialize_population(population_size, input_size, output_size):
    return [GeneticAgent(input_size, output_size) for _ in range(population_size)]

def evaluate_agent(env, agent, steps=1000):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(steps):
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        if not truncated and not terminated:
            if action == 0:
                reward += -1
            elif action == 1:
                reward += 3
            elif (action == 2 or action == 3):
                reward += 0
            elif action == 4:
                reward += -1
        elif truncated:
            reward += -3
        
        agent.save_transition(state, action, reward, next_state, terminated or truncated)
        agent.train_from_memory()

        total_reward += reward

        if terminated or truncated:
            break

        state = next_state

    agent.update_target_network()
    return total_reward

def select_parents(population, fitness_scores):
    sorted_indices = np.argsort(fitness_scores)[::-1]  
    return [population[i] for i in sorted_indices[:len(population)//2]]

def crossover(parent1, parent2):
    genome1 = parent1.get_genome()
    genome2 = parent2.get_genome()

    new_genome1 = [(g1 + g2) / 2 for g1, g2 in zip(genome1, genome2)] 
    new_genome2 = [(g2 + g1) / 2 for g1, g2 in zip(genome1, genome2)]

    parent1.set_genome(new_genome1)
    parent2.set_genome(new_genome2)

    return parent1, parent2

def mutate(agent, mutation_rate = 0.1):
    for param in agent.model.parameters():
        if np.random.rand() < mutation_rate:
            param.data += torch.randn_like(param) * 0.1  

def train_genetic_algorithm(env_name, generations, population_size, input_size, output_size, steps=1000, load_model=None):

    def save_agent_and_exit(sig, frame):
        print("Saving agent and exiting...")
        agent.save("best_agent_stopped.pt")
        env.close()
        exit()

    signal.signal(signal.SIGINT, save_agent_and_exit)

    if load_model:
        best_agent = GeneticAgent(input_size, output_size)
        best_agent.load(load_model)
        print("Model loaded")
        
        population = [best_agent for _ in range(population_size)]
    else:
        population = initialize_population(population_size, input_size, output_size)
    gym.register_envs(ale_py)
    env = gym.make(env_name)
    env = gym.wrappers.FrameStackObservation(env, 1) 
    prev_fitness = 0

    best_agent = None

    for generation in range(generations):
        print(f"Generation {generation + 1}")

        fitness_scores = []
        for i, agent in enumerate(population):
            score = evaluate_agent(env, agent, steps)
            fitness_scores.append(score)
            print(f"Iteration {i + 1}: Score = {score}")
        
        best_fitness = max(fitness_scores)
        print(f"Best Fitness: {best_fitness}")
        if best_fitness > prev_fitness:
            best_agent = population[np.argmax(fitness_scores)]
            best_agent.save('best_agent.pt')
            print("Model saved")
            prev_fitness = best_fitness

        new_population = []

        num_elites = 2  
        sorted_indices = np.argsort(fitness_scores)[::-1]
        new_population = [population[i] for i in sorted_indices[:num_elites]]

        parents = select_parents(population, fitness_scores) 

        while len(new_population) < population_size:
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            child1, child2 = crossover(parent1, parent2)

            
            if np.random.rand() < 0.5: mutate(child1)
            if np.random.rand() < 0.5: mutate(child2)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population
    
    print("Saving best agent from last generation")
    best_agent.save('best_agent_last_gen.pt')
    env.close()

if __name__ == "__main__":
    train_genetic_algorithm(
        env_name="ALE/Frogger-v5",
        generations=2,
        population_size=4,
        input_size=1,  
        output_size=5,  
        steps=1000,
        load_model="best_agent_21_54.pt"
    )