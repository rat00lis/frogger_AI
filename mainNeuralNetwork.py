import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Modelo de red neuronal para el agente
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class NeuralAgent:
    def __init__(self, input_size, output_size, hidden_size=128):
        self.model = NeuralNetwork(input_size, output_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def choose_action(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:  # Exploración
            return np.random.randint(0, self.model.fc2.out_features)
        else:  # Explotación
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def update(self, state, action, reward, next_state, done, discount_factor=0.99):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        q_values = self.model(state_tensor)

        with torch.no_grad():
            next_q_values = self.model(next_state_tensor)
            target = reward + (discount_factor * torch.max(next_q_values).item() * (1 - done))

        target_q_values = q_values.clone()
        target_q_values[0, action] = target

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_genome(self):
        return [param.data.numpy() for param in self.model.parameters()]

    def set_genome(self, genome):
        for param, new_weights in zip(self.model.parameters(), genome):
            param.data = torch.tensor(new_weights)

def initialize_population(population_size, input_size, output_size):
    return [NeuralAgent(input_size, output_size) for _ in range(population_size)]

def evaluate_agent(env, agent, steps=1000):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(steps):
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        agent.update(state, action, reward, next_state, terminated)
        total_reward += reward

        if terminated or truncated:
            break

        state = next_state

    return total_reward

def select_parents(population, fitness_scores):
    sorted_indices = np.argsort(fitness_scores)[::-1]  # Ordenar por fitness descendente
    return [population[i] for i in sorted_indices[:len(population)//2]]

def crossover(parent1, parent2):
    child1 = NeuralAgent(parent1.model.fc1.in_features, parent1.model.fc2.out_features)
    child2 = NeuralAgent(parent2.model.fc1.in_features, parent2.model.fc2.out_features)

    genome1 = parent1.get_genome()
    genome2 = parent2.get_genome()

    new_genome1 = [(g1 + g2) / 2 for g1, g2 in zip(genome1, genome2)]
    new_genome2 = [(g2 + g1) / 2 for g1, g2 in zip(genome1, genome2)]

    child1.set_genome(new_genome1)
    child2.set_genome(new_genome2)

    return child1, child2

def mutate(agent, mutation_rate=0.1):
    genome = agent.get_genome()
    for i, layer_weights in enumerate(genome):
        mutation_mask = np.random.rand(*layer_weights.shape) < mutation_rate
        genome[i] = layer_weights + mutation_mask * np.random.normal(size=layer_weights.shape)
    agent.set_genome(genome)

def run_genetic_algorithm(env_name, generations, population_size, input_size, output_size, steps=1000):
    population = initialize_population(population_size, input_size, output_size)
    gym.register_envs(ale_py)
    env = gym.make(env_name)

    for generation in range(generations):
        print(f"Generation {generation + 1}")

        fitness_scores = [evaluate_agent(env, agent, steps) for agent in population]

        best_fitness = max(fitness_scores)
        print(f"Best Fitness: {best_fitness}")

        parents = select_parents(population, fitness_scores)

        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    best_agent = population[np.argmax(fitness_scores)]
    visualize_agent(env_name, best_agent, steps)

    env.close()
    return best_agent

def visualize_agent(env_name, agent, steps=1000):
    env = gym.make(env_name, render_mode="human")
    state, _ = env.reset()

    for _ in range(steps):
        action = agent.choose_action(state, epsilon=0)
        state, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    env.close()

if __name__ == "__main__":
    run_genetic_algorithm(
        env_name="ALE/Frogger-v5",
        generations=50,
        population_size=100,
        input_size=237,  # Ajustar según la discretización del estado
        output_size=5,   # Número de acciones disponibles
        steps=1000
    )
