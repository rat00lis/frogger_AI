import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Common import *

class DQN(nn.Module):
    def __init__(self, learning_rate, input_dimension_x, input_dimension_y, num_channels, fc1_dimension, fc2_dimension, n_actions):
        super(DQN, self).__init__()

        # Hiperparámetros
        k1, k2, k3 = 8, 4, 3 # kernel size
        s1, s2, s3 = 4, 2, 1 # stride
        o1, o2, o3 = 32, 64, 64 # output channels

        # Dimensines del output
        # (W - K) / S + 1
        # to do: padding
        def conv_size(size, k, s):
            return (size - k) // s + 1
        
        self.output_dimension_x = conv_size(conv_size(conv_size(input_dimension_x, k1, s1), k2, s2), k3, s3)
        self.output_dimension_y = conv_size(conv_size(conv_size(input_dimension_y, k1, s1), k2, s2), k3, s3)

        # Dimensiones de las capas totalmente conectadas
        self.fc1_dimension = fc1_dimension
        self.fc2_dimension = fc2_dimension

        # numero de acciones posibles
        self.n_actions = n_actions

        # capas convolucionales
        self.conv1 = nn.Conv2d(num_channels, o1, k1, s1)
        self.conv2 = nn.Conv2d(o1, o2, k2, s2)
        self.conv3 = nn.Conv2d(o2, o3, k3, s3)

        # capas totalmente conectadas
        self.fc1 = nn.Linear(self.output_dimension_x * self.output_dimension_y * o3, self.fc1_dimension)
        self.fc2 = nn.Linear(self.fc1_dimension, self.fc2_dimension)
        self.fc3 = nn.Linear(self.fc2_dimension, self.n_actions)

        # funcion de perdida
        self.loss_function = nn.MSELoss() # Mean Squared Error Loss
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)

    def forward(self, observation):
        # capas convolucionales
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))

        # aplanar la salida de las capas convolucionales
        observation = observation.view(-1, self.output_dimension_x * self.output_dimension_y * 64)

        # capas totalmente conectadas
        observation = F.relu(self.fc1(observation))
        observation = F.relu(self.fc2(observation))
        observation = self.fc3(observation)

        return observation
    

class AgentDQN:
    def __init__(self, learning_rate=0.0005, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9995, batch_size=32, n_actions=5, num_channels=4, input_dimension_x=84, input_dimension_y=84, fc1_dimension=512, fc2_dimension=512, memory_size=1000, device=None, action_mapping={0:0, 1:1, 2:4, 3:3, 4:2}):
        # Hiperparámetros
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.num_channels = num_channels

        # Dimensiones de la entrada
        self.input_dimension_x = input_dimension_x
        self.input_dimension_y = input_dimension_y

        # Dimensiones de las capas totalmente conectadas
        self.fc1_dimension = fc1_dimension
        self.fc2_dimension = fc2_dimension

        # Tamaño del buffer de memoria
        self.memory_size = memory_size
        self.memory_counter = 0

        # Dispositivo de entrenamiento
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(
            self.learning_rate,
            self.input_dimension_x,
            self.input_dimension_y,
            self.num_channels,
            self.fc1_dimension,
            self.fc2_dimension,
            self.n_actions
        ).to(self.device)

        # red de target
        self.target_model = DQN(self.learning_rate, self.input_dimension_x, self.input_dimension_y, self.num_channels, self.fc1_dimension, self.fc2_dimension, self.n_actions).to(self.device)

        # Inicializar memoria de repeticion
        self.state_memory = np.zeros((self.memory_size, self.num_channels, self.input_dimension_x, self.input_dimension_y), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, self.num_channels, self.input_dimension_x, self.input_dimension_y), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

        self.action_mapping = action_mapping

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.model.optimizer, step_size=1000, gamma=0.9)

    def save_memory(self, state, action, reward, new_state, done):
        # Guardar la memoria de repetición
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done
        self.memory_counter += 1
        
    def choose_action(self, observation, deterministic=False):
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.choice(range(self.n_actions)) # Exploración
        else:
            state = preprocess_observation(observation).reshape(1, self.num_channels, self.input_dimension_x, self.input_dimension_y)
            state = torch.tensor(state, dtype=torch.float32).to(self.device) # Convert to PyTorch tensor
            actions = self.model.forward(state)
            return torch.argmax(actions).item()
        
    def store_transition(self, state, action, reward, new_state, done):
        index = self.memory_counter % self.memory_size
        state = preprocess_observation(state)
        new_state = preprocess_observation(new_state)
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done
    
        self.memory_counter += 1

    def learn(self):
        # Si la memoria de repetición no tiene suficientes instancias, no se aprende
        if self.memory_counter < self.batch_size:
            return
        
        self.model.optimizer.zero_grad() # Reiniciar gradientes
    
        # Seleccionar un lote aleatorio de transiciones
        max_mem = min(self.memory_counter, self.memory_size)
        random_batch = np.random.choice(max_mem, self.batch_size, replace=False)
    
        # Extraer los estados, acciones, recompensas, nuevos estados y terminales del lote
        state_batch = torch.tensor(self.state_memory[random_batch]).to(self.device)
        action_batch = torch.tensor(self.action_memory[random_batch], dtype=torch.int64).to(self.device)
        reward_batch = torch.tensor(self.reward_memory[random_batch]).to(self.device)
        new_state_batch = torch.tensor(self.new_state_memory[random_batch]).to(self.device)
        terminal_batch = torch.tensor(self.terminal_memory[random_batch]).to(self.device)
    
        # Calcular los valores Q actuales y los valores Q objetivo
        q_eval = self.model.forward(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
        q_next = self.target_model.forward(new_state_batch).max(1)[0]
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * q_next
    
        # Calcular la pérdida y realizar la retropropagación
        loss = self.model.loss_function(q_eval, q_target)
        loss.backward()
        self.model.optimizer.step()

        # Step the learning rate scheduler
        self.scheduler.step()

    def evaluate(self, env, num_episodes=10, debug=False):
        self.model.eval()
        with torch.no_grad():
            rewards = []
            for _ in range(num_episodes):
                reward = 0
                done = False
                observation = env.reset()
                while not done:
                    action = self.choose_action(observation)
                    observation_, reward_, done, info = env.step(action_mapping[action])
                    reward += reward_
                    observation = observation_
                rewards.append(reward)
            avg = np.mean(rewards)
            if debug:
                print(f"Average reward: {avg}")
                print(f"Rewards: {rewards}")
        self.model.train()
        return avg
    
    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save(self, path):
        path = path + ".pt"
        torch.save(self.model.state_dict(), path)
        print(f"Model saved in {path}")

    def load(self, path):
        path = path + ".pt"
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")