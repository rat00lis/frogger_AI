import random
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
        
        # optimizador
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
    def __init__(self, learning_rate=0.0005, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9995, batch_size=32, n_actions=5, num_channels=1, input_dimension_x=84, input_dimension_y=84, fc1_dimension=512, fc2_dimension=512, memory_size=1000000, device="cuda"):
        #Hiperparámetros
        self.learning_rate = learning_rate # tasa de aprendizaje
        self.gamma = gamma # factor de descuento
        self.epsilon = epsilon # factor de exploración
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size # tamaño de instancias de entrenamiento
        self.n_actions = n_actions # número de acciones posibles
        self.num_channels = num_channels # número de canales de la imagen
        
        # Dimensiones de la entrada
        self.input_dimension_x = input_dimension_x  
        self.input_dimension_y = input_dimension_y

        # Dimensiones de las capas totalmente conectadas
        self.fc1_dimension = fc1_dimension
        self.fc2_dimension = fc2_dimension

        # Tamaño del buffer de memoria
        self.memory_size = memory_size
        self.memory_counter = 0
        
        # Dispocitivo de entrenamiento
        self.device = device

        # Inicializar la red neuronal
        self.model = DQN(self.learning_rate, self.input_dimension_x, self.input_dimension_y, self.num_channels, self.fc1_dimension, self.fc2_dimension, self.n_actions).to(self.device)

        # red de target
        self.target_model = DQN(self.learning_rate, self.input_dimension_x, self.input_dimension_y, self.num_channels, self.fc1_dimension, self.fc2_dimension, self.n_actions).to(self.device)

        # Inicializar memoria de repeticion
        self.state_memory = np.zeros((self.memory_size, self.num_channels, self.input_dimension_x, self.input_dimension_y), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, self.num_channels, self.input_dimension_x, self.input_dimension_y), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

    def save_memory(self, state, action, reward, new_state, done):
        # Guardar la memoria de repetición
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = done
        self.memory_counter += 1

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.choice(range(self.n_actions)) # Exploración
        else:
            state = preprocess_observation(observation).reshape(1, self.num_channels, self.input_dimension_x, self.input_dimension_y)
            actions = self.model.forward(state)
            return torch.argmax(actions).item()
        
    def learn(self):
        # Si la memoria de repetición no tiene suficientes instancias, no se aprende
        if self.memory_counter < self.batch_size:
            return
        
        self.model.optimizer.zero_grad() # Reiniciar gradientes

        max_memory = min(self.memory_counter, self.memory_size)
        random_batch = np.random.choice(max_memory, self.batch_size, replace=False) # Seleccionar un lote aleatorio de instancias de entrenamiento

        batch_index = np.arange(self.batch_size, dtype=np.int32) # índices de las instancias de entrenamiento

