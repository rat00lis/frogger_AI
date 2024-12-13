import gymnasium as gym
import ale_py
from DQN import AgentDQN
from Random import AgentRandom
import numpy as np


# Soluciones actuales
solutions = ["Random"]


class Agent:
    # El agente elige una solución basada en el tipo de solución proporcionado
    def __init__(self, solution_type="Random", model_path=None):
        if solution_type == "DQN": # Si la solución es DQN, se crea un agente DQN
            self.model = AgentDQN(model_path)
        elif solution_type == "Random": # Si la solución es Random, se crea un agente Random
            self.model = AgentRandom()
        else:
            self.model = None

    def choose_action(self, observation):
        return self.model.choose_action(observation)

class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.2, discount_factor=0.99, epsilon=0.2):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.random.rand(num_states, num_actions)  # Inicializar Q-table aleatoriamente

    def preprocess_state(self, state):
        # Flatten the state array and convert it to a single integer index
        return hash(state.tostring()) % self.num_states
    
    # Elegir una acción basada en la política epsilon-greedy
    def choose_action(self, state):
        state_index = self.preprocess_state(state)
        if np.random.rand() < self.epsilon:  # Exploración
            response = np.random.randint(low=0, high=self.num_actions)
            return response
        else:  # Explotación
            response = np.argmax(self.q_table[state_index])
            return response
        
    def choose_action_no_epsilon(self, state):
        state_index = self.preprocess_state(state)
        response = np.argmax(self.q_table[state_index])
        return response

    # Actualizar la Q-table del agente
    def update(self, state, action, reward, next_state, done):
        state_index = self.preprocess_state(state)
        next_state_index = self.preprocess_state(next_state)
        target = reward
        if not done:
            target += self.discount_factor * np.max(self.q_table[next_state_index])

        #print(str(self.q_table[state_index, action]) + " to " + str(self.learning_rate * (target - self.q_table[state_index, action])) + " reward: " + str(reward))
        self.q_table[state_index, action] += self.learning_rate * (target - self.q_table[state_index, action])
        
    def print_q_table(self):
        print(self.q_table)

"""
# Ejecutar simulación de Frogger 
def run_frogger_simulation(steps=1000, render_mode="human", solution_type="Random", model_path=None):

    # Crear entorno de Frogger
    gym.register_envs(ale_py) # registrar entornos de ALE
    env = gym.make("ALE/Frogger-v5", render_mode=render_mode) # seleccionar entorno de Frogger
    observation, info = env.reset() # reiniciar entorno
    
    # crear agente con la solución proporcionada
    agent = Agent(solution_type, model_path)

    # ejecutar simulación
    for _ in range(steps): 
        # El agente elige una accion basada en la observación del estado actual del juego
        action = agent.choose_action(observation) 
        # El entorno toma la accion con .step() y devuelve:
        # - la observación del estado actual
        # - la recompensa obtenida
        # - si el juego ha terminado (la rana murio o llego al final)
        # - si el juego ha sido truncado (se ha alcanzado el límite de pasos)
        # - información adicional (metadatos)
        observation, reward, terminated, truncated, info = env.step(action) 

        # Si el juego ha terminado o ha sido truncado, reiniciar el entorno
        if terminated or truncated:
            observation, info = env.reset()

    env.close()

    # Funcion comparadora de soluciones
def evaluate_solutions():
    results = {}

    for solution in solutions:
        # Ejecutamos una simulación de Frogger para cada solución
        run_frogger_simulation(solution_type=solution)
        
        ##PLACEHOLDER##: Calcular métricas de rendimiento para cada solución
        # Las métricas pueden incluir:
        # - Puntaje total
        # - Tiempo de ejecución
        # - Número de vidas restantes
        # - Número de pasos realizados, etc

    # Aqui compararemos las soluciones y mostraremos los resultados
    for solution, metrics in results.items():
        # placeholder
        continue"""


def initialize_population(population_size, num_states, num_actions):
    return [QLearningAgent(num_states, num_actions) for _ in range(population_size)]

# Función para evaluar agentes utilizando su Q-table
def evaluate_agent(env, agent, steps=1000):
    state, _ = env.reset()
    total_reward = 0

    for _ in range(steps):
        action = agent.choose_action(state)
        next_state, stateReward, terminated, truncated, info = env.step(action)
        if not truncated:
            if action == 0:
                reward = -0.5
            elif action == 1:
                reward = 1
            elif (action == 2 or action == 3):
                reward = 0
            elif action == 4:
                reward = -1
        else:
            reward = -2
        agent.update(state, action, reward, next_state, terminated)

        total_reward += reward
        
        if terminated or truncated or info["lives"] == 3:
            break
        state = next_state

    return total_reward

# Función para seleccionar a los padres de la siguiente generación
def select_parents(population, fitness_scores):
    sorted_indices = np.argsort(fitness_scores)[::-1]  # Ordenar por fitness descendente
    return [population[i] for i in sorted_indices[:len(population)//2]]

# Función para cruzar dos agentes
def crossover(parent1, parent2):
    child1 = QLearningAgent(parent1.num_states, parent1.num_actions)
    child2 = QLearningAgent(parent2.num_states, parent2.num_actions)

    # Promedio entre Q-tables de los padres
    crossover_point = np.random.randint(0, parent1.num_states)
    child1.q_table[:crossover_point] = parent1.q_table[:crossover_point]
    child1.q_table[crossover_point:] = parent2.q_table[crossover_point:]
    child2.q_table[:crossover_point] = parent2.q_table[:crossover_point]
    child2.q_table[crossover_point:] = parent1.q_table[crossover_point:]

    return child1, child2

# Función para mutar la Q-table de un agente
def mutate(agent, mutation_rate=0.1):
    for state in range(agent.num_states):
        for action in range(agent.num_actions):
            if np.random.rand() < mutation_rate:
                agent.q_table[state, action] += np.random.normal()  # Perturbar ligeramente


# Algoritmo genético principal
def run_genetic_algorithm(env_name, generations, population_size, num_states, num_actions, steps=1000):
    population = initialize_population(population_size, num_states, num_actions)
    gym.register_envs(ale_py)
    env = gym.make(env_name)

    for generation in range(generations):
        print(f"Generation {generation+1}")

        # Evaluar la población
        fitness_scores = [evaluate_agent(env, agent, steps) for agent in population]

        # Mostrar estadísticas
        best_fitness = max(fitness_scores)
        best_agent = population[np.argmax(fitness_scores)]
        print(f"Best Fitness: {best_fitness}")

        # Seleccionar padres
        parents = select_parents(population, fitness_scores)

        # Crear nueva generación
        new_population = []
        """
        while len(new_population) < population_size:
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)"""
            
        for parent in parents:
        # Clonar el padre
            child = QLearningAgent(parent.num_states, parent.num_actions)
            child.q_table = np.copy(parent.q_table)  # Clonar la Q-table del padre

            # Mutar al hijo
            mutate(child)

            # Agregar el hijo a la nueva población
            new_population.append(child)

        # Si la población aún no está completa (por ejemplo, tamaño impar), rellenar con copias adicionales
        while len(new_population) < population_size:
            # Seleccionar aleatoriamente un padre, clonarlo, y mutarlo
            parent = np.random.choice(parents)
            child = QLearningAgent(parent.num_states, parent.num_actions)
            child.q_table = np.copy(parent.q_table)
            mutate(child)
            new_population.append(child)

            population = new_population

    #visualizar el mejor agente
    best_agent = population[np.argmax(fitness_scores)]
    visualize_agent(env_name="ALE/Frogger-v5", agent=best_agent, steps=1000)
    
    env.close()
    return best_agent

def visualize_agent(env_name, agent, steps=1000):
    """
    Visualiza el mejor agente de la generación en el entorno.
    
    Args:
        env_name (str): Nombre del entorno.
        best_agent (Agent): Agente con el mejor fitness de la generación.
        steps (int): Número de pasos para ejecutar la simulación.
    """
    env = gym.make(env_name, render_mode="human")  # Modo humano para mostrar visualmente
    state, _ = env.reset()  # Reiniciar el entorno

    for step in range(steps):
        # El agente elige una acción
        action = agent.choose_action_no_epsilon(state)
        next_state, reward, terminated, truncated, info = env.step(action)

        # Si el juego termina o es truncado, reiniciar el entorno
        if terminated or truncated or info["lives"] == 3:
            break
        state = next_state

    env.close()



if __name__ == "__main__":
    run_genetic_algorithm(
        env_name="ALE/Frogger-v5",
        generations=50,
        population_size=100,
        num_states=237,  # Ajustar según la discretización del entorno
        num_actions=5,  # Ajustar según las acciones disponibles
        steps=1000
    )
