import gymnasium as gym
import ale_py
import numpy as np
from DQN import AgentDQN
from Random import AgentRandom


# Soluciones actuales
solutions = ["DQN", "Random"]


class Agent:
    # El agente elige una solución basada en el tipo de solución proporcionado
    def __init__(self, solution_type="DQN", model_path=None):
        if solution_type == "DQN": # Si la solución es DQN, se crea un agente DQN
            self.model = AgentDQN()
        elif solution_type == "Random": # Si la solución es Random, se crea un agente Random
            self.model = AgentRandom()
        else:
            self.model = None

    def choose_action(self, observation, deterministic=False):
        return self.model.choose_action(observation, deterministic)
    
    def store_transition(self, state, action, reward, new_state, done):
        self.model.store_transition(state, action, reward, new_state, done)
    def learn(self):
        self.model.learn()
    def update_target_network(self):
        self.model.update_target_network()
    def save_memory(self, state, action, reward, new_state, done):
        self.model.save_memory(state, action, reward, new_state, done)
    def update_target_network(self):
        self.model.update_target()


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
        continue

def train_agent(solution_type, num_episodes, action_mapping):
    # Crear entorno de Frogger
    gym.register_envs(ale_py) # registrar entornos de ALE
    env = gym.make("ALE/Frogger-v5", obs_type="rgb") # seleccionar entorno de Frogger
    env = gym.wrappers.FrameStackObservation(env, 4) # apilar 4 fotogramas para capturar el movimiento
    # crear agente con la solución proporcionada
    agent = Agent(solution_type)

    greedy_count = 0 # contador de acciones greedy
    rewards = [] # lista de recompensas
    # Entrenar agente
    for episode in range(num_episodes):
        observation, info = env.reset() # reiniciar entorno
        done = False
        total_reward = 0

        step_num = 0
        while not done:
            if(step_num%150 == 0):
                greedy_count += 10
                step_num += 1

            if(greedy_count > 0):
                action = agent.choose_action(observation, deterministic = True)
                greedy_count -= 1
            else:
                action = agent.choose_action(observation)
                step_num +=1
            observation_, reward, terminated, truncated, info = env.step(action_mapping[action])
            done = terminated or truncated
            total_reward += reward
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        
        agent.update_target_network()
        rewards.append(total_reward)
        avg = np.mean(rewards[-100:])
        print(f"Episode {episode+1}/{num_episodes}: Total Reward: {total_reward}, Average Reward: {avg}")

        # Imprimir resultados del episodio
        print(f"Episode {episode+1}/{num_episodes}: Total Reward: {total_reward}")

    env.close()



if __name__ == "__main__":
    # evaluate_solutions()
    action_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    train_agent("DQN", 1000, action_mapping)
