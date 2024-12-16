import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
from DQN import AgentDQN
from Random import AgentRandom
import time
import csv
from tqdm import tqdm

class Agent:
    # El agente elige una solución basada en el tipo de solución proporcionado
    def __init__(self, solution_type="DQN", requires_training=False, model_path=None, config=None):
        if solution_type == "DQN": # Si la solución es DQN, se crea un agente DQN
            self.model = AgentDQN()
            if not requires_training and model_path:
                self.model.load(model_path)
            elif requires_training:
                if config:
                    learning_rate = config.get("learning_rate", 0.0005)
                    gamma = config.get("gamma", 0.99)
                    epsilon = config.get("epsilon", 1.0)
                    epsilon_min = config.get("epsilon_min", 0.1)
                    epsilon_decay = config.get("epsilon_decay", 0.9995)
                    batch_size = config.get("batch_size", 32)
                    n_actions = config.get("n_actions", 5)
                    num_channels = config.get("num_channels", 4)
                    input_dimension_x = config.get("input_dimension_x", 84)
                    input_dimension_y = config.get("input_dimension_y", 84)
                    fc1_dimension = config.get("fc1_dimension", 512)
                    fc2_dimension = config.get("fc2_dimension", 512)
                    memory_size = config.get("memory_size", 1000)
                    device = config.get("device", None)
                    action_mapping = config.get("action_mapping", {0:0, 1:1, 2:4, 3:3, 4:2})

                    self.model = AgentDQN(learning_rate=learning_rate, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay, batch_size=batch_size, n_actions=n_actions, num_channels=num_channels, input_dimension_x=input_dimension_x, input_dimension_y=input_dimension_y, fc1_dimension=fc1_dimension, fc2_dimension=fc2_dimension, memory_size=memory_size, device=device, action_mapping=action_mapping)
                else:
                    self.model = AgentDQN()
        elif solution_type == "Random": # Si la solución es Random, se crea un agente Random
            self.model = AgentRandom()
        else:
            self.model = None

    def choose_action(self, observation, deterministic=False):
        return self.model.choose_action(observation, deterministic)


# Ejecutar simulación de Frogger 
def run_frogger_simulation(steps=1000, render_mode="human", solution_type="Random", model_path=None):

    # Crear entorno de Frogger
    gym.register_envs(ale_py) # registrar entornos de ALE
    env = gym.make("ALE/Frogger-v5", render_mode=render_mode) # seleccionar entorno de Frogger
    observation, info = env.reset() # reiniciar entorno
    
    score = 0
    # crear agente con la solución proporcionada
    agent = Agent(solution_type, model_path)
    done = False
    won = False
    actions = 0
    # ejecutar simulación
    while not done:
        actions += 1
        # El agente elige una accion basada en la observación del estado actual del juego
        action = agent.choose_action(observation) 
        # print(action)
        # El entorno toma la accion con .step() y devuelve:
        # - la observación del estado actual
        # - la recompensa obtenida
        # - si el juego ha terminado (la rana murio o llego al final)
        # - si el juego ha sido truncado (se ha alcanzado el límite de pasos)
        # - información adicional (metadatos)
        observation, reward, terminated, truncated, info = env.step(action) 
        #print current frog position
        # print(f"{observation}\n\n\n")
        # Si el juego ha terminado o ha sido truncado, reiniciar el entorno
        if terminated or truncated:
            if info["lives"] > 0:
                won = True
            observation, info = env.reset()
            done = True # terminar simulación
            score = reward
            


    env.close()
    return score, won, actions

    # Funcion comparadora de soluciones
def evaluar_soluciones(solutions, iteraciones=1000):
    resultados = {}

    for solution in solutions:
        # Ejecutamos una simulación de Frogger para cada solución
        puntaje_promedio = 0
        veces_ganadas = 0
        puntaje_maximo = 0
        acciones_promedio = 0
        for _ in tqdm(range(iteraciones), desc=f"Evaluando {solution['name']}"):
            puntaje, ganado, acciones = run_frogger_simulation(solution_type=solution["name"], model_path=solution["model_path"], render_mode="rgb_array")
            puntaje_promedio += puntaje
            acciones_promedio += acciones
            if puntaje > puntaje_maximo:
                puntaje_maximo = puntaje
            if ganado:
                veces_ganadas += 1
        
        puntaje_promedio /= iteraciones
        porcentaje_ganado = veces_ganadas / iteraciones
        acciones_promedio /= iteraciones
        resultados[solution["name"]] = {"puntaje_promedio": puntaje_promedio, "puntaje_maximo": puntaje_maximo, "porcentaje_ganado": porcentaje_ganado, "acciones_promedio": acciones_promedio}

    # Guardar resultados en un archivo CSV
    with open('resultados_evaluacion.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Solución", "Puntaje Promedio", "Puntaje Máximo", "Porcentaje Ganado", "Acciones Promedio"])
        for solution, metrics in resultados.items():
            writer.writerow([solution, metrics["puntaje_promedio"], metrics["puntaje_maximo"], metrics["porcentaje_ganado"], metrics["acciones_promedio"]])

    # Graficar resultados
    nombres_soluciones = [solution for solution in resultados.keys()]
    puntajes_promedio = [metrics["puntaje_promedio"] for metrics in resultados.values()]
    puntajes_maximos = [metrics["puntaje_maximo"] for metrics in resultados.values()]
    porcentajes_ganados = [metrics["porcentaje_ganado"] for metrics in resultados.values()]
    acciones_promedio = [metrics["acciones_promedio"] for metrics in resultados.values()]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.bar(nombres_soluciones, puntajes_promedio, color='blue')
    plt.xlabel('Soluciones')
    plt.ylabel('Puntaje Promedio')
    plt.title('Puntaje Promedio por Solución')

    plt.subplot(2, 2, 2)
    plt.bar(nombres_soluciones, puntajes_maximos, color='green')
    plt.xlabel('Soluciones')
    plt.ylabel('Puntaje Máximo')
    plt.title('Puntaje Máximo por Solución')

    plt.subplot(2, 2, 3)
    plt.bar(nombres_soluciones, porcentajes_ganados, color='red')
    plt.xlabel('Soluciones')
    plt.ylabel('Porcentaje Ganado')
    plt.title('Porcentaje Ganado por Solución')

    plt.subplot(2, 2, 4)
    plt.bar(nombres_soluciones, acciones_promedio, color='purple')
    plt.xlabel('Soluciones')
    plt.ylabel('Acciones Promedio')
    plt.title('Acciones Promedio por Solución')

    plt.tight_layout()
    plt.savefig('resultados_evaluacion.png')
    plt.show()


def train_agent(solution_type, num_episodes, action_mapping, file_path="model", config=None, plot=False):
    print(f"Training agent with solution type: {solution_type}") 
    # Crear entorno de Frogger
    gym.register_envs(ale_py) # registrar entornos de ALE
    env = gym.make("ALE/Frogger-v5", obs_type="rgb") # seleccionar entorno de Frogger
    env = gym.wrappers.FrameStackObservation(env, 4) # apilar 4 fotogramas para capturar el movimiento
    # crear agente con la solución proporcionada
    agent = Agent(solution_type, requires_training=True, config=config)

    greedy_count = 0 # contador de acciones greedy
    rewards = [] # lista de recompensas
    avg_rewards = [] # lista de recompensas promedio
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
            agent.model.store_transition(observation, action, reward, observation_, done)
            agent.model.learn()
            observation = observation_
        
        agent.model.update_target()
        rewards.append(total_reward)
        avg = np.mean(rewards[-100:])
        avg_rewards.append(avg)
        print(f"Episode {episode+1}/{num_episodes}: Total Reward: {total_reward}, Average Reward: {avg}")

        # Imprimir resultados del episodio
        print(f"Episode {episode+1}/{num_episodes}: Total Reward: {total_reward}")

    
    env.reset()
    env.close()
    # Save model
    agent.model.save(file_path)

    # Plot learning curve
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(avg_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Learning Curve')
        plt.savefig(f"{file_path}_learning_curve.png")
        # plt.show()

if __name__ == "__main__":
    # run_frogger_simulation(1000,solution_type="DQN",model_path="action_mapping1")
    solutions = [
        { "name": "DQN", "model_path": "agent1"},
        { "name": "Random", "model_path": None}
    ]
    evaluate_solutions(solutions, iterations=100)
