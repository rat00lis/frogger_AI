import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
from DQN import AgentDQN
from Random import AgentRandom

# Soluciones actuales
solutions = ["DQN", "Random"]

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
    action_mapping1 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    action_mapping2 = {0:0, 1:1, 2:4, 3:3, 4:2}

    config1 = {
        "learning_rate": 0.1,
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "batch_size": 32,
        "n_actions": 5,
        "num_channels": 4,
        "input_dimension_x": 84,
        "input_dimension_y": 84,
        "fc1_dimension": 512,
        "fc2_dimension": 512,
        "memory_size": 1000,
        "device": None,
    }

    config2 = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 0.5,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.9995,
        "batch_size": 32,
        "n_actions": 5,
        "num_channels": 4,
        "input_dimension_x": 84,
        "input_dimension_y": 84,
        "fc1_dimension": 512,
        "fc2_dimension": 512,
        "memory_size": 1000,
        "device": None,
    }

    train_agent("DQN", 10, action_mapping1, "action_mapping1", config1, plot=True)
    print("trained agent 1")
    train_agent("DQN", 10, action_mapping2, "action_mapping2", config2, plot=True)
    print("trained agent 2")    