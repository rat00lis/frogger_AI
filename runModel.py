import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Common import preprocess_observation
from mainNeuralNetwork import GeneticAgent

def visualize_agent(agent, steps=1000):
    gym.register_envs(ale_py)
    env = gym.make("ALE/Frogger-v5", render_mode="human")
    env = gym.wrappers.FrameStackObservation(env, 1)

    state, _ = env.reset()

    for _ in range(steps):
        action = agent.choose_action(state, deterministic=True)
        next_state, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

        state = next_state

#load model
def load_model(path):
    model = GeneticAgent(input_channels=1, output_size=5)
    model.load(path)
    return model

if __name__ == "__main__":
    agent = load_model("best_agent_last_gen.pt")
    visualize_agent(agent)
