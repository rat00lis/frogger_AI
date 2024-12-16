import random

class AgentRandom:
    def choose_action(self, observation, deterministic=False):
        return random.choice([0, 1, 2, 3, 4])  # Replace with valid actions for Frogger

