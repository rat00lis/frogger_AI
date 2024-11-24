import gymnasium as gym
import ale_py
from DQN import AgentDQN
from Random import AgentRandom


# Soluciones actuales
solutions = ["DQN", "Random"]


class Agent:
    # El agente elige una solución basada en el tipo de solución proporcionado
    def __init__(self, solution_type="DQN", model_path=None):
        if solution_type == "DQN": # Si la solución es DQN, se crea un agente DQN
            self.model = AgentDQN(model_path)
        elif solution_type == "Random": # Si la solución es Random, se crea un agente Random
            self.model = AgentRandom()
        else:
            self.model = None

    def choose_action(self, observation):
        return self.model.choose_action(observation)


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


if __name__ == "__main__":
    evaluate_solutions()