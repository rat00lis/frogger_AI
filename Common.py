# Funciones comunes para el proyecto

import cv2

# si todo falla sapear esto
def preprocess_observation(observation):
    # Preprocesar la observación
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) # Convertir a escala de grises
    observation = cv2.resize(observation, (84, 110)) # Redimensionar a 84x110
    observation = observation[18:102, :] # Recortar la imagen para eliminar la información de la puntuación
    observation = observation / 255.0 # Normalizar los valores de píxeles
    return observation