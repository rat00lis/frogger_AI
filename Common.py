import cv2
import numpy as np

def preprocess_observation(observation):
    # print(f"Observation shape before preprocessing: {observation.shape}")
    # Preprocesar cada fotograma individualmente
    processed_frames = []
    for frame in observation:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Convertir a escala de grises
        frame = cv2.resize(frame, (84, 110)) # Redimensionar a 84x110
        frame = frame[18:102, :] # Recortar la imagen para eliminar la información de la puntuación
        frame = frame / 255.0 # Normalizar los valores de píxeles
        processed_frames.append(frame)
    
    # Apilar los fotogramas procesados
    processed_observation = np.stack(processed_frames, axis=0)
    return processed_observation