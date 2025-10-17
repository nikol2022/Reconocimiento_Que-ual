import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys

# --- CONFIGURACIÓN ---

# Nombre del archivo del modelo guardado
MODEL_PATH = 'best_model_quenual.keras' 
# Tamaño de la imagen de entrada (debe coincidir con el entrenamiento 224x224 RGB)
IMG_SIZE = (224, 224) 
# RUTA a la imagen que quieres probar. ¡DEBES CAMBIAR ESTA RUTA!
TEST_IMAGE_PATH = r"C:\Nikol\Universidad\DatasetTesis\p8.jpeg" 

# Definición de clases (Debe coincidir con el orden alfabético que usa Keras)
# Keras ordena las carpetas alfabéticamente: NO QUENUAL (0), QUENUAL (1)
CLASS_NAMES = {0: "NO QUEÑUAL", 1: "QUEÑUAL"}

# 1. Cargar el modelo
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: No se encontró el modelo en la ruta: {MODEL_PATH}")
    print("Asegúrate de que 'modeloCNN.py' se haya ejecutado y haya guardado el modelo.")
    sys.exit(1)

try:
    # Carga el modelo que fue guardado por el script de entrenamiento
    model = load_model(MODEL_PATH)
    print(f"Modelo cargado correctamente desde: {MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo. Puede ser un problema de versión de Keras/TensorFlow: {e}")
    sys.exit(1)

# 2. Función de predicción
def predict_single_image(img_path):
    # Cargar la imagen en modo RGB y redimensionar
    img = image.load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
    
    # Convertir a array (H, W, C)
    img_array = image.img_to_array(img)
    
    # Expandir dimensiones para el batch (1, H, W, C)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalizar los píxeles (MISMO PREPROCESAMIENTO que en el entrenamiento)
    img_array /= 255.0
    
    # Realizar la predicción: [0][0] obtiene la probabilidad de la clase 1 (QUEÑUAL)
    prediction_prob = model.predict(img_array, verbose=0)[0][0]
    
    # Decidir la clase usando un umbral de 0.5
    if prediction_prob > 0.85:
        result_class = CLASS_NAMES[1]
        confidence = prediction_prob
    else:
        result_class = CLASS_NAMES[0]
        # Si es NO QUEÑUAL (Clase 0), la confianza es 1 - probabilidad_quenual
        confidence = 1 - prediction_prob
        
    return result_class, confidence, prediction_prob

# 3. Ejecutar la predicción
if os.path.exists(TEST_IMAGE_PATH):
    print(f"\nAnalizando imagen: {os.path.basename(TEST_IMAGE_PATH)}")
    
    predicted_class, confidence, raw_prob = predict_single_image(TEST_IMAGE_PATH)
    
    print("\n=============================================")
    print(f"PREDICCIÓN: {predicted_class}")
    print(f"CONFIANZA DEL MODELO: {confidence*100:.2f}%")
    print(f"Probabilidad de ser QUEÑUAL (Raw): {raw_prob*100:.2f}%")
    print("=============================================")
else:
    print(f"ERROR: No se encontró la imagen en la ruta: {TEST_IMAGE_PATH}")