from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
import os # Asegúrate de importar os si lo necesitas para rutas

# --- 1. CONFIGURACIÓN DEL MODELO ---
# Asegúrate de que esta ruta sea correcta en tu servidor
MODEL_PATH = 'best_model_quenual.keras' 
IMG_SIZE = (224, 224) 
app = FastAPI()
model = None

@app.on_event("startup")
async def load_model_on_startup():
    global model
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Modelo {MODEL_PATH} cargado correctamente.")
    except Exception as e:
        print(f"❌ ERROR: No se pudo cargar el modelo Keras. Verifica la ruta. {e}")
        raise HTTPException(status_code=500, detail="El modelo de IA no está disponible en el servidor.")

# --- 2. ENDPOINT DE RECONOCIMIENTO (LÓGICA CORREGIDA) ---

@app.post("/recognize")
async def recognize_quenual(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="El modelo no se ha cargado.")
        
    try:
        # 1. Leer la imagen directamente en un objeto IO.BytesIO
        contents = await file.read()
        img_bytes = io.BytesIO(contents)
        
        # 2. Cargar y redimensionar la imagen a 224x224 (Pillow/Keras utils)
        # Esto es CRÍTICO para que el modelo funcione.
        img = image.load_img(img_bytes, target_size=IMG_SIZE)
        
        # 3. Convertir imagen a array de NumPy
        img_array = image.img_to_array(img)
        
        # 4. Expandir dimensiones: De (224, 224, 3) a (1, 224, 224, 3) para el batch
        img_array = np.expand_dims(img_array, axis=0)
        
        # 5. Normalización (CRÍTICO: Debe ser 1./255 como en tu entrenamiento)
        img_array /= 255.0 

        # 6. Ejecutar la predicción
        prediction = model.predict(img_array)
        
        # 7. Procesar la salida binaria (Sigmoid)
        quenual_confidence = float(prediction[0][0])
        no_quenual_confidence = 1.0 - quenual_confidence
        
        predicted_class = "QUEÑUAL" if quenual_confidence > 0.5 else "OTRO ÁRBOL"

        # 8. Devolver resultados
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "confidence_quenual": round(quenual_confidence * 100, 2),
            "confidence_no_quenual": round(no_quenual_confidence * 100, 2)
        })

    except Exception as e:
        print(f"❌ Error durante la inferencia o preprocesamiento: {e}")
        # Puedes devolver el error exacto para depuración
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {e}")