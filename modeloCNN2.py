import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns # Importación para visualizar la matriz

# --- 1. CONFIGURACIÓN INICIAL Y PARÁMETROS ---

# La ruta base donde se encuentran las carpetas 'QUENUAL' y 'NO QUENUAL'
BASE_DIR = r"C:\Nikol\Universidad\DatasetTesis"

# Parámetros
IMG_SIZE = (224, 224) 
BATCH_SIZE = 32
RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)

# Número de canales (RGB)
INPUT_CHANNELS = 3 
VALIDATION_SPLIT = 0.30 # 30% para Validación

# --- 2. PREPARACIÓN DE DATOS (Data Augmentation Corregido) ---

print(f"--- 1. PREPARACIÓN DE DATOS (División {1-VALIDATION_SPLIT} Train / {VALIDATION_SPLIT} Val, Modo: RGB) ---")

# Generador de datos con AUMENTO AGRESIVO (Corrección de Overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,     # Aumentado a 30
    width_shift_range=0.3, # Aumentado a 0.3
    height_shift_range=0.3,# Aumentado a 0.3
    shear_range=0.3,
    zoom_range=0.3,        # Aumentado a 0.3
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT 
)

# Generador de validación (solo reescalado, sin augmentation)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=VALIDATION_SPLIT)


# Carga de datos de entrenamiento (70%)
train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    subset='training',
    seed=RANDOM_SEED
)

# Carga de datos de validación (30%)
validation_generator = val_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    subset='validation',
    seed=RANDOM_SEED,
    shuffle=False # Crucial para la Matriz de Confusión
)

# Obtener los nombres de las clases para la matriz
CLASS_LABELS = list(validation_generator.class_indices.keys())


# --- 3. CONSTRUCCIÓN DEL MODELO (Regularización y Entrenabilidad Corregida) ---

print("\n--- 2. ARQUITECTURA DEL MODELO (Entrenamiento desde Cero con Regularización) ---")

base_model = EfficientNetB0(
    weights=None, # Entrenar desde pesos aleatorios.
    include_top=False, 
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], INPUT_CHANNELS)
)

print("\n⚠️ ADVERTENCIA: Entrenando desde CERO. Dropout y LR ajustados para combatir el sobreajuste.")

base_model.trainable = True 

model = Sequential([
    base_model,
    GlobalAveragePooling2D(), 
    Dense(256, activation='relu'), 
    Dropout(0.7), # <--- CORRECCIÓN: Dropout aumentado a 0.7 para regularización
    Dense(1, activation='sigmoid') 
], name="Quenual_Classifier")

# Compilación para la FASE 1 (Entrenamiento completo)
model.compile(
    optimizer=Adam(learning_rate=0.0005), # <--- CORRECCIÓN: Tasa de Aprendizaje reducida a 0.0005
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
MODEL_PATH = 'best_model_quenual2.keras'
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1), 
    ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
]

# --- 4. ENTRENAMIENTO (Fase 1: Entrenamiento Completo) ---

print("\n--- 3. INICIANDO FASE 1: ENTRENAMIENTO COMPLETO (30 ÉPOCAS) ---")
initial_epochs = 30 # Aumentamos las épocas a 30

history = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=validation_generator,
    callbacks=callbacks
)

# --- 5. AJUSTE FINO (Fase 2: Optimización del Aprendizaje) ---

print("\n--- 4. INICIANDO FASE 2: AJUSTE FINO (Optimización con LR bajo) ---")

try:
    model = load_model(MODEL_PATH)
except:
    print("No se pudo cargar el modelo guardado. Continuando con el modelo actual.")
    pass

# Re-compilación con tasa de aprendizaje MUY baja para el ajuste fino
model.compile(
    optimizer=Adam(learning_rate=0.00001), 
    loss='binary_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 20 
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], 
    validation_data=validation_generator,
    callbacks=callbacks
)

# --- 6. EVALUACIÓN Y ANÁLISIS DETALLADO (Falsos Positivos y Matriz) ---

print("\n--- 5. EVALUACIÓN Y ANÁLISIS DETALLADO SOBRE CONJUNTO DE VALIDACIÓN ---")

# Cargar el modelo final (el mejor)
try:
    final_model = load_model(MODEL_PATH)
except:
    final_model = model

# 1. Obtener Predicciones
validation_generator.reset()
predictions = final_model.predict(validation_generator, verbose=0)
predicted_classes = (predictions > 0.5).astype(int)
true_classes = validation_generator.classes 

# 2. Generar y Visualizar la Matriz de Confusión
cm = confusion_matrix(true_classes, predicted_classes)
print("\nMatriz de Confusión (Valores Brutos):\n", cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
plt.xlabel('Predicción')
plt.ylabel('Valor Verdadero')
plt.title('Matriz de Confusión')
plt.show() # Mostrar el gráfico de la matriz


# 3. Reporte de Clasificación (Incluye métricas como Precision, Recall y F1-Score)
report = classification_report(true_classes, predicted_classes, target_names=CLASS_LABELS)
print("\nReporte de Clasificación Detallado:\n", report)

# Evaluación de la precisión global y Falsos Positivos
val_loss, val_accuracy = final_model.evaluate(validation_generator, verbose=0)
fp_count = cm[0, 1] 

print("\n=======================================================")
print(f"| ✅ Precisión Global (Accuracy): {val_accuracy*100:.2f}% |")
print(f"| 🚨 Falsos Positivos (FP): {fp_count} imágenes         |")
print("=======================================================")

# --- 7. VISUALIZACIÓN DE RESULTADOS ---

# Combinar historial de ambas fases
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss_hist = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Precisión de Entrenamiento')
plt.plot(val_acc, label='Precisión de Validación')
plt.axhline(y=0.90, color='r', linestyle='--', label='Meta 90%')
plt.legend(loc='lower right')
plt.title('Precisión (Training vs Validation)')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Pérdida de Entrenamiento')
plt.plot(val_loss_hist, label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida (Training vs Validation)')
plt.show()