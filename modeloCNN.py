import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt

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
VALIDATION_SPLIT = 0.30 # <--- MODIFICACIÓN: 30% para Validación

# --- 2. PREPARACIÓN DE DATOS (Data Augmentation y División Train/Val) ---

print(f"--- 1. PREPARACIÓN DE DATOS (División {1-VALIDATION_SPLIT} Train / {VALIDATION_SPLIT} Val, Modo: RGB) ---")

# Generador de datos con AUMENTO para Entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalización de píxeles
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT # 30% de los datos totales para validación
)

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
validation_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    subset='validation',
    seed=RANDOM_SEED
)

# --- 3. CONSTRUCCIÓN DEL MODELO (EfficientNetB0 Entrenamiento desde Cero) ---

print("\n--- 2. ARQUITECTURA DEL MODELO (Entrenamiento desde Cero) ---")

# Entrenamiento desde CERO (`weights=None`) para evitar errores de compatibilidad
base_model = EfficientNetB0(
    weights=None, 
    include_top=False, 
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], INPUT_CHANNELS)
)

print("\n⚠️ ADVERTENCIA: Entrenando desde CERO (sin pesos de ImageNet) debido a la incompatibilidad.")

base_model.trainable = True # Permitir el entrenamiento de todas las capas

model = Sequential([
    base_model,
    GlobalAveragePooling2D(), 
    Dense(256, activation='relu'), 
    Dropout(0.4), 
    Dense(1, activation='sigmoid') 
], name="Quenual_Classifier")

# Tasa de aprendizaje ajustada para entrenamiento completo
model.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
MODEL_PATH = 'best_model_quenual.keras'
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1), 
    ModelCheckpoint(filepath=MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
]

# --- 4. ENTRENAMIENTO ÚNICO (Desde Cero) ---

print("\n--- 3. INICIANDO ENTRENAMIENTO DESDE CERO (50 ÉPOCAS) ---")
initial_epochs = 20

history = model.fit(
    train_generator,
    epochs=initial_epochs,
    validation_data=validation_generator,
    callbacks=callbacks
)

# --- 5. EVALUACIÓN Y RESULTADO FINAL (Validation Accuracy) ---

print("\n--- 4. EVALUACIÓN FINAL SOBRE CONJUNTO DE VALIDACIÓN ---")

# Cargar el modelo final
try:
    final_model = load_model(MODEL_PATH)
except Exception:
    final_model = model

val_loss, val_accuracy = final_model.evaluate(validation_generator)

print("\n=======================================================")
print(f"| ✅ Precisión Final del Modelo: {val_accuracy*100:.2f}% |")
print(f"| La meta del 90% ahora depende del volumen de datos. |")
print("=======================================================")

# --- 6. VISUALIZACIÓN DE RESULTADOS ---

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss_hist = history.history['val_loss']

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