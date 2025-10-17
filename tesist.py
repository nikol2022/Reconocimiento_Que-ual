import os
import sys
import json
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Recall 

# =========================
# CONFIGURACIÓN OPTIMIZADA
# =========================
SEED = 42
IMG_SIZE = (224, 224)
BATCH = 32
# AUMENTADO: Mucho más tiempo, crucial si ImageNet no carga y se entrena desde cero.
EPOCHS_STAGE1 = 50 
EPOCHS_STAGE2 = 50 
# Mantenido
FINE_TUNE_FROM_LAST_N_LAYERS = 40
USE_LABEL_SMOOTHING = 0.1
SPLIT_RATIOS = [0.7, 0.15, 0.15]

tf.random.set_seed(SEED)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# =========================
# RUTAS
# =========================
DATASET_PATH = Path(r"C:\Nikol\Universidad\DatasetTesis")

if not DATASET_PATH.exists():
    print(f"[ERROR] No existe la ruta del dataset: {DATASET_PATH}")
    sys.exit(1)

# Detectar carpetas de clases (solo Quenual y NoQuenual)
class_folders = [f for f in os.listdir(DATASET_PATH)
                 if os.path.isdir(DATASET_PATH / f) and f in ["Quenual", "NoQuenual"]]

if not class_folders:
    print("[ERROR] No se encontraron subcarpetas 'Quenual' o 'NoQuenual'.")
    sys.exit(1)
    
# =========================
# DIVISIÓN AUTOMÁTICA DEL DATASET
# =========================
print(f"\n=== División automática del dataset desde {DATASET_PATH} ===")

split_dirs = ["train", "val", "test"]
for split in split_dirs:
    for cls in class_folders:
        os.makedirs(DATASET_PATH / split / cls, exist_ok=True)

split_files = {}
total_files = 0
train_labels = [] # Lista para calcular pesos de clase

for cls in class_folders:
    class_path = DATASET_PATH / cls
    files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not files:
        print(f"[ADVERTENCIA] No se encontraron archivos en {cls}")
        continue
    
    total_files += len(files)

    # División aleatoria
    train_files, temp_files = train_test_split(files, test_size=1 - SPLIT_RATIOS[0], random_state=SEED)
    val_files, test_files = train_test_split(temp_files,
                                             test_size=SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]),
                                             random_state=SEED)

    split_files[cls] = {"train": train_files, "val": val_files, "test": test_files}
    train_labels.extend([cls] * len(train_files)) 

    for split, files_list in split_files[cls].items():
        for f in files_list:
            src = class_path / f
            dst = DATASET_PATH / split / cls / f
            if src.exists() and not dst.exists(): 
                shutil.move(src, dst)

    print(f"Dividido para {cls}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

print(f"✅ División completada. Total de imágenes: {total_files}")

# =========================
# CARGA DE DATOS (MOVIDA ANTES DE PESOS)
# =========================
def build_ds(root, training=False):
    return tf.keras.preprocessing.image_dataset_from_directory(
        root,
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=training,
        seed=SEED,
        color_mode="rgb"
    )

train_ds_raw = build_ds(DATASET_PATH / "train", training=True)
val_ds_raw = build_ds(DATASET_PATH / "val", training=False)
test_ds_raw = build_ds(DATASET_PATH / "test", training=False)

# DEFINICIÓN CRÍTICA DE class_names
class_names = train_ds_raw.class_names
num_classes = len(class_names)

print(f"Clases detectadas ({num_classes}): {class_names}")


# =========================
# CÁLCULO DE PONDERACIÓN DE CLASES (Ahora puede usar class_names)
# =========================
unique_labels = np.unique(train_labels)
weights = class_weight.compute_class_weight(
    'balanced',
    classes=unique_labels,
    y=train_labels
)
# Mapeamos los pesos a los nombres de las clases (cadenas de texto)
class_weight_dict = dict(zip(unique_labels, weights))
# Mapeamos a índices para Keras (USANDO class_names)
class_weights = {class_names.index(name): weight for name, weight in class_weight_dict.items()}

print("\n=== Ponderación de Clases ===")
print(class_weight_dict)


# =========================
# PREPROCESAMIENTO Y AUGMENTATION (Agresivo)
# =========================
AUTOTUNE = tf.data.AUTOTUNE

data_augment = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),                   
    layers.RandomZoom(0.2),                       
    layers.RandomContrast(0.3),                   
], name="augment")

def preprocess_batch(x, y):
    x = tf.keras.applications.efficientnet.preprocess_input(tf.cast(x, tf.float32))
    return x, y

def with_aug(ds):
    return ds.map(lambda x, y: (data_augment(x), y), num_parallel_calls=AUTOTUNE)

def with_preproc(ds):
    return ds.map(preprocess_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

train_ds = with_preproc(with_aug(train_ds_raw))
val_ds = with_preproc(val_ds_raw)
test_ds = with_preproc(test_ds_raw)

# =========================
# CONSTRUCCIÓN DEL MODELO
# =========================
def build_model(num_classes):
    try:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet",
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), pooling="avg"
        )
        imagenet_loaded = True
        print("[INFO] EfficientNetB0 con pesos ImageNet cargados correctamente.")
    except Exception as e:
        print("[AVISO] No se pudieron cargar pesos ImageNet. Motivo:")
        print(f"       {e}")
        print("-> Se usará EfficientNetB0 SIN pesos.")
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights=None,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), pooling="avg"
        )
        imagenet_loaded = False

    base.trainable = False
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(inputs, training=False)
    
    # Regularización L2 y Dropout fuerte
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(x) 
    x = layers.Dropout(0.6)(x) 
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="EffNetB0_PLANTS")

    return model, base, imagenet_loaded

model, base, imagenet_loaded = build_model(num_classes)
model.summary()

# =========================
# CALLBACKS
# =========================
# Paciencia aumentada
early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1) 
ckpt_path = "best_effb0.keras"
checkpoint = keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1)
callbacks = [early_stop, reduce_lr, checkpoint]

# =========================
# ENTRENAMIENTO – ETAPA 1
# =========================
print("\n[Etapa 1] Entrenando la cabeza del modelo...")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=USE_LABEL_SMOOTHING), 
              metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")])

history1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1, 
                     callbacks=callbacks, class_weight=class_weights)

# =========================
# ENTRENAMIENTO – ETAPA 2
# =========================
print("\n[Etapa 2] Fine-tuning del backbone...")
base.trainable = True
for layer in base.layers[:-FINE_TUNE_FROM_LAST_N_LAYERS]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=keras.losses.CategoricalCrossentropy(label_smoothing=USE_LABEL_SMOOTHING), 
              metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")])

history2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE2, 
                     callbacks=callbacks, class_weight=class_weights)


# =========================
# EVALUACIÓN DETALLADA
# =========================
print("\n[Evaluación] Cargando mejor checkpoint y evaluando en test...")
best_model = keras.models.load_model(ckpt_path)
test_metrics = best_model.evaluate(test_ds)
print("\n== Métricas de Test ==")
for name, val in zip(best_model.metrics_names, test_metrics):
    print(f"{name}: {val:.4f}")

# --- Cálculo de Sensibilidad (Recall) Específico para Queñual ---
# 1. Obtener predicciones y etiquetas reales
y_pred = np.concatenate([np.argmax(best_model.predict(x), axis=-1) for x, y in test_ds], axis=0)
y_true = np.concatenate([np.argmax(y.numpy(), axis=-1) for x, y in test_ds], axis=0)

# 2. Encontrar el índice de 'Queñual'
try:
    quenual_index = class_names.index("Quenual")
except ValueError:
    print("[ERROR] No se pudo encontrar la clase 'Quenual'.")
    sys.exit(1)

# 3. Calcular el Recall (Sensibilidad) para la clase 'Queñual'
tp = np.sum((y_pred == quenual_index) & (y_true == quenual_index))
fn = np.sum((y_pred != quenual_index) & (y_true == quenual_index))

if (tp + fn) > 0:
    recall_quenual = tp / (tp + fn)
    print(f"\n✅ Sensibilidad (Recall) para Queñual: {recall_quenual:.4f} ({recall_quenual*100:.2f}%)")
else:
    print("\n[ADVERTENCIA] No hay ejemplos de Queñual en el conjunto de prueba para calcular el Recall.")


# =========================
# GUARDADO FINAL
# =========================
best_model.save("efficientnetb0_plants_final.keras")
print("\nModelo guardado en: efficientnetb0_plants_final.keras")

with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print("✅ class_names.json escrito con el mapeo de clases.")