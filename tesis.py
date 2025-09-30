import os
import sys
import time
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# =========================
# Configuración general
# =========================
SEED = 42
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS_STAGE1 = 5     # entrenar solo la cabeza
EPOCHS_STAGE2 = 10    # fine-tuning del backbone
FINE_TUNE_FROM_LAST_N_LAYERS = 120  # descongelar las últimas N capas
USE_LABEL_SMOOTHING = 0.0           # 0.0 o 0.1 si hay ruido

tf.random.set_seed(SEED)

# Opcional: silenciar algunos logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# =========================
# Rutas del dataset (ajusta si cambian)
# =========================
base_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "yudhaislamisulistya" / "plants-type-datasets" / "versions" / "16"
train_dir = base_dir / "split_ttv_dataset_type_of_plants" / "Train_Set_Folder"
val_dir   = base_dir / "split_ttv_dataset_type_of_plants" / "Validation_Set_Folder"
test_dir  = base_dir / "split_ttv_dataset_type_of_plants" / "Test_Set_Folder"

print(f"Base dir: {base_dir}")
print("\n=== Diagnóstico de estructura ===")
print(f"train_dir: {train_dir}")
print(f"val_dir:   {val_dir}")
print(f"test_dir:  {test_dir}")
print("single_root (si aplica): None")
print("=================================\n")

for p in [train_dir, val_dir, test_dir]:
    if not p.exists():
        print(f"[ERROR] No existe la ruta: {p}")
        sys.exit(1)

# =========================
# Carga de datasets (RGB, clases categóricas)
# =========================
def build_ds(root, training=False):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        root,
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=training,
        seed=SEED,
        color_mode="rgb",  # <- CRÍTICO para evitar shape mismatch y poder cargar ImageNet
    )
    return ds

train_ds_raw = build_ds(train_dir, training=True)
val_ds_raw   = build_ds(val_dir, training=False)
test_ds_raw  = build_ds(test_dir, training=False)

class_names = train_ds_raw.class_names
num_classes = len(class_names)

print(f"Clases detectadas ({num_classes}): {class_names}")

# Conteo rápido de muestras
def count_files(root):
    total = 0
    for c in class_names:
        folder = Path(root) / c
        if folder.exists():
            total += sum(1 for _ in folder.glob("*"))
    return total

print(f"Found {count_files(train_dir)} files belonging to {num_classes} classes (train).")
print(f"Found {count_files(val_dir)} files belonging to {num_classes} classes (val).")
print(f"Found {count_files(test_dir)} files belonging to {num_classes} classes (test).")

# =========================
# Preprocesamiento y augmentations
# (NO usar Rescaling si usas preprocess_input)
# =========================
AUTOTUNE = tf.data.AUTOTUNE

data_augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
], name="augment")

def preprocess_batch(x, y):
    # EfficientNet preprocess_input ya incluye normalización adecuada
    x = tf.keras.applications.efficientnet.preprocess_input(tf.cast(x, tf.float32))
    return x, y

def with_aug(ds):
    return ds.map(lambda x, y: (data_augment(x), y), num_parallel_calls=AUTOTUNE)

def with_preproc(ds):
    return ds.map(preprocess_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

train_ds = with_preproc(with_aug(train_ds_raw))
val_ds   = with_preproc(val_ds_raw)
test_ds  = with_preproc(test_ds_raw)

# =========================
# Construcción del modelo
# =========================
def build_model(num_classes):
    # Intentar cargar pesos ImageNet
    base = None
    imagenet_loaded = False
    try:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",        # <- pesos pre-entrenados
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            pooling="avg"
        )
        imagenet_loaded = True
        print("[INFO] EfficientNetB0 con pesos ImageNet cargados correctamente.")
    except Exception as e:
        print("[AVISO] No se pudieron cargar pesos ImageNet. Motivo:")
        print(f"       {e}")
        print("-> Se usará EfficientNetB0 SIN pesos (entrenará desde cero).")
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            pooling="avg"
        )

    # Etapa 1: si hay ImageNet, congelamos; si no hay, entrenamos todo (o casi todo)
    if imagenet_loaded:
        base.trainable = False
    else:
        # Si no hay pesos, conviene entrenar el backbone, pero para CPU puede ser pesado.
        # Opción intermedia: congelar primeras capas y entrenar las últimas.
        for layer in base.layers[:-FINE_TUNE_FROM_LAST_N_LAYERS]:
            layer.trainable = False
        for layer in base.layers[-FINE_TUNE_FROM_LAST_N_LAYERS:]:
            layer.trainable = True

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    # OJO: NO añadir Rescaling porque ya usamos preprocess_input en el pipeline
    x = base(inputs, training=False)  # training=False mantiene BN en inference al principio
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="EffNetB0_PLANTS")

    return model, base, imagenet_loaded

model, base, imagenet_loaded = build_model(num_classes)

# Resumen
model.summary()

# =========================
# Callbacks
# =========================
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)

ckpt_path = "best_effb0.keras"
checkpoint = keras.callbacks.ModelCheckpoint(
    ckpt_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

callbacks = [early_stop, reduce_lr, checkpoint]

# =========================
# Entrenamiento – Etapa 1 (cabeza)
# =========================
print("\n[Etapa 1] Entrenando la cabeza del modelo...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=USE_LABEL_SMOOTHING),
    metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks,
    verbose=1
)

# =========================
# Entrenamiento – Etapa 2 (fine-tuning)
# =========================
print("\n[Etapa 2] Fine-tuning del backbone (últimas capas)...")

# Si cargamos ImageNet, ahora sí descongelamos parte/final del backbone
if imagenet_loaded:
    base.trainable = True
    # Descongelar solo la cola para estabilidad en CPU
    for layer in base.layers[:-FINE_TUNE_FROM_LAST_N_LAYERS]:
        layer.trainable = False
    for layer in base.layers[-FINE_TUNE_FROM_LAST_N_LAYERS:]:
        layer.trainable = True

# LR bajo para FT
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=USE_LABEL_SMOOTHING),
    metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks,
    verbose=1
)

# =========================
# Evaluación
# =========================
print("\n[Evaluación] Cargando mejor checkpoint y evaluando en test...")
best_model = keras.models.load_model(ckpt_path)
test_metrics = best_model.evaluate(test_ds, verbose=1)
metric_names = best_model.metrics_names
print("== Métricas de Test ==")
for name, val in zip(metric_names, test_metrics):
    print(f"{name}: {val:.4f}")

# =========================
# Guardado final
# =========================
export_path = "efficientnetb0_plants_final.keras"
best_model.save(export_path)
print(f"\nModelo guardado en: {export_path}")

# =========================
# Dump de clases (útil para inferencia)
# =========================
with open("class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print("class_names.json escrito con el mapeo de clases.")