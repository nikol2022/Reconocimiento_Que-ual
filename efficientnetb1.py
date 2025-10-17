import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# =========================
# Configuración general
# =========================
SEED = 42
IMG_SIZE = (240, 240)  # Tamaño de imagen
BATCH = 32
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 20
FINE_TUNE_FROM_LAST_N_LAYERS = 120

tf.random.set_seed(SEED)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# =========================
# Rutas del dataset
# =========================
base_dir = Path.home() / ".cache" / "kagglehub" / "datasets" / "yudhaislamisulistya" / "plants-type-datasets" / "versions" / "16"
train_dir = base_dir / "split_ttv_dataset_type_of_plants" / "Train_Set_Folder"
val_dir   = base_dir / "split_ttv_dataset_type_of_plants" / "Validation_Set_Folder"
test_dir  = base_dir / "split_ttv_dataset_type_of_plants" / "Test_Set_Folder"
# =========================
# Carga de datasets
# =========================
def build_ds(root, training=False):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        root,
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH,
        shuffle=training,
        seed=SEED,
        color_mode="grayscale",  # <- 1 canal
    )
    return ds

train_ds_raw = build_ds(train_dir, training=True)
val_ds_raw   = build_ds(val_dir, training=False)
test_ds_raw  = build_ds(test_dir, training=False)

num_classes = len(train_ds_raw.class_names)

# =========================
# Preprocesamiento
# =========================
AUTOTUNE = tf.data.AUTOTUNE

def preprocess_batch(x, y):
    x = tf.keras.applications.efficientnet.preprocess_input(tf.cast(x, tf.float32))
    return x, y

train_ds = train_ds_raw.map(preprocess_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds   = val_ds_raw.map(preprocess_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
test_ds  = test_ds_raw.map(preprocess_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# =========================
# Construcción del modelo
# =========================
def build_model(num_classes):
    base = tf.keras.applications.EfficientNetB1(
        include_top=False,
        weights=None,  # <- entrenar desde cero
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1),  # 1 canal
        pooling="avg"
    )
    base.trainable = True  # puedes hacer fine-tuning después

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
    x = base(inputs, training=True)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="EffNetB1_Gray")
    return model, base

model, base = build_model(num_classes)
model.summary()

# =========================
# Callbacks
# =========================
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3
)

ckpt_path = "best_effb1_gray.keras"
checkpoint = keras.callbacks.ModelCheckpoint(
    ckpt_path,
    monitor="val_accuracy",
    save_best_only=True
)

callbacks = [early_stop, reduce_lr, checkpoint]

# =========================
# Entrenamiento
# =========================
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks
)

# =========================
# Evaluación
# =========================
best_model = keras.models.load_model(ckpt_path)
test_metrics = best_model.evaluate(test_ds)
print(f"Accuracy en Test: {test_metrics[1]*100:.2f}%")
