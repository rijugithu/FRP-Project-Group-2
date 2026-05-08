"""
train_model.py  –  Improved Face Mask Detector Training
========================================================
Features:
1. Proper train/validation split
2. Validation data WITHOUT augmentation
3. Two-phase transfer learning
4. Class balancing
5. Label smoothing
6. TensorBoard support
7. Automatic best-model saving  (best_model.keras)
8. Better MobileNetV2 fine-tuning

Bug Fixes applied:
  FIX 1 – Checkpoint saves as .keras (not .h5) to match Face_Mask_Detection.py
  FIX 2 – make_callbacks() factory creates FRESH callback instances per phase
           so EarlyStopping state from Phase 1 never kills Phase 2 early
  FIX 3 – Removed unused `import tensorflow as tf`
"""

import os
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard,
)

from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight


# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
DATASET_PATH    = "dataset"

IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
SEED            = 42

# Phase-1 (Head Training)
PHASE1_EPOCHS   = 10
PHASE1_LR       = 1e-3

# Phase-2 (Fine-Tuning)
PHASE2_EPOCHS   = 30
PHASE2_LR       = 1e-4
FINE_TUNE_FROM  = -60           # unfreeze last 60 MobileNetV2 layers

LABEL_SMOOTHING = 0.1
DROPOUT_RATE    = 0.5


# ══════════════════════════════════════════════════════════════
# DATA GENERATORS
# ══════════════════════════════════════════════════════════════
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.20,
    rotation_range=25,
    zoom_range=0.20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.75, 1.25],
    channel_shift_range=30.0,
    fill_mode="nearest",
)

# Validation generator WITHOUT augmentation
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.20,
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    seed=SEED,
    shuffle=True,
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    seed=SEED,
    shuffle=False,
)

NUM_CLASSES = train_data.num_classes

print(f"\n✅ Classes found ({NUM_CLASSES}):")
print(train_data.class_indices)
print(f"\n📊 Train Samples : {train_data.samples}")
print(f"📊 Val   Samples : {val_data.samples}")


# ══════════════════════════════════════════════════════════════
# CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_data.classes,
)
class_weight_dict = dict(enumerate(class_weights_array))

print("\n📊 Class Weights:")
for class_name, idx in train_data.class_indices.items():
    print(f"   {class_name}: {class_weight_dict[idx]:.3f}")


# ══════════════════════════════════════════════════════════════
# BUILD MODEL
# ══════════════════════════════════════════════════════════════
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3),
)

# Freeze base model initially (Phase 1 only trains the head)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(DROPOUT_RATE / 2)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.summary()


# ══════════════════════════════════════════════════════════════
# LOSS FUNCTION  (label smoothing prevents overconfidence)
# ══════════════════════════════════════════════════════════════
loss_fn = CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)


# ══════════════════════════════════════════════════════════════
# CALLBACKS FACTORY
# ══════════════════════════════════════════════════════════════
os.makedirs("logs", exist_ok=True)

print("\n📂 Working Directory :", os.getcwd())
print("📂 Best model will be saved at:", os.path.abspath("best_model.keras"))

# ── FIX 2 ────────────────────────────────────────────────────
# Always call make_callbacks() to get FRESH instances for each
# phase. EarlyStopping stores its internal best-score as state.
# If you reuse the same object across phases it enters Phase 2
# already knowing the Phase 1 best score and stops at epoch 1.
def make_callbacks():
    return [
        # FIX 1: filepath is .keras (not .h5) — matches load_model()
        # call in Face_Mask_Detection.py
        ModelCheckpoint(
            filepath="best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(
            log_dir="logs",
            histogram_freq=1,
            write_graph=True,
        ),
    ]


# ══════════════════════════════════════════════════════════════
# PHASE 1 – TRAIN HEAD ONLY  (base frozen)
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("PHASE 1 – Warm-up Training  (head only, base frozen)")
print("═" * 60)

model.compile(
    optimizer=Adam(learning_rate=PHASE1_LR),
    loss=loss_fn,
    metrics=["accuracy"],
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=PHASE1_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=make_callbacks(),     # fresh instances
)


# ══════════════════════════════════════════════════════════════
# PHASE 2 – FINE-TUNE TOP LAYERS
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print(f"PHASE 2 – Fine-Tuning Last {abs(FINE_TUNE_FROM)} Layers")
print("═" * 60)

base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_FROM]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=PHASE2_LR),    # 10x lower LR
    loss=loss_fn,
    metrics=["accuracy"],
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=PHASE2_EPOCHS,
    class_weight=class_weight_dict,
    callbacks=make_callbacks(),     # fresh instances — not reused from Phase 1
)


# ══════════════════════════════════════════════════════════════
# SAVE FINAL MODEL
# ══════════════════════════════════════════════════════════════
model.save("mask_detector.keras")

print("\n✅ TRAINING COMPLETE!")
print("📁 Best Model  : best_model.keras  ← use this in Face_Mask_Detection.py")
print("📁 Final Model : mask_detector.keras")
print("\n💡 To launch TensorBoard:")
print("   tensorboard --logdir logs")
