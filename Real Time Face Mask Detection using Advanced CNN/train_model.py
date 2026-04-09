import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from sklearn.utils.class_weight import compute_class_weight
import json
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH   = "Train"
IMAGE_SIZE     = (224, 224)
BATCH_SIZE     = 32
EPOCHS_FROZEN  = 20        # Phase 1: train head only
EPOCHS_FINETUNE = 30       # Phase 2: fine-tune top layers
INIT_LR        = 1e-4
FINETUNE_LR    = 1e-5
UNFREEZE_LAYERS = 30       # how many layers from the end to unfreeze in Phase 2

# ─────────────────────────────────────────────
# DATA GENERATORS
# MobileNetV2's preprocess_input scales to [-1, 1]
# Use it instead of plain rescale=1./255
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42
)

# ─────────────────────────────────────────────
# SAVE LABEL MAPPING
# ─────────────────────────────────────────────
class_indices = train_data.class_indices
with open("labels.json", "w") as f:
    json.dump(class_indices, f, indent=2)

print("Class indices:", class_indices)
num_classes = len(class_indices)

# ─────────────────────────────────────────────
# CLASS WEIGHTS (helps with imbalanced data)
# ─────────────────────────────────────────────
labels_array = train_data.classes
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels_array),
    y=labels_array
)
class_weight_dict = dict(enumerate(class_weights_array))
print("Class weights:", class_weight_dict)

# ─────────────────────────────────────────────
# MODEL — MobileNetV2 + custom head
# ─────────────────────────────────────────────
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False   # Phase 1: freeze entire base

x = base_model.output
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ─────────────────────────────────────────────
# CALLBACKS (used in both phases)
# ─────────────────────────────────────────────
def get_callbacks(phase_name):
    return [
        ModelCheckpoint(
            f"mask_detector_{phase_name}.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]

# ─────────────────────────────────────────────
# PHASE 1: Train head with frozen base
# ─────────────────────────────────────────────
print("\n=== PHASE 1: Training head (base frozen) ===\n")
model.compile(
    optimizer=Adam(learning_rate=INIT_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FROZEN,
    class_weight=class_weight_dict,
    callbacks=get_callbacks("phase1")
)

# ─────────────────────────────────────────────
# PHASE 2: Fine-tune — unfreeze top layers
# ─────────────────────────────────────────────
print(f"\n=== PHASE 2: Fine-tuning (unfreezing last {UNFREEZE_LAYERS} layers) ===\n")

base_model.trainable = True
for layer in base_model.layers[:-UNFREEZE_LAYERS]:
    layer.trainable = False

# Lower LR to avoid destroying pre-trained weights
model.compile(
    optimizer=Adam(learning_rate=FINETUNE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weight_dict,
    callbacks=get_callbacks("phase2")
)

# ─────────────────────────────────────────────
# SAVE FINAL MODEL
# ─────────────────────────────────────────────
model.save("mask_detector.keras")
print("\n✅ Final model saved as mask_detector.keras")
print(f"✅ Labels saved as labels.json: {class_indices}")
