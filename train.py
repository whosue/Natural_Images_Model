#Nathan Hu
#CECS 456 -01
#12/16/2025

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# Setup
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "natural_images"
assert DATA_DIR.exists(), f"Dataset not found at {DATA_DIR}"

#Training Setup
# Image size
IMG_SIZE = (224, 224)

# Images per batch
BATCH_SIZE = 32
SEED = 42
# Maximum training epochs
EPOCHS = 20

#Load the Data
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
# Load validation images
val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print("Classes:", class_names)


AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000, seed=SEED).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# Data Augmentation
# Apply random transformations to training images to improve generalization
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])

# Load model
base_model = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the base model so its weights are not updated during training
base_model.trainable = False

# Define the full model
inputs = keras.Input(shape=(*IMG_SIZE, 3))

# data augmentation applied 
x = data_augmentation(inputs)

# Preprocess input images
x = keras.applications.mobilenet_v2.preprocess_input(x)

# Pass images through the pre-trained network
x = base_model(x, training=False)

# Reduce spacial dimensions
x = layers.GlobalAveragePooling2D()(x)

# Reduce dropout
x = layers.Dropout(0.3)(x)

# Final classification layer
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

# Create the model
model = keras.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Print model summary
model.summary()

# Save the best model during training
callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.3),
    keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
]

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save trained model
model.save("final_model.keras")
print("Training complete. Model saved.")


# Show the images with labels
import matplotlib.pyplot as plt
import numpy as np

for images, labels in val_ds.take(1):
    preds = model.predict(images)
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(12, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        true_name = class_names[int(labels[i])]
        pred_name = class_names[int(pred_labels[i])]
        plt.title(f"T: {true_name}\nP: {pred_name}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
