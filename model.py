import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# 1. Load datasets
train_df = pd.read_csv("train_dataset.csv")
test_df  = pd.read_csv("test_dataset.csv")


image_folder = "images_resized"


# 2. Prepare training data
X_train, y_train = [], []


for index, row in train_df.iterrows():
    img_path = os.path.join(image_folder, row["Image Name"])
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)   # EfficientNet preprocessing
    X_train.append(img_array)
    y_train.append(row["label_numeric"])


X_train = np.array(X_train)
y_train = np.array(y_train)


# 3. Shuffle training data
indices = np.arange(len(X_train))
np.random.seed(42)
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

print("Training dataset shape:", X_train.shape)


# 4. Prepare test data
X_test, y_test = [], []


for index, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row["Image Name"])
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    X_test.append(img_array)
    y_test.append(row["label_numeric"])


X_test = np.array(X_test)
y_test = np.array(y_test)


print("Testing dataset shape:", X_test.shape)


# 5. Build model with EfficientNet80


# Load EfficientNetB0 without top classification layers
# include_top=False means we add our own output layer
# Weights from ImageNet give us a strong starting point
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


# Freeze base model weights
# This prevents overwriting pretrained features during early training
base_model.trainable = False


# Build the full model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# 6. Define callbacks


# Stop training if val_loss does not improve for 5 epochs
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


# Reduce learning rate if val_loss stops improving for 3 epochs
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)


# 7. Train model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    shuffle=True    # Keras also shuffles each epoch during fit
)


# 8. Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("\nTest Loss:    ", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))


# 9. Save model
model.save("fixed_model.keras")
print("\nModel saved successfully as fixed_model.keras")
