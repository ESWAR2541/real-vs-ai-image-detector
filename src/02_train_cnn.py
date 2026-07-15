"""
STEP 2: BUILD AND TRAIN THE CNN
===================================
Architecture follows the original CIFAKE paper (Bird & Lotfi, 2023):
two convolutional layers (32 filters each) followed by two fully
connected layers - their published result with this exact design was
92.93% accuracy. We're using a smaller subset of data than the full
paper, so expect somewhat lower accuracy - that's noted honestly in
the evaluation output, not hidden.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------
# 1. BUILD THE CNN
# ---------------------------------------------------------------
# Conv2D layers learn visual patterns (edges, textures, artifacts) -
# early layers learn simple patterns, later layers combine them into
# more complex features. MaxPooling reduces the image size after each
# conv layer, keeping the most important signal while cutting compute.
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),  # turn the 2D feature maps into a 1D vector
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),  # randomly disables 30% of neurons during
                           # training to reduce overfitting on our
                           # relatively small subset of data
    layers.Dense(1, activation="sigmoid")  # single output: probability
                                             # of being class 1
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------------------------------------------
# 2. TRAIN
# ---------------------------------------------------------------
# EarlyStopping stops training automatically if validation loss stops
# improving, instead of training a fixed number of epochs regardless -
# this helps avoid overfitting and wastes less time.
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=20,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ---------------------------------------------------------------
# 3. EVALUATE ON THE TEST SET
# ---------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
print("(rows = actual, columns = predicted, order = [FAKE, REAL])")

# ---------------------------------------------------------------
# 4. SAVE THE TRAINED MODEL
# ---------------------------------------------------------------
model.save("outputs/cifake_detector.keras")
print("\nSaved trained model -> outputs/cifake_detector.keras")

with open("outputs/model_evaluation.txt", "w") as f:
    f.write("REAL VS AI-GENERATED IMAGE DETECTOR - EVALUATION\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n\n")
    f.write(f"Test accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test loss: {test_loss:.4f}\n\n")
    f.write(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
print("Saved outputs/model_evaluation.txt")
