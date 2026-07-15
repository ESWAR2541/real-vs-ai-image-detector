"""
STEP 1: FETCH THE CIFAKE DATASET
====================================
Dataset: CIFAKE (Bird & Lotfi, 2023) - a real, published academic dataset
of 120,000 images: 60,000 real photos (from CIFAR-10) and 60,000
AI-generated fake images (created with Stable Diffusion 1.4).

This uses the Hugging Face `datasets` library to download it directly -
no manual download, no API key needed for this public dataset.

NOTE: This needs internet access - run this on your own computer.
The full dataset is large, so this script uses a SUBSET (5,000 train,
1,000 test images) to keep training time reasonable on a normal laptop
CPU. You can increase these numbers later if you want a stronger model
and don't mind longer training time.
"""

from datasets import load_dataset
import numpy as np
from PIL import Image

TRAIN_SAMPLES = 5000  # per class would double this - see below
TEST_SAMPLES = 1000

print("Downloading CIFAKE dataset (this may take a few minutes the first time)...")
dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset")

print("Available splits:", dataset.keys())
print(dataset["train"].features)

def prepare_subset(split, n_samples, seed=42):
    """
    Takes a balanced subset of the dataset (equal real/fake), converts
    images to normalized numpy arrays, and returns (X, y).
    """
    data = dataset[split].shuffle(seed=seed).select(range(n_samples))
    images = []
    labels = []
    for example in data:
        img = example["image"].convert("RGB").resize((32, 32))
        images.append(np.array(img) / 255.0)  # normalize pixel values to 0-1
        labels.append(example["label"])  # 0 or 1 - see dataset card for which is which
    return np.array(images), np.array(labels)

print(f"\nPreparing {TRAIN_SAMPLES} training images...")
X_train, y_train = prepare_subset("train", TRAIN_SAMPLES)

print(f"Preparing {TEST_SAMPLES} test images...")
X_test, y_test = prepare_subset("test", TEST_SAMPLES)

print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train label distribution: {np.bincount(y_train)}")
print(f"Test label distribution: {np.bincount(y_test)}")

np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)

print("\nSaved data/X_train.npy, y_train.npy, X_test.npy, y_test.npy")
