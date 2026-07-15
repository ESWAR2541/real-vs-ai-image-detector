"""
STEP 3: TEST THE MODEL ON YOUR OWN IMAGE
============================================
Loads the trained model and classifies any image you point it at.

USAGE:
    python src/03_predict_own_image.py path/to/your/image.jpg
"""

import sys
import numpy as np
from tensorflow import keras
from PIL import Image

if len(sys.argv) < 2:
    print("Usage: python src/03_predict_own_image.py path/to/image.jpg")
    sys.exit(1)

image_path = sys.argv[1]

model = keras.models.load_model("outputs/cifake_detector.keras")

img = Image.open(image_path).convert("RGB").resize((32, 32))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

print(f"\nImage: {image_path}")
print(f"Raw model output: {prediction:.4f}")
print("(Label mapping confirmed from dataset: 0 = FAKE, 1 = REAL)")

if prediction > 0.5:
    print(f"-> Prediction: REAL (confidence: {prediction:.1%})")
else:
    print(f"-> Prediction: FAKE / AI-generated (confidence: {1-prediction:.1%})")

print("\nIMPORTANT LIMITATION: this model was trained on CIFAKE, which uses")
print("Stable Diffusion 1.4 (a 2022-era model) for its fake images and")
print("32x32 low-resolution CIFAR-10 photos for its real images. It will")
print("likely NOT generalize well to modern AI generators (Midjourney,")
print("DALL-E 3, Flux) or high-resolution real photos - this is a known,")
print("published limitation of CIFAKE-trained detectors, not unique to")
print("this implementation. Worth mentioning honestly if asked.")
