# Real vs. AI-Generated Image Detector

A convolutional neural network (CNN) that classifies images as real
photographs or AI-generated synthetic images, trained on CIFAKE — a
real, published academic dataset.

## Results (actual run)

- **Test accuracy: 85.4%** on 1,000 held-out test images (5,000 training
  images used)
- Precision/recall balanced across both classes (FAKE: 0.85 precision/
  0.85 recall; REAL: 0.85 precision/0.86 recall)
- This is a strong result — close to the original paper's 92.93% despite
  using roughly 1/24th of their training data (5,000 vs. 120,000 images)

## Data Source

- **CIFAKE dataset** (Bird & Lotfi, 2023) — 120,000 images: 60,000 real
  photos from CIFAR-10, and 60,000 AI-generated images created with
  Stable Diffusion 1.4
- Downloaded via Hugging Face: `dragonintelligence/CIFAKE-image-dataset`
- This project trains on a **5,000/1,000 subset** (train/test) to keep
  training time reasonable on a standard laptop CPU — the full paper
  used all 120,000 images and reported 92.93% accuracy with the same
  architecture used here

## Why This Project

Every other project in my portfolio uses tabular data. This one uses
**computer vision** — a genuinely different skill set (convolutional
neural networks, image preprocessing, TensorFlow/Keras), and directly
demonstrates the deep learning frameworks listed on my resume rather
than leaving them unverified.

## Architecture

Follows the CNN design from the original CIFAKE paper: two convolutional
layers (32 filters each) with max pooling, followed by two dense layers
with dropout for regularization. This is a deliberately simple,
well-validated architecture rather than something over-engineered.

## Important Limitations — Know These Before Presenting Results

1. **Trained on a subset, not the full dataset** — expect somewhat
   lower accuracy than the published 92.93% benchmark, since less
   training data generally means a weaker model.

2. **CIFAKE's "fake" images are from Stable Diffusion 1.4 (2022)** —
   a real, published limitation is that CIFAKE-trained detectors don't
   generalize well to newer generators (Midjourney v6, DALL-E 3, Flux).
   This model will likely perform worse on modern AI images than on
   the CIFAKE test set. This is an active, acknowledged research problem
   — not a flaw specific to this implementation.

3. **Trained on 32x32 low-resolution images** — real-world photos are
   much higher resolution; performance on full-size images is untested
   here.

Being able to name these limitations clearly is a stronger interview
answer than claiming the model "detects AI images" without qualification.

## Pipeline

| Step | Script | What it does |
|------|--------|---------------|
| 1 | `src/01_fetch_data.py` | Downloads CIFAKE via Hugging Face, prepares a train/test subset |
| 2 | `src/02_train_cnn.py` | Builds and trains the CNN, evaluates on the test set |
| 3 | `src/03_predict_own_image.py` | Classifies any image you provide |

## How To Run

```bash
pip install tensorflow datasets pillow scikit-learn numpy
python src/01_fetch_data.py
python src/02_train_cnn.py
python src/03_predict_own_image.py path/to/your/image.jpg
```

Note: `01_fetch_data.py` downloads real data and needs internet access.
Training in step 2 may take several minutes on a CPU.

## Tech Stack

Python, TensorFlow/Keras, Hugging Face `datasets`, scikit-learn, PIL