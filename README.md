# adaptive-learning-movielens
Hebbian, Oja and Anti-Hebbian learning rules applied to MovieLens user behaviour data
## Overview

This project explores **adaptive intelligent systems for human behaviour** using the MovieLens 100K dataset. Instead of standard supervised learning, we implement and compare three biologically-inspired learning rules:

- **Hebbian Learning** — "Neurons that fire together, wire together"
- **Oja's Rule** — Stable Hebbian learning that performs PCA
- **Anti-Hebbian Learning** — Forces neurons to learn independent features (Foldiak's network)

We also compare these unsupervised rules with standard **Backpropagation** to highlight the difference between supervised and unsupervised learning.

---

## Dataset

**MovieLens 100K** — [GroupLens Research](https://grouplens.org/datasets/movielens/100k/)

- 943 users
- 1,682 movies
- 100,000 ratings (1–5 scale)

---

## Methods

### 1. Hebbian Learning
The simplest learning rule: strengthen connections between co-activating neurons.

$$\Delta W = \eta \cdot y \cdot x^T$$

**Result:** Weights explode to infinity — no stability mechanism.

### 2. Oja's Rule
Adds a normalisation term to Hebbian learning, making it stable and equivalent to PCA.

$$\Delta W_k = \eta (y_k x^T - y_k^2 W_k)$$

**Result:** Stable weights, learns the principal component of user-movie interactions — identifies the most discriminative films.

### 3. Anti-Hebbian Learning (Foldiak's Network)
Uses both feedforward weights (W, updated with Oja's rule) and recurrent inhibitory weights (B, updated with Anti-Hebbian rule) to force neurons to learn independent features.

$$\Delta B = -\eta \cdot y \cdot y^T$$

**Result:** Neurons learn partially independent directions in the data — different user behaviour patterns.

### 4. Backpropagation (Supervised Baseline)
Standard neural network with cross-entropy loss, predicting whether a user will click/rate a movie highly.

**Result:** MAE ≈ 0.905 on held-out test set.

---

## Key Findings

| Method | Stability | Labels Needed | What it learns |
|--------|-----------|---------------|----------------|
| Hebbian | ❌ Explodes | No | Co-activation patterns |
| Oja | ✅ Stable | No | Principal component (PCA) |
| Anti-Hebbian | ✅ Stable | No | Independent user profiles |
| Backpropagation | ✅ Stable | Yes | Rating prediction |

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

---

## Usage

```bash
# Download MovieLens 100K dataset
# Place u.data in the project directory

jupyter notebook adaptive_learning_movielens.ipynb
```

---

## Project Structure

```
├── README.md
├── adaptive_learning_movielens.ipynb   # Main notebook
└── u.data                              # MovieLens dataset (download separately)
```

---

## Background

This project was developed as part of a self-study program on **adaptive intelligent systems**, drawing from:

- Bishop, C. (2006). *Pattern Recognition and Machine Learning*
- Sutton & Barto. *Reinforcement Learning: An Introduction*
- Google Machine Learning — Recommendation Systems

---

## Author

Developed as a portfolio project for graduate school applications in Machine Learning / Adaptive Systems.
