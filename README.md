# Fashion-MNIST CNN Classifier

A convolutional neural network built from scratch (TensorFlow/Keras) to classify Fashion-MNIST images into 10 clothing categories. This was my first end-to-end CNN project — built to practice a proper, professional pipeline (not just filling in Coursera notebook blanks): data exploration, preprocessing, baseline modeling, iterative regularization, and a disciplined, one-time final test evaluation.

**Final test accuracy: 92.99%** (test loss: 0.2056)

---

## Problem

- **Input:** 28x28 grayscale images of clothing items
- **Output:** 10 mutually exclusive classes (single-label classification)
- **Dataset:** 60,000 training images + 10,000 test images, perfectly class-balanced (6,000 images/class in training)
- **Metric:** Accuracy — appropriate here since classes are balanced
- **Baseline to beat:** a linear SVC scored ~0.44 in earlier experimentation; a random/majority-class guess would land around 10%
- **Target:** ~90% accuracy, a reasonable bar for a well-built CNN on this dataset

---

## Project Structure

```
Fashion_MNIST_classifier/
├── notebooks/
│   ├── 01_data_exploration.ipynb      # loading, inspection, visualization
│   ├── 02_model_training.ipynb        # all training iterations/experiments
│   └── 03_final_evaluation.ipynb      # one-time test set evaluation
├── src/
│   └── data_loader.py                 # reusable load + preprocess pipeline
├── models/                            # saved weights per experiment
├── results/                           # loss/accuracy plots, architecture summaries
└── README.md
```

Datasets are not stored in the repo — Fashion-MNIST is loaded directly via `tensorflow.keras.datasets.fashion_mnist`, cached automatically by Keras at runtime.

---

## Data Pipeline

1. **Load** via `fashion_mnist.load_data()`
2. **Verify** — checked shapes, pixel range (0–255), label set (0–9), and class balance (exactly 6,000 images/class) before assuming anything about the data
3. **Visualize** a sample grid to sanity-check that labels correspond to plausible, consistent clothing types
4. **Normalize** — pixel values scaled from 0–255 to 0–1 (stabilizes gradients, standard practice for image inputs)
5. **Train/validation split** — 54,000 / 6,000 split carved out of the *training* set only (`train_test_split`, `random_state=42` for reproducibility). The test set was never touched until final evaluation, to keep it an honest, unbiased final check.
6. **Reshape** — added the channel dimension (`(28,28)` → `(28,28,1)`) since `Conv2D` expects `(height, width, channels)`
7. **Augmentation** (used in Runs #4–#6 only — **not** part of the final selected model, Run #3) — applied via `ImageDataGenerator`, training set only, never to validation/test:
   - `horizontal_flip=True` — valid for clothing images (a mirrored shirt/shoe still looks realistic)
   - `rotation_range=15`, `zoom_range=0.1`, `width/height_shift_range=0.1` — small, realistic perturbations appropriate for small 28x28 images
   - Vertical flip deliberately excluded — upside-down clothing isn't realistic training data

All loading/preprocessing logic lives in `src/data_loader.py` as a single reusable function, imported into every notebook that needs it — avoids duplicating pipeline code across notebooks.

---

## Model Architecture (Final / Best-Performing Version)

```
Conv2D(32, 3x3, relu, padding=same)
Conv2D(32, 3x3, relu, padding=same)
MaxPooling2D(2x2)
Dropout(0.2)
Conv2D(64, 3x3, relu, padding=same)
Conv2D(64, 3x3, relu, padding=same)
MaxPooling2D(2x2)
Dropout(0.2)
Flatten
Dense(512, relu)
Dropout(0.3)
Dense(10, softmax)
```

**Note:** the selected final model (Run #3) was trained with Dropout only — **no data augmentation**. Augmentation was introduced starting at Run #4 as a separate experiment, and while it successfully closed the train/val gap, it came at the cost of worse raw validation performance (see Experiment Log below). Since Run #3 had the best validation numbers overall, it was chosen as final, and it does not include augmentation in its training pipeline.

- **Optimizer:** Adam (adaptive learning rate + momentum, more forgiving out-of-the-box than plain SGD)
- **Loss:** Sparse categorical crossentropy (integer labels, not one-hot)
- **Callbacks:** `EarlyStopping` (monitor `val_loss`, patience=5, `restore_best_weights=True`) and `ModelCheckpoint` (saves best-val-loss weights to disk) — used as a practical safety net against wasted compute and uncontrolled overfitting, not as a substitute for proper regularization.

Design reasoning:
- Small 3x3 kernels throughout (VGG-style convention) rather than large kernels — better detail capture with fewer parameters, appropriate for small 28x28 inputs
- Filter counts double with depth (32→64) — early layers detect simple/generic patterns, deeper layers combine them into more complex ones
- Two pooling stages (28→14→7) — a 28x28 image can't meaningfully support more than 2-3 pooling stages before losing too much spatial resolution
- Dense(512) sized down from an initial over-large Dense(2048) guess, after recognizing 2048 units was disproportionate to the task's complexity

---

## Experiment Log

All experiments compared using **validation** metrics only. Test set was touched exactly once, at the very end, on the selected final model.

| # | Version | Regularization | Best Val Loss | Val Acc | Train Acc | Epochs (stopped/max) | Takeaway |
|---|---------|-----------------|---------------|---------|-----------|----------------------|----------|
| 1 | Baseline | None | 0.1958 | 92.73% | ~94% | 9 / 50 | Clear overfitting after epoch 4; train raced to ~98% by the time it stopped |
| 2 | + Dropout (dense only, 0.3) | Dropout before output layer | 0.1981 | 93.47% | ~95.7% | 14 / 50 | Slight improvement; overfitting delayed, still present |
| 3 | + Dropout (dense + conv, 0.2/0.3) | Dropout after each pooling + before output | **0.1884** | **93.58%** | ~95.5% | 14 / 50 | **Best raw validation performance** of all experiments; real but moderate train/val gap |
| 4 | Dropout(3-layer) + Augmentation | Same as #3 + flip/rotate/zoom/shift | ~0.214–0.217 | ~92.4% | ~90.7% | 21 / 50 | Overfitting gap essentially closed, but worse raw performance — likely over-regularized (too much combined with augmentation) |
| 5 | Reduced Dropout + Augmentation | Lower dropout rates, same augmentation | 0.1970 | 92.78% | 91.38% | 30 / 50 (35 run) | Confirms #4 was over-regularized; healthier gap than #3, but still below #3 on raw val performance |
| 6 | Reduced architecture (2 conv layers) | Same dropout/augmentation as #5 | ~0.228 | ~91.5% | ~88.1% | 17 / 50 (21 run) | **Underfitting** — worse on both train and val; disproved the "architecture was oversized" hypothesis at this reduction level |

**Final decision: Run #3** was selected as the final model. It had the best validation loss and accuracy of any experiment. Its ~2-percentage-point train/val gap is real but moderate — not the runaway overfitting seen in the baseline — and reducing that gap further (runs #4/#5) consistently cost more in raw performance than it gained in generalization safety, based on the evidence gathered.

---

## Final Result

Run #3's saved weights (Dropout only, no augmentation) were loaded fresh in a dedicated evaluation notebook and evaluated **once** on the untouched 10,000-image test set:

```
Test Loss:     0.2056
Test Accuracy: 92.99%
```

Test accuracy (92.99%) tracked closely with validation accuracy (93.58%) — only a ~0.6 point drop, and test loss was only modestly higher than validation loss. This close correspondence is a good sign that the validation-driven model-selection process was sound, and that this number is a trustworthy estimate of real-world generalization, not an artifact of an unrepresentative validation set.

This exceeds the original ~90% target and dramatically outperforms the earlier linear SVC baseline (~44%).

---

## What I'd Try Next

- BatchNormalization (not used here — a natural next regularization/stabilization technique to test)
- L2 weight decay as an alternative/complement to Dropout
- A more targeted architecture search between the 4-conv (run #3/#5) and 2-conv (run #6) extremes, rather than jumping straight to the smaller end
- Transfer learning comparison, to see how a from-scratch CNN stacks up against a fine-tuned pretrained backbone on this dataset

---

## Key Lessons From This Project

- Accuracy and loss can tell different stories — loss is the more sensitive early-warning signal for overfitting, since it reflects prediction *confidence*, not just correctness
- Validation and test sets serve different purposes: validation guides iteration, test is touched exactly once, at the end, to avoid indirectly "fitting" to it through repeated decisions
- A smaller train/val gap isn't automatically better — it needs to be weighed against absolute validation performance, not treated as a goal in itself
- Data augmentation must be applied only to training data — augmenting validation/test would corrupt their purpose as honest, unaltered generalization checks
- GPU vs CPU makes an enormous practical difference in iteration speed (~270ms/step → ~25ms/step switching CPU to T4 GPU) — matters even for a dataset this small
