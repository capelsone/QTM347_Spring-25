# Bird Calls in Colombia: Machine Learning

## Abstract
We developed a deep learning pipeline to classify bird species from short audio recordings, using spectrogram-based CNN models. Despite challenges like severe class imbalance, noise, and limited compute resources, we achieved reasonable performance (~24% precision) on a subset of species, laying a strong foundation for future improvements through better audio augmentation, efficient fine-tuning, and smarter model design.

---

## Objective
Classify bird species from audio recordings using deep learning models.  
Real-world challenges: overlapping bird calls, background noise, limited labeled data.  
Applications: conservation monitoring, biodiversity research.

---

## Dataset: BirdCLEF 2025
- 38,000+ audio recordings (.ogg) from El Silencio Natural Reserve, Colombia.
- Metadata provided via CSV and TXT files.
- Focus: streamline bird species classification to aid conservation research.

---

## Introduction
Goal: Build an accurate bird species classifier under GPU/time constraints, competing with lightweight baselines like BirdNetLite.

Approach:
- Preprocess audio to spectrograms.
- Build and train CNN-based deep learning models.
- Evaluate performance using Precision, Recall, and F1-Score metrics.

---

## Early Attempts
- Initial simple CNNs with basic augmentation.
- Very low accuracy (~1–2%), models struggled to learn meaningful features.

---

## Audio Preprocessing Pipeline
- Load and pad audio recordings.
- Generate mel-spectrograms (log-scaled).
- Save spectrograms as `.npy` arrays for efficient training.
- 70/30 stratified train-test split to handle class balance.

---

## Model 1: Simple CNN

**Setup:**
- Model: CNN (Conv2D → MaxPool → Flatten → Fully Connected Layer).
- Activation: ELU.
- Optimizer: Adam (learning rate 0.001).
- Batch Size: 32.
- Epochs: 50.

**Training Focus:**
- Subset: Top 50 most common bird species.

---

## Simple CNN Results
- **Precision:** 0.24
- **Recall:** 0.21
- Cross-Entropy Loss steadily decreased during training.
- Some species learned correctly, but generalization remained limited.

---

## Model 2: Complex CNN with Residual Connections

**Setup:**
- Custom CNN using skip (residual) connections.
- Stronger augmentations applied:
  - Random time shifts (±0.5s)
  - Random pitch shifts (±1 semitone)
  - Gaussian noise injection
- Cached spectrograms as `.npy` for speed.
- 5-Fold Stratified Cross-Validation.

---

## Complex CNN Results
- Validation Accuracy ≈ 4–5%.
- Severe class confusion: only dominant species (e.g., Great Kiskadee) recognized.
- Macro-averaged precision/recall close to 0 for most classes.
- Indicates dataset quality and augmentation mattered more than model complexity.

---

## Key Limitations
- Severe class imbalance (many rare species).
- Environmental noise and overlapping calls.
- Spectrograms lost fine-grained timing details.
- Computational resource constraints limited hyperparameter tuning and model scaling.

---

## Critical Discussion: Audio Augmentation and Hyperparameter Tuning Challenges
- **Audio augmentation is critical** to generalizing across noisy and variable bird calls.
- Our augmentation execution (time shift, pitch shift, noise) was basic and not fully tuned.
- Poor augmentation diversity likely harmed model generalization more than architectural limitations.
- Fine-tuning hyperparameters (learning rates, augmentation ranges, batch sizes) proved difficult under tight computational limits.
- Future success will depend on smarter, more efficient tuning strategies** — including early stopping, learning rate scheduling, lightweight architectures, and prioritizing augmentation.
- Our overall project framework closely matches the pipelines used by past BirdCLEF competition winners, validating our direction even if our execution was limited by resources.

---

## Next Steps
- **Strengthen audio augmentation pipeline:**
  - Implement dynamic time stretching, pitch shifting, environmental noise mixing.
  - Apply spectrogram-level augmentation (e.g., SpecAugment: frequency and time masking).
- **Tune augmentation parameters** for maximum class-preserving variability.
- **Smarter fine-tuning of hyperparameters** (early stopping, learning rate decay, regularization).
- Focus classification on top 10 most common species to reduce imbalance.
- Only after augmentation improvements, **fine-tune larger pretrained models** (e.g., MobileNetV2, PANNs).
- Explore semi-supervised learning (pseudo-labeling) to expand the effective training dataset.

---

## Repository Structure
- `QTM_347_project_final.ipynb`: Final simple CNN model with preprocessing and evaluation.
- `deep_learning_model.ipynb` and `preprocessing_pipeline.ipynb`: Early-stage model explorations and preprocessing.
- `ModelV2Eric.ipynb`: Complex CNN with skip connections, strong augmentation, and cross-validation.

---

## Conclusion
This project successfully built a complete ML pipeline from raw bird call audio recordings to deep learning classification. Despite challenges like class imbalance, environmental noise, and limited compute, we achieved measurable progress. Our framework aligns closely with past BirdCLEF competition strategies, setting a strong foundation for future improvement through better augmentation, smarter fine-tuning, and more powerful model training.

---

## References
- [BirdCLEF 2025 Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2025)
- [Velardo, Valerio. "The Sound of AI" YouTube Channel](https://www.youtube.com/@ValerioVelardoTheSoundofAI)
- [Bird Sound Recognition Using CNNs (ResearchGate)](https://www.researchgate.net/publication/334163277_Bird_Sound_Recognition_Using_a_Convolutional_Neural_Network)

---
