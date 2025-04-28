# QTM347_Spring-25: BirdCLEF 2025 Bird Call Classification Project

## Project Overview

This project aims to classify bird species based on their audio recordings, inspired by real-world applications like the Merlin Bird App.
Using the BirdCLEF 2025 dataset, we generate spectrograms from audio files and train a lightweight convolutional neural network (CNN) model.

## Data

**Source**: BirdCLEF 2025 dataset (Xeno-Canto recordings)

**Samples**: Hundreds of bird species with varying sample counts, many classes severely imbalanced.

## Model Evolution Overview

### Initial Models Attempted

#### 1. First Attempt (Baseline CNN)
- Simple convolutional model trained directly on mel-spectrograms.
- No data augmentation.
- Trained on 50+ bird species simultaneously.

**Key Issues:**
- Severe class imbalance.
- Quick overfitting to training data.
- Very low validation accuracy (~1-2%).

#### 2. Second Attempt (Bootstrapping + Larger CNN)
- Introduced basic bootstrapping (oversampling underrepresented classes).
- Used a deeper CNN model.

**Outcomes:**
- Slight improvement in validation accuracy (~4-5%).
- Training slowed considerably.
- Still significant confusion between bird species.
- Key limitation remained: small dataset size relative to the number of classes.

### Final Model (ModelV2Eric.ipynb)

#### Data Preprocessing
- Selected the top 10 species from metadata.
- Extracted 5-second fixed-length audio clips.
- Converted audio clips into mel-spectrograms (128 mel bands).
- Saved spectrograms as `.npy` files to speed up future training runs.

#### Data Augmentation
- Applied random augmentations during spectrogram generation:
  - Time Shifting (±0.5 seconds)
  - Pitch Shifting (±1 semitone)
  - Gaussian Noise Injection (standard deviation = 0.005)

**Purpose:**
- Simulate a larger dataset.
- Force the model to learn more generalized features rather than memorizing.

#### Model Architecture (Mini-ResNet)
- Input Layer: Mel-spectrogram input.
- Two Conv2D layers, each followed by BatchNormalization.
- Skip connection (residual) between input and conv layers.
- Global Average Pooling.
- Dense output layer with softmax activation for multi-class classification.

#### Training Strategy
- Stratified 5-Fold Cross Validation to maintain class balance across folds.
- EarlyStopping and ReduceLROnPlateau callbacks to optimize training.
- Optimizer: Adam
- Batch Size: 32
- Maximum Epochs: 70 (with early stopping enabled)

## Results

**Validation Accuracy:**
- Remained low, around 4-5%.

**Confusion Matrix Observations:**
- Only a few species (e.g., Great Kiskadee) were consistently recognized.
- Most classes were not correctly classified at all.

**Overall Trends:**
- The model made marginal improvements over the very first baseline.
- However, the fundamental issues of extreme class imbalance, limited data per species, and environmental noise could not be overcome without significantly more data or a much more advanced model.

**Key Insight:**
> Despite multiple improvements to preprocessing, data augmentation, and model architecture, results remained poor. This underscores the critical importance of dataset quality, quantity, and careful class balance in real-world machine learning applications.

## Key Limitations

- Severe class imbalance in the dataset.
- Very few audio samples per class for some species.
- Environmental noise and recording artifacts.
- Limited compute resources restricted exploration of larger models.
- Mini-ResNet, while lightweight, lacked the complexity needed for fine-grained classification.

## Next Steps

- Narrow focus even further to only the most common species.
- Utilize pre-trained audio networks (e.g., YAMNet, PANNs) for better initial feature extraction.
- Experiment with semi-supervised learning techniques (pseudo-labeling, self-training).
- Implement stronger augmentations and/or synthetic data generation.
- Consider fine-tuning larger architectures if more compute resources are available.
