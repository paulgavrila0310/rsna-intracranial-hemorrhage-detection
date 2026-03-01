# **Brain Hemorrhage Detection from CT Scans**

This project implements a PyTorch-based Convolutional Neural Network (CNN) to detect and classify different types of brain hemorrhages in Computer Tomography (CT) scans. It performs multi-label classification for 5 specific hemorrhage subtypes and one general "healthy/sick" indicator.

### Overview

- Dataset: Uses a custom RSNADataset class with a stratified 80/20 train-validation split to ensure balanced healthy/sick patient distribution.

- Preprocessing: 1-channel grayscale images are resized to 256×256. A CLAHE filter is applied to enhance local contrast, followed by normalization and random augmentations (cropping, flipping) to prevent overfitting.

- Model: HemorrhageCNN, a custom baseline feature extractor with 4 convolutional layers. It outputs raw logits to be optimized for multi-label classification.

- Training: Trained on Kaggle using dual NVIDIA T4 GPUs for 25 epochs. Uses BCEWithLogitsLoss with a pos_weight of 2.0 to heavily penalize false negatives, and the Adam optimizer (learning rate = 0.0001).

### Results

The baseline model achieved the following performance metrics:

- Exact Match Accuracy: 30.19% 

- Per-Label Accuracy: 81.53% 

- Top F1-Scores: 0.93 for general detection ("Any") and 0.77 for Intraventricular hemorrhages.
