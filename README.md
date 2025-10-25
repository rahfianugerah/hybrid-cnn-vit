<<<<<<< HEAD
### Hybrid CNN–Vision Transformer Notebook for Image Classification

<p align="justify">
This repository contains a single Jupyter Notebook implementing a hybrid Convolutional Neural Network (CNN) and Vision Transformer (ViT) architecture for image classification. The approach combines convolutional feature extraction with transformer-based global self-attention, including patch embedding to convert image blocks into token sequences for the transformer encoder.
</p>

<p align="justify">
All training, hyperparameter search, model evaluation, and logging are performed inside the notebook without external Python scripts.
</p>

### Project Objective
<p align="justify">
The goal of this work is to develop and evaluate a hybrid CNN–ViT model using the CIFAR-10 dataset. The model leverages:
</p>

- CNN layers for localized spatial feature extraction  
- Patch embedding to convert image patches into token representations  
- Transformer encoder layers for global context modeling  
- Optuna for automated hyperparameter tuning  
- GPU acceleration for efficient training  

This notebook provides a complete, reproducible training pipeline suitable for academic research, benchmarking, and competition submissions.

### Notebook Pipeline Overview

The notebook is structured into the following sections:

1. Environment setup and library installation
2. Data loading and augmentation pipeline
3. Hybrid CNN–ViT model definition including patch embedding
4. Training loop with logging and checkpointing
5. Optuna hyperparameter optimization
6. Evaluation and performance reporting
7. Saving model artifacts and metrics

### Patch Embedding
<p align="justify">
The Vision Transformer component applies a <b>patch embedding stage</b> to convert the input image tensor into a sequence of patch tokens.
</p>

Key Mechanism:

- The image is split into fixed-size patches
- Each patch is flattened and projected into an embedding dimension
- Positional embeddings are added before feeding into transformer layers

This process allows the transformer to attend globally across the image.

### Dataset

- Dataset: `CIFAR-10`
- Image size: `32×32`
- Classes: `10`  

Split used in notebook:

- `50000` training samples  
- `10000` testing samples  

Data augmentation includes standard normalization, random cropping, and horizontal flipping.

### Training Configuration

Training setup executed inside the notebook includes:

- Loss: Cross-Entropy
- Optimizer: Adam
- Learning rate scheduling: Cosine annealing
- Hardware: CUDA GPU
- Automated hyperparameter search: Optuna trials

The notebook automatically logs:

- Training accuracy
- Test accuracy
- F1-score
- Loss curves
- Best model checkpoint

### Results

Extracted from notebook execution logs:

- Best Top-1 Test Accuracy: `85.3%`
- Best Test F1-Score: `0.85` 

These metrics were obtained after running Optuna-tuned training on GPU.

### How to Use the Notebook

1. Open the `.ipynb` file in Jupyter, Colab, or VSCode
2. Run all cells sequentially
3. The notebook will:
   - Train the hybrid CNN-ViT model
   - Perform hyperparameter search
   - Save best model weights and metrics automatically

### Files Included

- `hybrid-cnn-vit-updated.ipynb` > end-to-end notebook

Other directories created automatically during notebook execution may include:

- Checkpoint files
- Metric logs
- Saved Optuna studies
- Output artifacts (e.g., confusion matrices, evaluation reports)

### Reproducibility

The notebook implements:

- Seed initialization
- Deterministic settings where supported
- Saving of best trial configuration
- Saving of trained model state

### Citation
```bash
@software{
   hybrid_cnn_vit_optuna_2025,
   title={Hybrid CNN–Vision Transformer for Image Classification},
   author={Clavino Ourizqi Rachmadi, Naufal Rahfi Anugerah},
   year={2025},
   url={https://github.com/your-repo}
}
```

### Notes
<p align="justify">
This notebook demonstrates a complete hybrid deep learning pipeline inside one reproducible file. Results reported are taken directly from the executed training logs. The patch embedding layer is part of the Vision Transformer module integrated into the hybrid architecture.
</p>