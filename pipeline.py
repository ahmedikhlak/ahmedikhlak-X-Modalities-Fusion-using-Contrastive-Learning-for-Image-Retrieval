# pipeline.py
# This script serves as a reproducible pipeline to execute the full experimental workflow
# from training to evaluation for the XM (X-Modalities) framework.

import os
import subprocess

# Step 1: Set environment variables or paths
# Adjust as needed depending on your local setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dataset_root = "./data"
model_save_dir = "./checkpoints"
log_dir = "./logs"

# Step 2: Train the combiner (Relational Network) with aligned features
# This uses combiner_train.py with CLIP-extracted embeddings
print("Starting RN Combiner Training...")
subprocess.run([
    "python", "combiner_train.py",
    "--dataset", "fashioniq",
    "--epochs", "20",
    "--batch_size", "32",
    "--clip_model", "ViT-B/32",
    "--log_dir", log_dir,
    "--save_dir", model_save_dir
])

# Step 3: Fine-tune CLIP using caption-guided training
# Adapts CLIP using FashionIQ-specific captions
print("Running CLIP Fine-tuning...")
subprocess.run([
    "python", "clip_fine_tune.py",
    "--encoder", "text",
    "--batch_size", "32",
    "--num_epochs", "10",
    "--learning_rate", "5e-5"
])

# Step 4: Run inference on CIRR test set
# Generates CIRR predictions using trained fusion strategy
print("Running CIRR Test Submission Inference...")
subprocess.run([
    "python", "cirr_test_submission.py",
    "--dataset", "cirr",
    "--clip_model", "ViT-B/32",
    "--save_dir", "./results"
])

# Step 5: Validate the model using CIRR and FashionIQ metrics
# Computes R@K, rank accuracy etc.
print("Evaluating Final Model on CIRR and FashionIQ...")
subprocess.run([
    "python", "validate.py",
    "--dataset", "fashioniq",
    "--clip_model", "ViT-B/32",
    "--batch_size", "32"
])