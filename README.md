# Representation Learning with AE and VAE

This repository contains a complete implementation of the assignment requirements:

- Autoencoder (AE) for reconstruction
- Variational Autoencoder (VAE) with reparameterization and KL regularization
- tf.data-based input pipeline from the provided image folders
- Latent space visualization
- Reconstruction comparison
- VAE sample generation
- Denoising capability check
- Loss tracking and plots

## Project Structure

- models/auto_encoder.py: AE model
- models/variatonal_auto_encoder.py: VAE model
- utils/data_pipeline.py: tf.data pipeline and split utilities
- utils/visualization.py: plotting helpers
- train_ae_vae.py: end-to-end training and analysis script

## Dataset Usage

The script uses the provided dataset directly from folder structure such as:

- data/AbdomenCT
- data/BreastMRI
- data/ChestCT
- data/CXR
- data/Hand
- data/HeadCT

No CSV or NPZ conversion is used.

## Local Run

1) Install dependencies

pip install tensorflow matplotlib numpy

2) Train and generate all artifacts

python train_ae_vae.py \
  --data_root data \
  --output_dir outputs \
  --image_size 128 \
  --channels 1 \
  --batch_size 64 \
  --epochs 20 \
  --memory_growth

Optional (faster steady-state, but may print extra XLA/cuDNN autotuning warnings):

python train_ae_vae.py --data_root data --output_dir outputs --jit_compile --memory_growth

## Google Drive / Colab Run

1) Upload the project and dataset folders to Google Drive
2) Mount drive in Colab
3) Set working directory to the project root
4) Run the same training command, for example:

python train_ae_vae.py --data_root /content/drive/MyDrive/your_project/data

## Outputs

After training, artifacts are saved under outputs/:

- outputs/models/ae.weights.h5
- outputs/models/vae.weights.h5
- outputs/plots/ae_training_curves.png
- outputs/plots/vae_training_curves.png
- outputs/plots/ae_reconstructions.png
- outputs/plots/vae_reconstructions.png
- outputs/plots/ae_denoising.png
- outputs/plots/vae_denoising.png
- outputs/plots/vae_generated_samples.png
- outputs/plots/ae_latent_scatter.png
- outputs/plots/vae_latent_scatter.png
