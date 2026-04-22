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

pip install -r requirements.txt

2) Train and generate all artifacts

python train_ae_vae.py \
  --data_root data \
  --output_dir outputs \
  --image_size 128 \
  --channels 1 \
  --batch_size 64 \
  --epochs 20 \
  --memory_growth \
  --early_stopping \
  --early_stopping_patience 5 \
  --early_stopping_min_delta 1e-4 \
  --restore_best_weights

Train separate AE/VAE pairs per anatomical region:

python train_ae_vae.py \
  --data_root data \
  --output_dir outputs \
  --image_size 128 \
  --channels 1 \
  --batch_size 64 \
  --epochs 20 \
  --memory_growth \
  --early_stopping \
  --early_stopping_patience 5 \
  --early_stopping_min_delta 1e-4 \
  --restore_best_weights \
  --separate_by_region

Optional (faster steady-state, but may print extra XLA/cuDNN autotuning warnings):

python train_ae_vae.py --data_root data --output_dir outputs --jit_compile --memory_growth

## MLflow Tracking

Enable MLflow tracking (local by default, creates mlruns/):

python train_ae_vae.py \
  --data_root data \
  --output_dir outputs \
  --image_size 128 \
  --channels 1 \
  --batch_size 8 \
  --epochs 10 \
  --memory_growth \
  --enable_mlflow \
  --mlflow_experiment AE_VAE_Representation_Learning \
  --mlflow_run_name run_bs8_e10

Use a remote tracking server by adding:

--mlflow_tracking_uri http://127.0.0.1:5000

What gets logged:

- Training configuration parameters
- Dataset split sizes
- AE and VAE per-epoch metrics
- AE and VAE test metrics
- Early stopping configuration parameters
- Saved model weights and plots as artifacts

## Google Drive / Colab Run

1) Upload the project and dataset folders to Google Drive
2) Mount drive in Colab
3) Set working directory to the project root
4) Run the same training command, for example:

python train_ae_vae.py --data_root /content/drive/MyDrive/your_project/data

## Outputs

After training, artifacts are saved under a unique run directory to avoid overwriting:

- outputs/runs/<timestamp>_<run_name_or_run_id>/models/ae.weights.h5
- outputs/runs/<timestamp>_<run_name_or_run_id>/models/vae.weights.h5
- outputs/runs/<timestamp>_<run_name_or_run_id>/plots/ae_training_curves.png
- outputs/runs/<timestamp>_<run_name_or_run_id>/plots/vae_training_curves.png
- outputs/runs/<timestamp>_<run_name_or_run_id>/plots/ae_reconstructions.png
- outputs/runs/<timestamp>_<run_name_or_run_id>/plots/vae_reconstructions.png
- outputs/runs/<timestamp>_<run_name_or_run_id>/plots/ae_denoising.png
- outputs/runs/<timestamp>_<run_name_or_run_id>/plots/vae_denoising.png
- outputs/runs/<timestamp>_<run_name_or_run_id>/plots/vae_generated_samples.png
- outputs/runs/<timestamp>_<run_name_or_run_id>/plots/ae_latent_scatter.png
- outputs/runs/<timestamp>_<run_name_or_run_id>/plots/vae_latent_scatter.png

When using --separate_by_region, artifacts are grouped by region:

- outputs/runs/<timestamp>_<run_name_or_run_id>/regions/<RegionName>/models/ae.weights.h5
- outputs/runs/<timestamp>_<run_name_or_run_id>/regions/<RegionName>/models/vae.weights.h5
- outputs/runs/<timestamp>_<run_name_or_run_id>/regions/<RegionName>/plots/*.png
