"""Plotting and qualitative analysis helpers for AE/VAE experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_training_curves(
    history: dict[str, list[float]],
    output_path: str,
    title: str,
    include_kl: bool = False,
) -> None:
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    if "loss" in history:
        plt.plot(history["loss"], label="train")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="val")
    plt.title(f"{title} Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    if include_kl:
        if "reconstruction_loss" in history:
            plt.plot(history["reconstruction_loss"], label="train recon")
        if "val_reconstruction_loss" in history:
            plt.plot(history["val_reconstruction_loss"], label="val recon")
        if "kl_loss" in history:
            plt.plot(history["kl_loss"], label="train kl")
        if "val_kl_loss" in history:
            plt.plot(history["val_kl_loss"], label="val kl")
        plt.title(f"{title} Recon/KL")
    else:
        if "mae" in history:
            plt.plot(history["mae"], label="train mae")
        if "val_mae" in history:
            plt.plot(history["val_mae"], label="val mae")
        plt.title(f"{title} Reconstruction MAE")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_reconstructions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    output_path: str,
    title: str,
    num_images: int = 8,
) -> None:
    batch = next(iter(dataset))
    x_in, x_true = batch

    x_in = x_in[:num_images]
    x_true = x_true[:num_images]
    x_pred = model(x_in, training=False)

    n = x_in.shape[0]
    plt.figure(figsize=(2.2 * n, 6.5))
    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(tf.squeeze(x_in[i]), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Input")

        plt.subplot(3, n, n + i + 1)
        plt.imshow(tf.squeeze(x_true[i]), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Target")

        plt.subplot(3, n, 2 * n + i + 1)
        plt.imshow(tf.squeeze(x_pred[i]), cmap="gray")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Reconstruction")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_generated_samples(samples: tf.Tensor, output_path: str, title: str) -> None:
    samples = tf.convert_to_tensor(samples)
    n = samples.shape[0]
    cols = min(8, n)
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(2 * cols, 2 * rows))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(tf.squeeze(samples[i]), cmap="gray")
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def project_to_2d(latent_vectors: np.ndarray) -> np.ndarray:
    centered = latent_vectors - latent_vectors.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    projected = centered @ components
    return projected


def save_latent_scatter(
    latent_vectors: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output_path: str,
    title: str,
) -> None:
    projected = project_to_2d(latent_vectors)

    plt.figure(figsize=(8, 6))
    for class_id, class_name in enumerate(class_names):
        mask = labels == class_id
        if np.any(mask):
            plt.scatter(projected[mask, 0], projected[mask, 1], s=8, alpha=0.65, label=class_name)

    plt.title(title)
    plt.xlabel("Latent Component 1")
    plt.ylabel("Latent Component 2")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
