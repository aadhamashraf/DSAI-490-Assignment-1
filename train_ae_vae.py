"""Assignment training script for AE and VAE representation learning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.auto_encoder import AutoEncoder
from models.variatonal_auto_encoder import VariationalAutoEncoder
from utils.data_pipeline import (
    build_autoencoder_dataset,
    build_embedding_dataset,
    prepare_splits,
)
from utils.visualization import (
    ensure_dir,
    save_generated_samples,
    save_latent_scatter,
    save_reconstructions,
    save_training_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AE and VAE on the medical image dataset.")
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--image_size", type=int, default=128, help="Input image size.")
    parser.add_argument("--channels", type=int, default=1, choices=[1, 3], help="Image channels.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--latent_dim_ae", type=int, default=32, help="Latent size for AE.")
    parser.add_argument("--latent_dim_vae", type=int, default=16, help="Latent size for VAE.")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta coefficient for VAE KL term.")
    parser.add_argument("--denoise_noise", type=float, default=0.2, help="Noise stddev for denoising demo.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--jit_compile",
        action="store_true",
        help="Enable XLA JIT compile for faster steady-state training (can add startup warnings).",
    )
    parser.add_argument(
        "--memory_growth",
        action="store_true",
        help="Enable GPU memory growth to avoid grabbing all GPU memory at once.",
    )
    return parser.parse_args()


def build_latent_arrays(
    ae: AutoEncoder,
    vae: VariationalAutoEncoder,
    embed_ds: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ae_latents: list[np.ndarray] = []
    vae_latents: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for images, y in embed_ds:
        z_ae = ae.encode(images, training=False).numpy()
        z_mean, _, _ = vae.encoder(images, training=False)
        z_vae = z_mean.numpy()

        ae_latents.append(z_ae)
        vae_latents.append(z_vae)
        labels.append(y.numpy())

    ae_array = np.concatenate(ae_latents, axis=0)
    vae_array = np.concatenate(vae_latents, axis=0)
    labels_array = np.concatenate(labels, axis=0)
    return ae_array, vae_array, labels_array


def main() -> None:
    args = parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    if args.memory_growth:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    image_size = (args.image_size, args.image_size)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(output_dir / "models")
    ensure_dir(output_dir / "plots")

    split = prepare_splits(data_root=args.data_root, val_ratio=0.1, test_ratio=0.1, seed=args.seed)

    train_ds = build_autoencoder_dataset(
        split.train_paths,
        batch_size=args.batch_size,
        image_size=image_size,
        channels=args.channels,
        shuffle=True,
        noise_stddev=0.0,
    )
    val_ds = build_autoencoder_dataset(
        split.val_paths,
        batch_size=args.batch_size,
        image_size=image_size,
        channels=args.channels,
        shuffle=False,
        noise_stddev=0.0,
    )
    test_ds = build_autoencoder_dataset(
        split.test_paths,
        batch_size=args.batch_size,
        image_size=image_size,
        channels=args.channels,
        shuffle=False,
        noise_stddev=0.0,
    )

    denoise_val_ds = build_autoencoder_dataset(
        split.val_paths,
        batch_size=args.batch_size,
        image_size=image_size,
        channels=args.channels,
        shuffle=False,
        noise_stddev=args.denoise_noise,
    )

    input_shape = (args.image_size, args.image_size, args.channels)

    ae = AutoEncoder(input_shape=input_shape, latent_dim=args.latent_dim_ae)
    ae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        jit_compile=args.jit_compile,
    )

    ae_history = ae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
    )

    ae_test_metrics = ae.evaluate(test_ds, return_dict=True, verbose=0)
    print("AE test metrics:", ae_test_metrics)

    ae.save_weights(output_dir / "models" / "ae.weights.h5")

    vae = VariationalAutoEncoder(
        input_shape=input_shape,
        latent_dim=args.latent_dim_vae,
        beta=args.beta,
    )
    vae.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        jit_compile=args.jit_compile,
    )

    vae_history = vae.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        verbose=1,
    )

    vae_test_metrics = vae.evaluate(test_ds, return_dict=True, verbose=0)
    print("VAE test metrics:", vae_test_metrics)

    # With a custom train_step, Keras may not mark the outer subclassed model as built.
    _ = vae(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)
    vae.save_weights(output_dir / "models" / "vae.weights.h5")

    save_training_curves(
        ae_history.history,
        output_path=str(output_dir / "plots" / "ae_training_curves.png"),
        title="Autoencoder",
        include_kl=False,
    )
    save_training_curves(
        vae_history.history,
        output_path=str(output_dir / "plots" / "vae_training_curves.png"),
        title="Variational Autoencoder",
        include_kl=True,
    )

    save_reconstructions(
        ae,
        val_ds,
        output_path=str(output_dir / "plots" / "ae_reconstructions.png"),
        title="AE Reconstructions (Clean)",
    )
    save_reconstructions(
        vae,
        val_ds,
        output_path=str(output_dir / "plots" / "vae_reconstructions.png"),
        title="VAE Reconstructions (Clean)",
    )

    save_reconstructions(
        ae,
        denoise_val_ds,
        output_path=str(output_dir / "plots" / "ae_denoising.png"),
        title="AE Denoising",
    )
    save_reconstructions(
        vae,
        denoise_val_ds,
        output_path=str(output_dir / "plots" / "vae_denoising.png"),
        title="VAE Denoising",
    )

    generated = vae.sample(num_samples=32)
    save_generated_samples(
        generated,
        output_path=str(output_dir / "plots" / "vae_generated_samples.png"),
        title="VAE Generated Samples",
    )

    max_embed = min(2000, len(split.val_paths))
    embed_ds = build_embedding_dataset(
        paths=split.val_paths[:max_embed],
        labels=split.val_labels[:max_embed],
        batch_size=args.batch_size,
        image_size=image_size,
        channels=args.channels,
    )

    ae_latent, vae_latent, embed_labels = build_latent_arrays(ae, vae, embed_ds)

    save_latent_scatter(
        ae_latent,
        embed_labels,
        class_names=split.class_names,
        output_path=str(output_dir / "plots" / "ae_latent_scatter.png"),
        title="AE Latent Space (2D PCA Projection)",
    )
    save_latent_scatter(
        vae_latent,
        embed_labels,
        class_names=split.class_names,
        output_path=str(output_dir / "plots" / "vae_latent_scatter.png"),
        title="VAE Latent Space (2D PCA Projection)",
    )

    print("All artifacts saved under:", output_dir.resolve())


if __name__ == "__main__":
    main()
