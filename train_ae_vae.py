"""Assignment training script for AE and VAE representation learning."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--enable_mlflow",
        action="store_true",
        help="Enable MLflow experiment tracking.",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="",
        help="MLflow tracking URI. Leave empty to use default local mlruns directory.",
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default="AE_VAE_Representation_Learning",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow_run_name",
        type=str,
        default="",
        help="Optional MLflow run name.",
    )
    return parser.parse_args()


def _log_history_metrics(
    mlflow_module: Any,
    model_name: str,
    history: dict[str, list[float]],
) -> None:
    if not history:
        return

    max_epochs = max(len(values) for values in history.values() if isinstance(values, list))
    for epoch in range(max_epochs):
        for metric_name, values in history.items():
            if epoch < len(values):
                mlflow_module.log_metric(
                    f"{model_name}_{metric_name}",
                    float(values[epoch]),
                    step=epoch,
                )


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

    mlflow_module = None
    mlflow_run = None
    if args.enable_mlflow:
        try:
            import mlflow as _mlflow  # type: ignore

            mlflow_module = _mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflow is not installed. Install it with: pip install mlflow"
            ) from exc

        if args.mlflow_tracking_uri:
            mlflow_module.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow_module.set_experiment(args.mlflow_experiment)

        mlflow_run = mlflow_module.start_run(
            run_name=args.mlflow_run_name if args.mlflow_run_name else None
        )

    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    if args.memory_growth:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    image_size = (args.image_size, args.image_size)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = args.mlflow_run_name.strip() if args.mlflow_run_name else "run"
    if mlflow_run is not None:
        run_label = mlflow_run.info.run_id

    run_dir = output_dir / "runs" / f"{timestamp}_{run_label}"
    models_dir = run_dir / "models"
    plots_dir = run_dir / "plots"
    ensure_dir(models_dir)
    ensure_dir(plots_dir)

    split = prepare_splits(data_root=args.data_root, val_ratio=0.1, test_ratio=0.1, seed=args.seed)

    if mlflow_module is not None:
        mlflow_module.log_params(
            {
                "data_root": args.data_root,
                "output_dir": str(output_dir),
                "run_dir": str(run_dir),
                "image_size": args.image_size,
                "channels": args.channels,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "latent_dim_ae": args.latent_dim_ae,
                "latent_dim_vae": args.latent_dim_vae,
                "beta": args.beta,
                "denoise_noise": args.denoise_noise,
                "seed": args.seed,
                "jit_compile": args.jit_compile,
                "memory_growth": args.memory_growth,
            }
        )
        mlflow_module.log_params(
            {
                "num_classes": len(split.class_names),
                "num_train": len(split.train_paths),
                "num_val": len(split.val_paths),
                "num_test": len(split.test_paths),
            }
        )

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

    if mlflow_module is not None:
        _log_history_metrics(mlflow_module, "ae", ae_history.history)
        for metric_name, metric_value in ae_test_metrics.items():
            mlflow_module.log_metric(f"ae_test_{metric_name}", float(metric_value))

    ae.save_weights(models_dir / "ae.weights.h5")

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

    if mlflow_module is not None:
        _log_history_metrics(mlflow_module, "vae", vae_history.history)
        for metric_name, metric_value in vae_test_metrics.items():
            mlflow_module.log_metric(f"vae_test_{metric_name}", float(metric_value))

    # With a custom train_step, Keras may not mark the outer subclassed model as built.
    _ = vae(tf.zeros((1, *input_shape), dtype=tf.float32), training=False)
    vae.save_weights(models_dir / "vae.weights.h5")

    save_training_curves(
        ae_history.history,
        output_path=str(plots_dir / "ae_training_curves.png"),
        title="Autoencoder",
        include_kl=False,
    )
    save_training_curves(
        vae_history.history,
        output_path=str(plots_dir / "vae_training_curves.png"),
        title="Variational Autoencoder",
        include_kl=True,
    )

    save_reconstructions(
        ae,
        val_ds,
        output_path=str(plots_dir / "ae_reconstructions.png"),
        title="AE Reconstructions (Clean)",
    )
    save_reconstructions(
        vae,
        val_ds,
        output_path=str(plots_dir / "vae_reconstructions.png"),
        title="VAE Reconstructions (Clean)",
    )

    save_reconstructions(
        ae,
        denoise_val_ds,
        output_path=str(plots_dir / "ae_denoising.png"),
        title="AE Denoising",
    )
    save_reconstructions(
        vae,
        denoise_val_ds,
        output_path=str(plots_dir / "vae_denoising.png"),
        title="VAE Denoising",
    )

    generated = vae.sample(num_samples=32)
    save_generated_samples(
        generated,
        output_path=str(plots_dir / "vae_generated_samples.png"),
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
        output_path=str(plots_dir / "ae_latent_scatter.png"),
        title="AE Latent Space (2D PCA Projection)",
    )
    save_latent_scatter(
        vae_latent,
        embed_labels,
        class_names=split.class_names,
        output_path=str(plots_dir / "vae_latent_scatter.png"),
        title="VAE Latent Space (2D PCA Projection)",
    )

    if mlflow_module is not None:
        mlflow_module.log_artifacts(str(models_dir), artifact_path=f"{run_dir.name}/models")
        mlflow_module.log_artifacts(str(plots_dir), artifact_path=f"{run_dir.name}/plots")

    print("All artifacts saved under:", run_dir.resolve())

    if mlflow_run is not None:
        mlflow_module.end_run()


if __name__ == "__main__":
    main()
