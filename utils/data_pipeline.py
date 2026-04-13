"""tf.data input pipeline utilities for the AE/VAE assignment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import tensorflow as tf

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class SplitData:
    train_paths: list[str]
    train_labels: list[int]
    val_paths: list[str]
    val_labels: list[int]
    test_paths: list[str]
    test_labels: list[int]
    class_names: list[str]


def discover_image_paths(data_root: str) -> tuple[list[str], list[int], list[str]]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not class_names:
        raise ValueError(f"No class directories were found in: {data_root}")

    paths: list[str] = []
    labels: list[int] = []

    for class_index, class_name in enumerate(class_names):
        class_dir = root / class_name
        image_paths = sorted(
            p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        )
        for image_path in image_paths:
            paths.append(str(image_path))
            labels.append(class_index)

    if not paths:
        raise ValueError(f"No supported image files were found in: {data_root}")

    return paths, labels, class_names


def split_dataset(
    paths: Sequence[str],
    labels: Sequence[int],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int], list[str], list[int]]:
    if len(paths) != len(labels):
        raise ValueError("Paths and labels must have the same length.")

    indices = list(range(len(paths)))
    random.Random(seed).shuffle(indices)

    total_count = len(indices)
    test_count = int(total_count * test_ratio)
    val_count = int(total_count * val_ratio)

    test_idx = indices[:test_count]
    val_idx = indices[test_count : test_count + val_count]
    train_idx = indices[test_count + val_count :]

    def gather(idx_list: list[int]) -> tuple[list[str], list[int]]:
        return [paths[i] for i in idx_list], [labels[i] for i in idx_list]

    train_paths, train_labels = gather(train_idx)
    val_paths, val_labels = gather(val_idx)
    test_paths, test_labels = gather(test_idx)

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def _decode_image(path: tf.Tensor, image_size: tuple[int, int], channels: int) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=channels, expand_animations=False)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def build_autoencoder_dataset(
    paths: Sequence[str],
    batch_size: int,
    image_size: tuple[int, int] = (128, 128),
    channels: int = 1,
    shuffle: bool = False,
    noise_stddev: float = 0.0,
    cache: bool = False,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(list(paths))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p: _decode_image(p, image_size=image_size, channels=channels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if noise_stddev > 0.0:
        def add_noise(x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=noise_stddev)
            noisy = tf.clip_by_value(x + noise, 0.0, 1.0)
            return noisy, x

        ds = ds.map(add_noise, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_embedding_dataset(
    paths: Sequence[str],
    labels: Sequence[int],
    batch_size: int,
    image_size: tuple[int, int] = (128, 128),
    channels: int = 1,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    ds = ds.map(
        lambda p, y: (_decode_image(p, image_size=image_size, channels=channels), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def prepare_splits(
    data_root: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> SplitData:
    paths, labels, class_names = discover_image_paths(data_root)
    (
        train_paths,
        train_labels,
        val_paths,
        val_labels,
        test_paths,
        test_labels,
    ) = split_dataset(paths, labels, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    return SplitData(
        train_paths=train_paths,
        train_labels=train_labels,
        val_paths=val_paths,
        val_labels=val_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        class_names=class_names,
    )
