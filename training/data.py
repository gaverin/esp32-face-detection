import argparse
import json
import math
import shutil
from pathlib import Path
from random import Random

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_SOURCE_DIR = Path(__file__).resolve().parent / "data" / "og_data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "og_data_split"
DEFAULT_SEED = 42


def _list_class_images(class_dir: Path) -> list[Path]:
    return sorted(path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _compute_split_counts(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, int]:
    
    # check we have more than 3 images
    if total < 3:
        raise ValueError(
            f"Need at least 3 images per class to create train/val/test splits, got {total}."
        )
    
    # compute training, validationn and test sizes
    train_count = math.floor(total * train_ratio)
    remaining = total - train_count
    holdout_ratio = val_ratio + test_ratio
    val_share = val_ratio / holdout_ratio
    val_count = math.floor(remaining * val_share)
    test_count = remaining - val_count

    counts = {
        "train": train_count,
        "val": val_count,
        "test": test_count,
    }

    return counts


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}.")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must all be greater than zero.")


def prepare_dataset(
    source_dir: str | Path = DEFAULT_SOURCE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Create deterministic train/val/test folder splits for Keras loaders.

    The destination directory is deleted and rebuilt on every run so repeated
    runs with the same seed produce a clean, reproducible output tree.
    """

    _validate_ratios(train_ratio, val_ratio, test_ratio)

    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()

    # check source and destination directories
    if not source_path.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"Source dataset path is not a directory: {source_path}")
    # check dataset is not empty
    class_dirs = sorted(path for path in source_path.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found in {source_path}.")

    rng = Random(seed)
    class_images: dict[str, list[Path]] = {}
    for class_dir in class_dirs:
        images = _list_class_images(class_dir)
        if not images:
            raise ValueError(f"Class directory has no supported image files: {class_dir}")
        class_images[class_dir.name] = images

    # rebuild output directory
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "source_dir": str(source_path),
        "output_dir": str(output_path),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "classes": {},
    }

    # shuffle images, compute split counts
    for class_name, images in class_images.items():
        shuffled = list(images)
        rng.shuffle(shuffled)
        counts = _compute_split_counts(
            total=len(shuffled),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        train_end = counts["train"]
        val_end = train_end + counts["val"]
        split_map = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

        class_summary = {
            "total": len(images),
            "train": len(split_map["train"]),
            "val": len(split_map["val"]),
            "test": len(split_map["test"]),
        }
        summary["classes"][class_name] = class_summary
        
        # copy images into new folders
        for split_name, split_images in split_map.items():
            destination_dir = output_path / split_name / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_images:
                shutil.copy2(image_path, destination_dir / image_path.name)

    return summary


def rename_images(path):
    """
    Loop through subdirectories of the provided path. Each subdirectory
    represents a class for an ML problem.

    Rename all supported images in each subdirectory using the naming rule:
    <subdir_name>_0001.png, <subdir_name>_0002.png, and so on.
    """

    root_path = Path(path).resolve()

    if not root_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root_path}")

    class_dirs = sorted(class_dir for class_dir in root_path.iterdir() if class_dir.is_dir())
    summary: dict[str, int] = {}

    for class_dir in class_dirs:
        images = _list_class_images(class_dir)
        temp_paths: list[Path] = []

        for index, image_path in enumerate(images, start=1):
            temp_path = class_dir / f".__rename_images_tmp_{index:04d}{image_path.suffix.lower()}"
            image_path.rename(temp_path)
            temp_paths.append(temp_path)

        for index, temp_path in enumerate(temp_paths, start=1):
            target_path = class_dir / f"{class_dir.name}_{index:04d}.png"
            temp_path.rename(target_path)

        summary[class_dir.name] = len(images)

    return summary

def resize_images(path):
    source_path = Path(path).resolve()
    output_path = source_path.parent / f"{source_path.name}_224x224"
    target_size = (224, 224)

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {source_path}")

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}

    for class_dir in sorted(class_dir for class_dir in source_path.iterdir() if class_dir.is_dir()):
        destination_dir = output_path / class_dir.name
        destination_dir.mkdir(parents=True, exist_ok=True)

        processed_count = 0
        for image_path in _list_class_images(class_dir):
            destination_path = destination_dir / image_path.name

            with Image.open(image_path) as image:
                width, height = image.size
                if (width, height) == target_size:
                    shutil.copy2(image_path, destination_path)
                else:
                    if width < target_size[0] or height < target_size[1]:
                        raise ValueError(
                            f"Image is smaller than {target_size[0]}x{target_size[1]} and cannot be center-cropped: {image_path}"
                        )

                    left = (width - target_size[0]) // 2
                    top = (height - target_size[1]) // 2
                    cropped = image.crop((left, top, left + target_size[0], top + target_size[1]))
                    cropped.save(destination_path)

            processed_count += 1

        summary[class_dir.name] = processed_count

    return {
        "source_dir": str(source_path),
        "output_dir": str(output_path),
        "target_size": target_size,
        "classes": summary,
    }



if __name__ == "__main__":
    
    #rename_images("/Users/gab/repos/esp32-face-detection/training/data/og_data")
    #prepare_dataset()
    resize_images("/Users/gab/repos/esp32-face-detection/training/data/og_data")
