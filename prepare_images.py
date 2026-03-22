'''
AI Assistance Disclosure:
ChatGPT (OpenAI) was used to help generate example Python code for advanced
pre-processing of wildlife images for improved training on a smaller image set,
and automatically split images into test, training, and validation data sets.
'''

import argparse
import random
import shutil
from pathlib import Path

# Image manipulation (Pillow)
# https://pillow.readthedocs.io/en/stable/
from PIL import Image, UnidentifiedImageError
import imagehash

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".jfif"}

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS

def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False

def compute_hash(path: Path, hash_size: int = 16):
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return imagehash.phash(img, hash_size=hash_size)
    except Exception:
        return None

def gather_class_files(source_dir: Path):
    class_map = {}
    for class_dir in sorted(source_dir.iterdir()):
        if class_dir.is_dir():
            files = [p for p in sorted(class_dir.iterdir()) if is_image_file(p)]
            class_map[class_dir.name] = files
    return class_map

def remove_duplicates_for_class(
    class_name: str,
    files,
    duplicate_threshold: int,
    min_width: int,
    min_height: int,
):
    kept = []
    removed_duplicates = []
    removed_invalid = []

    seen_hashes = []

    for path in files:
        if not verify_image(path):
            removed_invalid.append(path)
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
        except Exception:
            removed_invalid.append(path)
            continue

        if width < min_width or height < min_height:
            removed_invalid.append(path)
            continue

        img_hash = compute_hash(path)
        if img_hash is None:
            removed_invalid.append(path)
            continue

        is_duplicate = False
        for existing_hash, existing_path in seen_hashes:
            if abs(img_hash - existing_hash) <= duplicate_threshold:
                removed_duplicates.append(path)
                is_duplicate = True
                break

        if not is_duplicate:
            seen_hashes.append((img_hash, path))
            kept.append(path)

    print(
        f"{class_name}: kept={len(kept)}, "
        f"duplicates_removed={len(removed_duplicates)}, "
        f"invalid_removed={len(removed_invalid)}"
    )
    return kept, removed_duplicates, removed_invalid


def split_files(files, train_ratio, val_ratio, test_ratio, seed):
    if not files:
        return [], [], []

    if round(train_ratio + val_ratio + test_ratio, 6) != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    files = files[:]
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files


def copy_split(files, split_name: str, class_name: str, output_dir: Path):
    split_class_dir = output_dir / split_name / class_name
    split_class_dir.mkdir(parents=True, exist_ok=True)

    for src in files:
        dst = split_class_dir / src.name

        counter = 1
        while dst.exists():
            dst = split_class_dir / f"{src.stem}_{counter}{src.suffix}"
            counter += 1

        shutil.copy2(src, dst)


def write_log(log_path: Path, lines):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Remove duplicate wildlife images and split into train/val/test."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="inat_dataset",
        help="Source dataset root with one folder per class",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output dataset root",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training split ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=int,
        default=6,
        help="Max perceptual hash distance to treat as duplicate",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=224,
        help="Minimum allowed image width",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=224,
        help="Minimum allowed image height",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42, # Random seed 42.
        help="Random seed",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if output_dir.exists():
        print(f"Output directory already exists: {output_dir}")
        print("New files may be added into existing folders.")

    class_map = gather_class_files(source_dir)
    if not class_map:
        raise ValueError(f"No class folders found in: {source_dir}")

    all_duplicate_lines = []
    all_invalid_lines = []
    summary_lines = []

    print("\n--- Removing duplicates and invalid images ---")
    cleaned_class_map = {}

    for class_name, files in class_map.items():
        kept, removed_duplicates, removed_invalid = remove_duplicates_for_class(
            class_name=class_name,
            files=files,
            duplicate_threshold=args.duplicate_threshold,
            min_width=args.min_width,
            min_height=args.min_height,
        )
        cleaned_class_map[class_name] = kept

        for p in removed_duplicates:
            all_duplicate_lines.append(f"{class_name},{p}")
        for p in removed_invalid:
            all_invalid_lines.append(f"{class_name},{p}")

    print("\n--- Splitting dataset ---")
    for class_name, files in cleaned_class_map.items():
        train_files, val_files, test_files = split_files(
            files=files,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        copy_split(train_files, "train", class_name, output_dir)
        copy_split(val_files, "val", class_name, output_dir)
        copy_split(test_files, "test", class_name, output_dir)

        line = (
            f"{class_name}: total={len(files)}, "
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )
        summary_lines.append(line)
        print(line)

    logs_dir = output_dir / "_logs"
    write_log(logs_dir / "duplicates_removed.txt", all_duplicate_lines)
    write_log(logs_dir / "invalid_removed.txt", all_invalid_lines)
    write_log(logs_dir / "summary.txt", summary_lines)

    print("\nDone.")
    print(f"Prepared dataset saved to: {output_dir}")
    print(f"Logs saved to: {logs_dir}")


if __name__ == "__main__":
    main()