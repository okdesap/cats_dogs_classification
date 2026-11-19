"""
Build the folder structure expected by the other scripts from a cat-and-dog dataset:

    data/cats_and_dogs/
    ├── cats/
    └── dogs/

Supported source (ZIP or extracted folder): Kaggle "Cats and Dogs": PetImages/Cat and PetImages/Dog folders containing 0.jpg, 1.jpg, ...

Select N images randomly per class using a fixed seed. Broken images are skipped.
    N = 1000  -> 2000 images (how does performance change when using only 2000 instead of 25000 images?)
    N = 0     -> all images

Example:
    python prepare.py --src ~/Downloads/PetImages --n 0 --out data/cats_and_dogs_full
"""
import argparse
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


def find_images(root):
    """Return {"cats": [...], "dogs": [...]} for either supported layout."""
    all_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    # Layout A: filenames start with cat. / dog.
    a = {
        "cats": [p for p in all_files if p.name.lower().startswith("cat.")],
        "dogs": [p for p in all_files if p.name.lower().startswith("dog.")],
    }
    if a["cats"] and a["dogs"]:
        return a

    # Layout B: folders are named Cat/Dog or cats/dogs, regardless of case.
    b = {
        "cats": [p for p in all_files if any(part.lower() in ("cat", "cats") for part in p.parts)],
        "dogs": [p for p in all_files if any(part.lower() in ("dog", "dogs") for part in p.parts)],
    }
    return b


def is_valid_image(path):
    """Check whether the image can be fully loaded, as ImageFolder will load it later."""
    try:
        with Image.open(path) as im:
            im.convert("RGB")
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert a cat-and-dog dataset to ImageFolder structure")
    parser.add_argument("--src", required=True, help="ZIP file or folder (e.g. PetImages or an extracted train folder)")
    parser.add_argument("--out", default="data/cats_and_dogs", help="Output folder (default: data/cats_and_dogs)")
    parser.add_argument("--n", type=int, default=1000, help="Images per class, 0 = all (default: 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Selection seed (default: 42)")
    args = parser.parse_args()

    src = Path(args.src).expanduser()
    out = Path(args.out)
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    with tempfile.TemporaryDirectory() as tmp:
        if src.suffix.lower() == ".zip":
            print("Extracting ZIP ...")
            with zipfile.ZipFile(src) as z:
                z.extractall(tmp)
            root = Path(tmp)
        else:
            root = src

        classes = find_images(root)
        for name, files in classes.items():
            if not files:
                raise SystemExit(
                    f"No images found for '{name}' in {src}. "
                    "Expected: cat.*.jpg/dog.*.jpg or Cat/ and Dog/ folders."
                )

        rng = random.Random(args.seed)
        for name, files in classes.items():
            files = sorted(files)
            rng.shuffle(files)
            destination = out / name
            destination.mkdir(parents=True, exist_ok=True)

            copied, broken = 0, 0
            for f in files:
                if args.n > 0 and copied >= args.n:
                    break
                if not is_valid_image(f):
                    broken += 1
                    continue
                shutil.copy(f, destination / f.name)
                copied += 1

            if args.n > 0 and copied < args.n:
                raise SystemExit(f"Only {copied} valid images found for '{name}'; requested: {args.n}")
            print(f"{name}: {copied} images -> {destination} ({broken} broken images skipped)")

    print("Done.")


if __name__ == "__main__":
    main()