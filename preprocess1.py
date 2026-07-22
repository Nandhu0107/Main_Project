import os
import cv2
import random
from tqdm import tqdm

RAW_DIR = "raw_images"
OUT_DIR = "processed_images"
IMG_SIZE = (224, 224)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

def is_image(file):
    return file.lower().endswith(VALID_EXTENSIONS)

def collect_images():
    data = []
    for label in ["Emergency", "Normal"]:
        folder = os.path.join(RAW_DIR, label)
        for img in os.listdir(folder):
            if is_image(img):
                data.append((os.path.join(folder, img), label))
    return data

def prepare_dirs():
    for split in ["train", "val", "test"]:
        for label in ["Emergency", "Normal"]:
            os.makedirs(os.path.join(OUT_DIR, split, label), exist_ok=True)

def preprocess(data):
    random.shuffle(data)
    total = len(data)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": data[:train_end],
        "val": data[train_end:val_end],
        "test": data[val_end:]
    }

    for split, items in splits.items():
        for path, label in tqdm(items, desc=f"{split}"):
            img = cv2.imread(path)
            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            img = img / 255.0

            save_path = os.path.join(
                OUT_DIR, split, label, os.path.basename(path)
            )
            cv2.imwrite(save_path, (img * 255).astype("uint8"))

if __name__ == "__main__":
    data = collect_images()
    prepare_dirs()
    preprocess(data)
    print("Image preprocessing COMPLETED")