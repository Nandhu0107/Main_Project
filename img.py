import os
import cv2
import numpy as np
from numpy.lib.format import open_memmap

IMG_SIZE = 224
data_root = "processed_images"
splits = ["train", "val", "test"]

valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

classes_set = set()
for split in splits:
    split_path = os.path.join(data_root, split)
    if not os.path.isdir(split_path):
        continue
    for d in os.listdir(split_path):
        p = os.path.join(split_path, d)
        if os.path.isdir(p):
            classes_set.add(d)

classes = sorted(classes_set)
print(
    f"Found {len(classes)} classes under {data_root} (splits: {', '.join(s for s in splits if os.path.isdir(os.path.join(data_root, s)))})"
)

image_items = []
for split in splits:
    split_path = os.path.join(data_root, split)
    if not os.path.isdir(split_path):
        continue
    for class_name in classes:
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            ext = os.path.splitext(img_name)[1].lower()
            if ext in valid_exts:
                label = classes.index(class_name)
                image_items.append((os.path.join(class_path, img_name), label))

total = len(image_items)
print(f"Found {total} images across splits")

tmp_x = "X_images_tmp.npy"
tmp_y = "y_image_labels_tmp.npy"
X_mm = open_memmap(tmp_x, mode="w+", dtype="float32", shape=(total, IMG_SIZE, IMG_SIZE, 3))
y_mm = open_memmap(tmp_y, mode="w+", dtype="int32", shape=(total,))

skipped = 0
write_idx = 0
for idx, (img_path, label) in enumerate(image_items, start=1):
    img = cv2.imread(img_path)
    if img is None:
        skipped += 1
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0

    X_mm[write_idx] = img
    y_mm[write_idx] = label
    write_idx += 1

    if idx % 500 == 0:
        print(f"Processed {idx}/{total}")

X_mm.flush()
y_mm.flush()
del X_mm, y_mm

if skipped > 0:
    X_mm = np.load(tmp_x, mmap_mode="r")
    y_mm = np.load(tmp_y, mmap_mode="r")
    np.save("X_images.npy", X_mm[:write_idx])
    np.save("y_image_labels.npy", y_mm[:write_idx])
    del X_mm, y_mm
    os.remove(tmp_x)
    os.remove(tmp_y)
else:
    os.replace(tmp_x, "X_images.npy")
    os.replace(tmp_y, "y_image_labels.npy")

np.save("image_classes.npy", np.array(classes, dtype=object))

print(f"X_images.npy and y_image_labels.npy created successfully! Skipped: {skipped}")
