import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import json
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, RandomFlip, RandomRotation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split

IMG_SIZE = 224

X_structured = np.load("X_time_windows.npy", mmap_mode="r")
y_structured = np.load("y_labels.npy", allow_pickle=True)
X_images = np.load("X_images.npy", mmap_mode="r")
y_images = np.load("y_image_labels.npy")
image_classes = np.load("image_classes.npy", allow_pickle=True)

structured_classes = np.unique(y_structured)
allowed_classes = [c for c in image_classes if c in set(structured_classes)]
missing = [c for c in image_classes if c not in set(structured_classes)]
if missing:
    print(f"Dropping image classes not in structured labels: {missing}")

labels_str = image_classes[y_images]
mask = np.isin(labels_str, allowed_classes)
if not np.all(mask):
    keep = np.where(mask)[0]
    X_f = np.lib.format.open_memmap(
        "X_images_filtered.npy",
        mode="w+",
        dtype=X_images.dtype,
        shape=(keep.size, *X_images.shape[1:])
    )
    y_f = np.empty(keep.size, dtype=y_images.dtype)
    for i, src in enumerate(keep, start=1):
        X_f[i - 1] = X_images[src]
        y_f[i - 1] = y_images[src]
        if i % 500 == 0:
            print(f"Filtered images: {i}/{keep.size}")
    X_f.flush()
    X_images, y_images = X_f, y_f
    labels_str = image_classes[y_images]

image_classes = np.array(allowed_classes, dtype=object)
class_to_idx = {c: i for i, c in enumerate(image_classes)}
y = np.array([class_to_idx[c] for c in labels_str], dtype="int32")
num_classes = len(image_classes)

rng = np.random.default_rng(42)
X_structured_aligned = np.lib.format.open_memmap(
    "X_structured_aligned.npy",
    mode="w+",
    dtype=X_structured.dtype,
    shape=(X_images.shape[0], X_structured.shape[1], X_structured.shape[2])
)

for cls in image_classes:
    img_idx = np.where(labels_str == cls)[0]
    struct_idx = np.where(y_structured == cls)[0]
    if struct_idx.size == 0:
        raise ValueError(f"No structured samples for class '{cls}'.")
    choice = rng.choice(struct_idx, size=img_idx.size, replace=struct_idx.size < img_idx.size)
    X_structured_aligned[img_idx] = X_structured[choice]

X_structured_aligned.flush()
del X_structured

skip_upsample = os.getenv('SKIP_UPSAMPLE', '0') == '1'
counts = [int(np.sum(y == i)) for i in range(num_classes)]
max_count = max(counts) if counts else 0
if max_count > 0 and not all(c == max_count for c in counts):
    print(f"Upsampling classes to {max_count} samples each (was {counts})")
if skip_upsample:
    print("SKIP_UPSAMPLE=1 detected -- skipping on-disk upsampling. Training will use class_weight only.")
else:
    rng = np.random.default_rng(42)
    total = max_count * num_classes
    h, w, ch = X_images.shape[1:]
    s1, s2 = X_structured_aligned.shape[1:]
    X_images_up = np.lib.format.open_memmap(
        "X_images_upsampled.npy",
        mode="w+",
        dtype=X_images.dtype,
        shape=(total, h, w, ch)
    )
    X_struct_up = np.lib.format.open_memmap(
        "X_structured_aligned_upsampled.npy",
        mode="w+",
        dtype=X_structured_aligned.dtype,
        shape=(total, s1, s2)
    )
    y_up = np.empty((total,), dtype=y.dtype)

    pos = 0
    for i in range(num_classes):
        idx = np.where(y == i)[0]
        if idx.size == 0:
            for k in range(max_count):
                X_images_up[pos + k] = np.zeros((h, w, ch), dtype=X_images.dtype)
                X_struct_up[pos + k] = np.zeros((s1, s2), dtype=X_structured_aligned.dtype)
                y_up[pos + k] = i
            pos += max_count
            continue
        choice = rng.choice(idx, size=max_count, replace=(idx.size < max_count))
        for k, src in enumerate(choice):
            X_images_up[pos + k] = X_images[src]
            X_struct_up[pos + k] = X_structured_aligned[src]
            y_up[pos + k] = i
        pos += max_count

    X_images_up.flush()
    X_struct_up.flush()
    X_images, X_structured_aligned, y = X_images_up, X_struct_up, y_up
    print(
        f"Upsampled dataset shape: images={X_images.shape}, structured={X_structured_aligned.shape}, y={y.shape}"
    )

X_img_train, X_img_test, X_str_train, X_str_test, y_train, y_test = train_test_split(
    X_images, X_structured_aligned, y, test_size=0.2, random_state=42
)

cnn_backbone = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg"
)
cnn_backbone.trainable = False

def _patch_keras_batch_shape(path):
    path = Path(path)
    patched_path = path.with_name(f"{path.stem}_patched{path.suffix}")
    with zipfile.ZipFile(path, "r") as src:
        if "config.json" not in src.namelist():
            raise FileNotFoundError("config.json not found in .keras model archive")
        config = json.loads(src.read("config.json").decode("utf-8"))

    def _replace(obj):
        if isinstance(obj, dict):
            if "batch_shape" in obj and "batch_input_shape" not in obj:
                obj["batch_input_shape"] = obj.pop("batch_shape")
            if "dtype" in obj and isinstance(obj["dtype"], dict):
                dtype_cfg = obj["dtype"]
                if dtype_cfg.get("class_name") in ("DTypePolicy", "Policy") and isinstance(dtype_cfg.get("config"), dict):
                    name = dtype_cfg["config"].get("name")
                    if isinstance(name, str):
                        obj["dtype"] = name
            if "dtype_policy" in obj and isinstance(obj["dtype_policy"], dict):
                dtype_cfg = obj["dtype_policy"]
                if dtype_cfg.get("class_name") in ("DTypePolicy", "Policy") and isinstance(dtype_cfg.get("config"), dict):
                    name = dtype_cfg["config"].get("name")
                    if isinstance(name, str):
                        obj["dtype_policy"] = name
            for value in obj.values():
                _replace(value)
        elif isinstance(obj, list):
            for value in obj:
                _replace(value)

    _replace(config)

    with zipfile.ZipFile(path, "r") as src, zipfile.ZipFile(patched_path, "w") as dst:
        for item in src.infolist():
            if item.filename == "config.json":
                dst.writestr(item, json.dumps(config))
            else:
                dst.writestr(item, src.read(item.filename))
    return str(patched_path)


def _load_bilstm_model(path):
    try:
        return load_model(path, safe_mode=False, compile=False)
    except Exception as e:
        msg = str(e)
        if not any(token in msg for token in ("batch_shape", "DTypePolicy", "dtype_policy", "mixed_precision", "dtype")):
            raise
        print("Retrying bilstm load with config fix...", flush=True)
        patched = _patch_keras_batch_shape(path)
        return load_model(patched, safe_mode=False, compile=False)


bilstm_model = _load_bilstm_model("bilstm_traffic_model.keras")


def _get_input(model, fallback_shape=None):
    if getattr(model, "inputs", None):
        return model.inputs[0]
    if fallback_shape is not None:
        model.build(fallback_shape)
        return model.inputs[0]
    raise ValueError("Model has no defined inputs.")


bilstm_input = _get_input(
    bilstm_model,
    fallback_shape=(None, X_structured_aligned.shape[1], X_structured_aligned.shape[2])
)
bilstm_feature_extractor = Model(
    inputs=bilstm_input, outputs=bilstm_model.layers[-2].output
)
bilstm_feature_extractor.trainable = False

if hasattr(tf.keras.layers, "RandomBrightness"):
    brightness_layer = tf.keras.layers.RandomBrightness(0.1)
else:
    def _rand_brightness(x):
        x = tf.image.random_brightness(x, max_delta=0.1)
        return tf.clip_by_value(x, 0.0, 255.0)

    brightness_layer = tf.keras.layers.Lambda(
        _rand_brightness, name="random_brightness"
    )

image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
aug = RandomFlip("horizontal")(image_input)
aug = RandomRotation(0.04)(aug)
aug = brightness_layer(aug)
aug = tf.keras.layers.Lambda(lambda x: x * 255.0, name="rescale_255")(aug)
aug = preprocess_input(aug)
cnn_features = cnn_backbone(aug, training=False)

structured_input = Input(
    shape=(X_structured_aligned.shape[1], X_structured_aligned.shape[2])
)
lstm_features = bilstm_feature_extractor(structured_input)

x = Concatenate()([cnn_features, lstm_features])
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

multimodal_model = Model(inputs=[image_input, structured_input], outputs=output)
multimodal_model.compile(
    optimizer=Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
multimodal_model.summary()

if __name__ == '__main__':
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

    classes_unique = np.unique(y_train)
    cw = compute_class_weight(class_weight='balanced', classes=classes_unique, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes_unique, cw)}
    print("Class Weights:", class_weights)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
    )
    checkpoint = ModelCheckpoint(
        'multimodal_traffic_model_best.keras', save_best_only=True, monitor='val_loss'
    )

    fine_tune_epochs = int(os.getenv("FINE_TUNE_EPOCHS", "0"))
    fine_tune_at = int(os.getenv("FINE_TUNE_AT", "0"))
    if fine_tune_epochs > 0:
        cnn_backbone.trainable = True
        if fine_tune_at > 0:
            for layer in cnn_backbone.layers[:fine_tune_at]:
                layer.trainable = False
        multimodal_model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    multimodal_model.fit(
        [X_img_train, X_str_train],
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[reduce_lr, checkpoint]
    )

    loss, accuracy = multimodal_model.evaluate(
        [X_img_test, X_str_test], y_test, batch_size=32
    )
    print("Multimodal Test Accuracy:", accuracy)

    y_pred = np.argmax(
        multimodal_model.predict([X_img_test, X_str_test], batch_size=32, verbose=0),
        axis=1
    )

    try:
        classes = np.load('image_classes.npy', allow_pickle=True)
        print(classification_report(y_test, y_pred, target_names=classes))
    except Exception:
        print(classification_report(y_test, y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=classes_unique, zero_division=0
    )
    recall_map = {int(c): r for c, r in zip(classes_unique, recall)}
    print("Per-class recall:", recall_map)

    thresholds = {'Emergency': 0.80, 'Normal': 0.80}
    thresholds_met = True
    try:
        class_names = np.load('image_classes.npy', allow_pickle=True)
        for cname, thresh in thresholds.items():
            if cname in class_names:
                idx = int(np.where(class_names == cname)[0][0])
                r = recall_map.get(idx, 0.0)
                print(f"Recall for {cname}: {r:.3f} (target {thresh})")
                if r < thresh:
                    thresholds_met = False
            else:
                print(f"Class {cname} not found in image_classes.npy")
    except Exception:
        thresholds_met = False

    if accuracy < 0.85:
        print(f"Warning: Accuracy {accuracy:.3f} below desired 0.85")
        thresholds_met = False

    if thresholds_met:
        print("Performance targets met: Emergency/Normal recall >0.80 and accuracy>0.85")
    else:
        print("Performance targets NOT met. Consider more training, data augmentation, or class rebalancing.")

    multimodal_model.save("multimodal_traffic_model.keras")
    print("Multimodal model saved successfully!")
