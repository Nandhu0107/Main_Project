import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

train_dir = "processed_images/train"
val_dir = "processed_images/val"
test_dir = "processed_images/test"

use_upsampling = os.getenv("USE_UPSAMPLING", "0") == "1"

train_base_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=None,
    label_mode="int",
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

num_classes = len(train_base_ds.class_names)

if hasattr(tf.keras.layers, "RandomBrightness"):
    brightness_layer = tf.keras.layers.RandomBrightness(0.1)
else:
    def _rand_brightness(x):
        x = tf.image.random_brightness(x, max_delta=0.1)
        return tf.clip_by_value(x, 0.0, 255.0)

    brightness_layer = tf.keras.layers.Lambda(_rand_brightness, name="random_brightness")

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.04),
        brightness_layer,
    ],
    name="augmentation"
)

class_counts = np.zeros(num_classes, dtype=np.int64)
for _, labels in train_base_ds:
    labels_np = np.atleast_1d(labels.numpy())
    class_counts += np.bincount(labels_np, minlength=num_classes)

train_size = int(class_counts.sum())
class_weights = {
    i: float(train_size / (num_classes * c)) if c > 0 else 0.0
    for i, c in enumerate(class_counts)
}

print(f"Class counts: {class_counts.tolist()}")
print(f"Class weights: {class_weights}")

if use_upsampling:
    per_class_datasets = []
    for i in range(num_classes):
        ds_i = train_base_ds.filter(lambda x, y, i=i: tf.equal(y, i)).repeat()
        per_class_datasets.append(ds_i)
    weights = [1.0 / num_classes] * num_classes
    train_ds = tf.data.Dataset.sample_from_datasets(
        per_class_datasets,
        weights=weights,
        seed=42
    )
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
else:
    train_ds = train_base_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg"
)
base_model.trainable = False

inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(patience=5, restore_best_weights=True)

epochs = int(os.getenv("EPOCHS", "25"))
fine_tune_epochs = int(os.getenv("FINE_TUNE_EPOCHS", "5"))
fine_tune_at = int(os.getenv("FINE_TUNE_AT", "0"))
steps = None
val_steps = None
if os.getenv("SMOKE_TEST"):
    epochs = 1
    fine_tune_epochs = 1
    steps = 2
    val_steps = 1

fit_kwargs = {}
if steps is not None:
    fit_kwargs["steps_per_epoch"] = steps
elif use_upsampling:
    fit_kwargs["steps_per_epoch"] = max(1, math.ceil(train_size / BATCH_SIZE))
if val_steps is not None:
    fit_kwargs["validation_steps"] = val_steps

if os.getenv("SMOKE_TEST"):
    train_ds_run = train_ds.take(steps)
    val_ds_run = val_ds.take(val_steps)
else:
    train_ds_run = train_ds
    val_ds_run = val_ds

model.fit(
    train_ds_run,
    validation_data=val_ds_run,
    epochs=epochs,
    callbacks=[early_stop],
    class_weight=None if use_upsampling else class_weights,
    **fit_kwargs
)

if fine_tune_epochs > 0:
    base_model.trainable = True
    if fine_tune_at > 0:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_ds_run,
        validation_data=val_ds_run,
        epochs=fine_tune_epochs,
        callbacks=[early_stop],
        class_weight=None if use_upsampling else class_weights,
        **fit_kwargs
    )

if os.getenv("SMOKE_TEST"):
    test_ds_run = test_ds.take(1)
else:
    test_ds_run = test_ds

loss, accuracy = model.evaluate(test_ds_run)
print("CNN Test Accuracy:", accuracy)

model.save("cnn_traffic_model.keras")
print("CNN Model Saved Successfully!")
