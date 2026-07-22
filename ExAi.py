import os
import sys
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_keras as keras

sys.modules["keras"] = keras

import shap
from main_system import build_decision_summary


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "exai_outputs"
IMAGE_SHAPE = (224, 224, 3)
CLASS_NAMES = ("Emergency", "Normal")
SHAP_BACKGROUND_CACHE = BASE_DIR / "image_feature_background.npy"
SHAP_BACKGROUND_SAMPLES = 24
SHAP_MAX_EVALS = 220
IMAGE_STRUCTURED_FEATURE_NAMES = [
    "Brightness",
    "Contrast",
    "Saturation",
    "Edge Density",
    "Red Dominance",
    "Blue Dominance",
    "Green Dominance",
    "Dark Pixel Ratio",
    "Bright Pixel Ratio",
    "Texture Strength",
]

_MODEL = None
_CLASS_NAMES = None


def resolve_model_path():
    candidates = [
        BASE_DIR / "model.h5",
        BASE_DIR / "multimodal_traffic_model.h5",
        BASE_DIR / "multimodal_traffic_model.keras",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("No supported multimodal model file found in the project root.")


def load_model():
    global _MODEL
    if _MODEL is None:
        model_path = resolve_model_path()
        print(f"Loading model from: {model_path.name}")
        _MODEL = keras.models.load_model(model_path, compile=False)
    return _MODEL


def load_dataset():
    images = np.load(BASE_DIR / "X_images.npy").astype(np.float32)
    structured = np.load(BASE_DIR / "X_structured_aligned.npy").astype(np.float32)
    labels = np.load(BASE_DIR / "y_image_labels.npy")
    return images, structured, labels, load_class_names()


def load_class_names():
    global _CLASS_NAMES
    if _CLASS_NAMES is None:
        class_path = BASE_DIR / "image_classes.npy"
        if class_path.exists():
            _CLASS_NAMES = tuple(np.load(class_path, allow_pickle=True).tolist())
        else:
            _CLASS_NAMES = CLASS_NAMES
    return _CLASS_NAMES


def _extract_band_features(band_rgb):
    band_uint8 = np.clip(band_rgb * 255.0, 0, 255).astype(np.uint8)
    band_gray = cv2.cvtColor(band_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    band_hsv = cv2.cvtColor(band_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    edges = cv2.Canny(band_uint8, 80, 160)
    laplacian = cv2.Laplacian(band_gray, cv2.CV_32F)

    r = band_rgb[:, :, 0]
    g = band_rgb[:, :, 1]
    b = band_rgb[:, :, 2]

    brightness = float(np.mean(band_gray))
    contrast = float(np.std(band_gray))
    saturation = float(np.mean(band_hsv[:, :, 1] / 255.0))
    edge_density = float(np.mean(edges > 0))
    red_dominance = float(np.mean(np.clip(r - ((g + b) * 0.5), 0.0, 1.0)))
    blue_dominance = float(np.mean(np.clip(b - ((r + g) * 0.5), 0.0, 1.0)))
    green_dominance = float(np.mean(np.clip(g - ((r + b) * 0.5), 0.0, 1.0)))
    dark_pixel_ratio = float(np.mean(band_gray < 0.25))
    bright_pixel_ratio = float(np.mean(band_gray > 0.75))
    texture_strength = float(np.clip(np.std(laplacian) / 0.25, 0.0, 1.0))

    return [
        brightness,
        contrast,
        saturation,
        edge_density,
        red_dominance,
        blue_dominance,
        green_dominance,
        dark_pixel_ratio,
        bright_pixel_ratio,
        texture_strength,
    ]


def extract_image_structured_features(image):
    image_rgb = image[0] if image.ndim == 4 else image
    bands = np.array_split(image_rgb, 5, axis=0)
    feature_rows = [_extract_band_features(band) for band in bands]
    feature_tensor = np.asarray(feature_rows, dtype=np.float32)
    return np.expand_dims(np.clip(feature_tensor, 0.0, 1.0), axis=0)


def preprocess_uploaded_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Invalid image file: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def predict_image(model, image, structured, class_names):
    probs = model.predict([image, structured], verbose=0)[0]
    pred_label = int(np.argmax(probs))
    return {
        "pred_probs": probs,
        "pred_label": pred_label,
        "predicted_name": class_names[pred_label],
        "predicted_confidence": float(probs[pred_label]),
    }


def choose_sample(model, images, structured, labels, scan_limit=128):
    batch_images = images[:scan_limit]
    batch_structured = structured[:scan_limit]
    batch_labels = labels[:scan_limit]

    preds = model.predict([batch_images, batch_structured], verbose=0)
    pred_ids = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    correct_mask = pred_ids == batch_labels

    if np.any(correct_mask):
        candidate_ids = np.where(correct_mask)[0]
        best_local = candidate_ids[np.argmax(confidences[candidate_ids])]
    else:
        best_local = int(np.argmax(confidences))

    return {
        "index": int(best_local),
        "image": batch_images[best_local : best_local + 1],
        "structured": batch_structured[best_local : best_local + 1],
        "true_label": int(batch_labels[best_local]),
        "pred_probs": preds[best_local],
        "pred_label": int(pred_ids[best_local]),
    }


def save_original_image(image, output_path):
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR))


def build_gradcam_heatmap(model, image, structured, last_conv_name="top_conv"):
    image_backbone = model.get_layer("efficientnetb0")
    structured_branch = model.get_layer("model")
    feature_extractor = keras.Model(
        image_backbone.input,
        [image_backbone.get_layer(last_conv_name).output, image_backbone.output],
    )

    image_tensor = tf.convert_to_tensor(image)
    structured_tensor = tf.convert_to_tensor(structured)

    with tf.GradientTape() as tape:
        x = model.get_layer("random_flip")(image_tensor, training=False)
        x = model.get_layer("random_rotation")(x, training=False)
        x = model.get_layer("random_brightness")(x, training=False)
        x = model.get_layer("rescale_255")(x)

        conv_outputs, image_features = feature_extractor(x, training=False)
        tape.watch(conv_outputs)

        structured_features = structured_branch(structured_tensor, training=False)
        fused = model.get_layer("concatenate")([image_features, structured_features])
        fused = model.get_layer("dense")(fused, training=False)
        fused = model.get_layer("dropout")(fused, training=False)
        preds = model.get_layer("dense_1")(fused, training=False)

        class_index = tf.argmax(preds[0])
        class_score = preds[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(class_index.numpy())


def save_heatmap_figure(heatmap, output_path, title):
    plt.figure(figsize=(7, 6))
    plt.imshow(heatmap, cmap="jet")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_overlay(image, heatmap, output_path, alpha=0.45):
    image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    resized_heatmap = cv2.resize(heatmap, (image_uint8.shape[1], image_uint8.shape[0]))
    normalized = cv2.normalize(resized_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_map = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(
        cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR),
        1.0 - alpha,
        color_map,
        alpha,
        0,
    )
    cv2.imwrite(str(output_path), overlay)


def summarize_heatmap(heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    normalized = heatmap / (np.max(heatmap) + 1e-8)
    active_mask = normalized >= 0.6
    active_fraction = float(np.mean(active_mask))
    peak_strength = float(np.max(normalized))

    y_idx, x_idx = np.indices(normalized.shape)
    total = float(np.sum(normalized))
    if total <= 1e-8:
        x_center = 0.5
        y_center = 0.5
    else:
        x_center = float(np.sum(x_idx * normalized) / total) / max(1, normalized.shape[1] - 1)
        y_center = float(np.sum(y_idx * normalized) / total) / max(1, normalized.shape[0] - 1)

    if y_center < 0.33:
        vertical = "upper"
    elif y_center > 0.66:
        vertical = "lower"
    else:
        vertical = "central"

    if x_center < 0.33:
        horizontal = "left"
    elif x_center > 0.66:
        horizontal = "right"
    else:
        horizontal = "center"

    if vertical == "central" and horizontal == "center":
        region = "center"
    elif vertical == "central":
        region = f"{horizontal}-center"
    elif horizontal == "center":
        region = f"{vertical}-center"
    else:
        region = f"{vertical}-{horizontal}"

    if active_fraction >= 0.28:
        spread = "broad"
    elif active_fraction >= 0.12:
        spread = "moderate"
    else:
        spread = "focused"

    return {
        "region": region,
        "spread": spread,
        "active_fraction": active_fraction,
        "peak_strength": peak_strength,
    }


def get_shap_background(structured_template, sample_size=SHAP_BACKGROUND_SAMPLES):
    expected_shape = tuple(structured_template.shape[1:])

    if SHAP_BACKGROUND_CACHE.exists():
        cached = np.load(SHAP_BACKGROUND_CACHE).astype(np.float32)
        if cached.ndim == 3 and tuple(cached.shape[1:]) == expected_shape:
            return cached, "cached image-feature background"

    image_path = BASE_DIR / "X_images.npy"
    if image_path.exists():
        images = np.load(image_path, mmap_mode="r")
        if len(images) > 0:
            sample_size = min(sample_size, len(images))
            indices = np.linspace(0, len(images) - 1, num=sample_size, dtype=int)
            background = np.stack(
                [extract_image_structured_features(images[idx : idx + 1])[0] for idx in indices],
                axis=0,
            ).astype(np.float32)
            try:
                np.save(SHAP_BACKGROUND_CACHE, background)
            except Exception:
                pass
            return background, "dataset-derived image-feature background"

    offsets = np.linspace(-0.18, 0.18, num=sample_size, dtype=np.float32)
    base = structured_template[0]
    background = np.stack([np.clip(base + offset, 0.0, 1.0) for offset in offsets], axis=0)
    return background.astype(np.float32), "synthetic local fallback background"


def predict_structured_logits(model, image, structured_batch):
    image_batch = np.repeat(image.astype(np.float32), len(structured_batch), axis=0)
    probs = model.predict([image_batch, structured_batch], verbose=0)
    return np.log(np.clip(probs, 1e-7, 1.0)).astype(np.float32)


def explain_structured_with_shap(model, image, structured, class_names):
    background, background_source = get_shap_background(structured)
    background_flat = background.reshape((background.shape[0], -1))
    sample_flat = structured.astype(np.float32).reshape(1, -1)

    def predict_structured_only(flat_structured):
        flat_structured = np.asarray(flat_structured, dtype=np.float32)
        structured_batch = flat_structured.reshape((-1, structured.shape[1], structured.shape[2]))
        return predict_structured_logits(model, image, structured_batch)

    explainer = shap.Explainer(
        predict_structured_only,
        background_flat,
        algorithm="permutation",
        feature_names=[f"band{t + 1}_{name}" for t in range(structured.shape[1]) for name in IMAGE_STRUCTURED_FEATURE_NAMES],
        output_names=list(class_names),
    )
    shap_values = explainer(sample_flat, max_evals=SHAP_MAX_EVALS)
    pred_logits = predict_structured_logits(model, image, structured)[0]
    pred_class = int(np.argmax(pred_logits))

    flat_values = shap_values.values[0, :, pred_class]
    value_matrix = flat_values.reshape(structured.shape[1], structured.shape[2])
    feature_signed = value_matrix.mean(axis=0)
    feature_importance = np.abs(value_matrix).mean(axis=0)
    band_signed = value_matrix.mean(axis=1)
    band_importance = np.abs(value_matrix).mean(axis=1)

    base_values = np.asarray(shap_values.base_values)
    if base_values.ndim == 2:
        base_value = float(base_values[0, pred_class])
    elif base_values.ndim == 1:
        base_value = float(base_values[pred_class])
    else:
        base_value = float(np.ravel(base_values)[0])

    return {
        "pred_class": pred_class,
        "pred_logit": float(pred_logits[pred_class]),
        "base_value": base_value,
        "feature_signed": feature_signed.astype(np.float32),
        "feature_importance": feature_importance.astype(np.float32),
        "band_signed": band_signed.astype(np.float32),
        "band_importance": band_importance.astype(np.float32),
        "total_importance": float(np.abs(flat_values).sum()),
        "background_source": background_source,
    }


def summarize_shap(feature_names, feature_importance, feature_signed, top_k=3):
    order = np.argsort(feature_importance)[::-1][:top_k]
    top_features = []
    for idx in order:
        signed_value = float(feature_signed[idx])
        top_features.append(
            {
                "name": feature_names[idx],
                "importance": float(feature_importance[idx]),
                "direction": "toward" if signed_value >= 0 else "away from",
                "signed_value": signed_value,
            }
        )
    return top_features


def summarize_bands(band_importance, band_signed, top_k=2):
    order = np.argsort(band_importance)[::-1][:top_k]
    return [
        {
            "name": f"band {int(idx) + 1}",
            "importance": float(band_importance[idx]),
            "direction": "toward" if float(band_signed[idx]) >= 0 else "away from",
        }
        for idx in order
    ]


def summarize_image_features(feature_matrix, top_k=4):
    band_means = np.mean(feature_matrix[0], axis=0)
    order = np.argsort(band_means)[::-1][:top_k]
    return [
        f"{IMAGE_STRUCTURED_FEATURE_NAMES[idx]}={band_means[idx]:.2f}"
        for idx in order
    ]


def format_shap_value(value):
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.4f}"


def build_shap_text_explainer(prediction, shap_summary, band_summary, shap_details):
    predicted_name = prediction["predicted_name"]
    confidence_text = f"{prediction['predicted_confidence'] * 100:.1f}%"
    total_importance = shap_details["total_importance"]
    base_value = shap_details["base_value"]
    pred_logit = shap_details["pred_logit"]
    influence_text = "very weak" if total_importance < 1e-4 else "moderate" if total_importance < 1e-2 else "strong"

    if not shap_summary:
        return (
            f"The SHAP analysis could not isolate strong image-derived feature drivers for the "
            f"{predicted_name} output node."
        )

    support = [item for item in shap_summary if item["direction"] == "toward"]
    oppose = [item for item in shap_summary if item["direction"] == "away from"]

    contribution_lines = []
    for item in shap_summary:
        direction_text = (
            f"increased the {predicted_name.lower()} score"
            if item["direction"] == "toward"
            else f"reduced the {predicted_name.lower()} score"
        )
        contribution_lines.append(
            f"{item['name']} {direction_text} (|SHAP|={format_shap_value(item['importance'])})"
        )

    band_text = ", ".join(
        f"{band['name']} pushed {band['direction']} {predicted_name.lower()} "
        f"(strength {format_shap_value(band['importance'])})"
        for band in band_summary
    ) if band_summary else "no image band stood out"

    support_text = ", ".join(item["name"] for item in support[:2]) if support else "none"
    oppose_text = ", ".join(item["name"] for item in oppose[:2]) if oppose else "none"

    return (
        f"SHAP text explanation for the image-derived feature branch: the prediction module "
        f"selected {predicted_name} with {confidence_text} confidence. "
        f"Relative to the SHAP baseline log-score {format_shap_value(base_value)}, the extracted "
        f"image features moved the {predicted_name.lower()} log-score to {format_shap_value(pred_logit)}. "
        f"Structured influence was {influence_text} overall, which means the final decision was "
        f"{'mostly driven by the visual branch' if total_importance < 1e-4 else 'materially supported by the feature branch'}. "
        f"Top supporting features: {support_text}. Top opposing features: {oppose_text}. "
        f"The most influential image bands were: {band_text}. "
        f"Detailed feature effects: {'; '.join(contribution_lines)}. "
        f"Background source: {shap_details['background_source']}."
    )


def build_text_explainer(prediction, decision, heatmap_summary, shap_summary):
    if shap_summary:
        shap_text = "; ".join(
            f"{item['name']} pushed {item['direction']} {prediction['predicted_name'].lower()}"
            for item in shap_summary
        )
    else:
        shap_text = "image-derived SHAP evidence was limited"

    confidence_text = f"{prediction['predicted_confidence'] * 100:.1f}% confidence"
    heatmap_text = (
        f"Grad-CAM attention was {heatmap_summary['spread']} and concentrated in the "
        f"{heatmap_summary['region']} region of the image"
    )

    return (
        f"The model predicted {decision['predicted_state']} with {confidence_text}. "
        f"{heatmap_text}. "
        f"The strongest image-derived signals were: {shap_text}. "
        f"Combined visual and image-derived feature evidence supports the recommended action: "
        f"{decision['recommended_action']}."
    )


def run_analysis_pipeline(model, image, prediction_structured, explanation_structured, class_names, output_dir):
    prediction = predict_image(model, image, prediction_structured, class_names)
    decision = build_decision_summary(
        prediction["predicted_name"],
        prediction["predicted_confidence"],
    )
    image_feature_summary = summarize_image_features(explanation_structured)

    save_original_image(image[0], output_dir / "sample_image.png")

    gradcam_heatmap, gradcam_class = build_gradcam_heatmap(model, image, prediction_structured)
    save_heatmap_figure(
        gradcam_heatmap,
        output_dir / "gradcam_heatmap.png",
        f"Grad-CAM Heatmap: {class_names[gradcam_class]}",
    )
    save_overlay(image[0], gradcam_heatmap, output_dir / "gradcam_overlay.png")

    heatmap_summary = summarize_heatmap(gradcam_heatmap)
    shap_details = explain_structured_with_shap(model, image, explanation_structured, class_names)
    shap_summary = summarize_shap(
        IMAGE_STRUCTURED_FEATURE_NAMES,
        shap_details["feature_importance"],
        shap_details["feature_signed"],
    )
    band_summary = summarize_bands(shap_details["band_importance"], shap_details["band_signed"])

    return {
        "prediction_details": prediction,
        "decision": decision,
        "prediction": decision["predicted_state"],
        "prediction_confidence": decision["prediction_confidence"],
        "congestion_level": decision["congestion_level"],
        "emergency_detected": decision["emergency_detected"],
        "recommended_action": decision["recommended_action"],
        "priority": decision["priority"],
        "reason": decision["reason"],
        "reference_context": "Image-derived features extracted from the uploaded frame",
        "image_feature_summary": ", ".join(image_feature_summary),
        "shap_text_explainer": build_shap_text_explainer(prediction, shap_summary, band_summary, shap_details),
        "text_explainer": build_text_explainer(prediction, decision, heatmap_summary, shap_summary),
        "heatmap_summary": heatmap_summary,
        "top_shap_features": shap_summary,
        "files": {
            "sample_image": output_dir / "sample_image.png",
            "gradcam_heatmap": output_dir / "gradcam_heatmap.png",
            "gradcam_overlay": output_dir / "gradcam_overlay.png",
        },
    }


def analyze_image(image_path, output_dir=None):
    model = load_model()
    class_names = load_class_names()
    image = preprocess_uploaded_image(image_path)
    structured = extract_image_structured_features(image)

    if output_dir is None:
        output_dir = OUTPUT_DIR / f"analysis_{uuid4().hex}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return run_analysis_pipeline(
        model,
        image,
        structured,
        structured,
        class_names,
        output_dir,
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading model...")
    model = load_model()
    print("Model loaded successfully")

    images, structured, labels, class_names = load_dataset()
    sample = choose_sample(model, images, structured, labels)

    print(f"Using dataset sample index: {sample['index']}")
    print(
        "Prediction:",
        f"{class_names[sample['pred_label']]} ({sample['pred_probs'][sample['pred_label']]:.4f})",
        f"| True label: {class_names[sample['true_label']]}",
    )
    derived_structured = extract_image_structured_features(sample["image"])
    analysis = run_analysis_pipeline(
        model,
        sample["image"],
        sample["structured"],
        derived_structured,
        class_names,
        OUTPUT_DIR,
    )
    print("Text explainer:")
    print(analysis["text_explainer"])
    print("SHAP text explainer:")
    print(analysis["shap_text_explainer"])

    print("Generated files:")
    print(str(analysis["files"]["sample_image"]))
    print(str(analysis["files"]["gradcam_heatmap"]))
    print(str(analysis["files"]["gradcam_overlay"]))


if __name__ == "__main__":
    main()
