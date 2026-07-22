import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import json
import zipfile

CONFIRMED_EMERGENCY_CONFIDENCE = 0.66


def ontology_reasoning(congestion_label, emergency_flag):
    if congestion_label == "High" and emergency_flag == 1:
        return {
            "action": "Activate Green Wave",
            "priority": "Critical",
            "reason": "High congestion with emergency vehicle detected",
        }
    if congestion_label == "High":
        return {
            "action": "Activate Green Wave",
            "priority": "High",
            "reason": "High congestion or network risk detected",
        }
    if congestion_label == "Medium":
        return {
            "action": "Adjust Signal Timing",
            "priority": "Moderate",
            "reason": "Moderate congestion detected",
        }
    return {
        "action": "Normal Monitoring",
        "priority": "Low",
        "reason": "Traffic operating normally",
    }


def build_decision_summary(predicted_label, predicted_confidence, confirmed_emergency_confidence=CONFIRMED_EMERGENCY_CONFIDENCE):
    predicted_label = str(predicted_label)
    predicted_confidence = float(predicted_confidence)

    if predicted_label == "Emergency" and predicted_confidence >= confirmed_emergency_confidence:
        return {
            "predicted_state": "Emergency",
            "prediction_confidence": predicted_confidence,
            "congestion_level": "High",
            "emergency_detected": 1,
            "recommended_action": "Activate Green Wave",
            "priority": "Critical",
            "reason": "Emergency condition detected. Priority signal handling is recommended.",
        }

    if predicted_label == "Emergency":
        return {
            "predicted_state": "High Congestion",
            "prediction_confidence": predicted_confidence,
            "congestion_level": "High",
            "emergency_detected": 0,
            "recommended_action": "Adjust Signal Timing",
            "priority": "High",
            "reason": "Heavy traffic congestion detected. Signal timing adjustment is recommended.",
        }

    return {
        "predicted_state": "Normal",
        "prediction_confidence": predicted_confidence,
        "congestion_level": "Low",
        "emergency_detected": 0,
        "recommended_action": "Normal Monitoring",
        "priority": "Low",
        "reason": "Traffic is operating normally.",
    }


def _load_model():
    if tf.io.gfile.exists("multimodal_savedmodel"):
        sm = tf.saved_model.load("multimodal_savedmodel")
        sig = sm.signatures.get("serving_default") or sm.signatures.get("serve")
        if sig is None:
            sig = list(sm.signatures.values())[0]
        return "savedmodel", sm, sig
    model_path = Path("multimodal_traffic_model.keras")
    try:
        m = tf.keras.models.load_model(str(model_path), safe_mode=False, compile=False)
        return "keras", m, None
    except Exception:
        patched_path = _patch_keras_batch_shape(model_path)
        m = tf.keras.models.load_model(str(patched_path), safe_mode=False, compile=False)
        return "keras", m, None


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
    return patched_path


MODEL_TYPE = None
model = None
_infer = None


def _ensure_model_loaded():
    global MODEL_TYPE, model, _infer
    if MODEL_TYPE is None:
        MODEL_TYPE, model, _infer = _load_model()


image_classes = np.load("image_classes.npy", allow_pickle=True)


def _expected_structured_shape():
    _ensure_model_loaded()
    if MODEL_TYPE == "savedmodel" and _infer is not None:
        args, kwargs = _infer.structured_input_signature
        specs = list(kwargs.values()) if kwargs else list(args)
        for spec in specs:
            shape = spec.shape.as_list() if spec.shape is not None else None
            if shape and len(shape) == 3 and shape[1:] != [224, 224, 3]:
                return tuple(shape[1:])
    if MODEL_TYPE == "keras":
        try:
            shape = model.input_shape[1]
            if shape:
                return tuple(shape[1:])
        except Exception:
            pass
    return (5, 10)


def preprocess_image(img_path):
    IMG_SIZE = 224
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_structured(structured_input):
    structured_input = np.array(structured_input, dtype="float32")
    expected = _expected_structured_shape()
    if structured_input.ndim == 1:
        if structured_input.size == int(np.prod(expected)):
            structured_input = structured_input.reshape(expected)
        else:
            raise ValueError(
                f"structured_features must have shape {expected} or size {int(np.prod(expected))}"
            )
    if structured_input.ndim == 2:
        if expected and structured_input.shape != expected:
            raise ValueError(
                f"structured_features must have shape {expected}, got {structured_input.shape}"
            )
        structured_input = np.expand_dims(structured_input, axis=0)
    elif structured_input.ndim == 3:
        if expected and structured_input.shape[1:] != expected:
            raise ValueError(
                f"structured_features must have shape (batch,{expected}), got {structured_input.shape}"
            )
    else:
        raise ValueError(
            f"structured_features must be 2D or 3D, got shape {structured_input.shape}"
        )
    return structured_input


def _predict(image_input, structured_input):
    _ensure_model_loaded()
    if MODEL_TYPE == "keras":
        return model.predict([image_input, structured_input])
    img = tf.convert_to_tensor(image_input, dtype=tf.float32)
    struct = tf.convert_to_tensor(structured_input, dtype=tf.float32)
    args, kwargs = _infer.structured_input_signature
    if kwargs:
        mapped = {}
        for name, spec in kwargs.items():
            shape = spec.shape.as_list() if spec.shape is not None else None
            if shape and len(shape) == 4 and shape[1:] == [224, 224, 3]:
                mapped[name] = img
            else:
                mapped[name] = struct
        outputs = _infer(**mapped)
    else:
        try:
            outputs = _infer(img, struct)
        except Exception:
            outputs = _infer([img, struct])
    if isinstance(outputs, dict):
        outputs = list(outputs.values())[0]
    return outputs.numpy()


def predict_system(image_path, structured_features):
    _ensure_model_loaded()

    image_input = preprocess_image(image_path)
    structured_input = preprocess_structured(structured_features)

    prediction = _predict(image_input, structured_input)

    predicted_class = np.argmax(prediction)
    predicted_label = image_classes[predicted_class]
    predicted_confidence = float(prediction[0][predicted_class])
    decision = build_decision_summary(predicted_label, predicted_confidence)

    return {
        "Predicted Label": predicted_label,
        "Prediction Confidence": predicted_confidence,
        "Predicted State": decision["predicted_state"],
        "Congestion Level": decision["congestion_level"],
        "Emergency Detected": decision["emergency_detected"],
        "Recommended Action": decision["recommended_action"],
        "Priority": decision["priority"],
        "Reason": decision["reason"],
    }


if __name__ == "__main__":
    data_root = Path("processed_images") / "test"
    candidates = list(data_root.rglob("*.jpg")) + list(data_root.rglob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No test images found under {data_root}")
    image_path = str(candidates[0])
    structured_features = np.random.rand(*_expected_structured_shape())

    result = predict_system(image_path, structured_features)

    for key, value in result.items():
        print(f"{key}: {value}")
