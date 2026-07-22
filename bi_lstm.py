import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

print("Loading models...")

cnn_path = Path("cnn_traffic_model.keras")
bilstm_path = Path("bilstm_traffic_model.keras")
if not cnn_path.exists():
    raise FileNotFoundError(f"Missing model: {cnn_path}")
if not bilstm_path.exists():
    raise FileNotFoundError(f"Missing model: {bilstm_path}")


def _export_model(model, export_dir, exported_keras_path=None):
    try:
        if hasattr(model, "export"):
            model.export(str(export_dir))
        else:
            tf.saved_model.save(model, str(export_dir))
        print(f"Exported model to: {export_dir}", flush=True)
    except Exception as exc:
        print(f"Export failed for {export_dir}: {exc}", flush=True)
    if exported_keras_path is not None:
        try:
            model.save(str(exported_keras_path))
            print(f"Exported Keras model to: {exported_keras_path}", flush=True)
        except Exception as exc:
            print(f"Keras export failed for {exported_keras_path}: {exc}", flush=True)


def _load_model_with_export(path):
    try:
        return load_model(str(path), safe_mode=False)
    except Exception as exc:
        print(f"Load failed for {path}: {exc}", flush=True)
        model = load_model(str(path), safe_mode=False, compile=False)
        export_dir = path.with_name(f"{path.stem}_savedmodel")
        export_keras = path.with_name(f"{path.stem}_exported.keras")
        _export_model(model, export_dir, exported_keras_path=export_keras)
        return model


cnn_model = _load_model_with_export(cnn_path)
bilstm_model = _load_model_with_export(bilstm_path)

image_classes_path = Path("image_classes.npy")
if not image_classes_path.exists():
    raise FileNotFoundError(f"Missing classes file: {image_classes_path}")
image_classes = np.load(str(image_classes_path), allow_pickle=True)


def _load_structured_classes():
    classes_path = Path("structured_classes.npy")
    if classes_path.exists():
        return np.load(str(classes_path), allow_pickle=True)
    y_labels_path = Path("y_labels.npy")
    if y_labels_path.exists():
        classes = np.unique(np.load(str(y_labels_path), allow_pickle=True))
        try:
            np.save(str(classes_path), classes)
        except Exception:
            pass
        return classes
    return np.array(["Normal", "Emergency"], dtype=object)


structured_classes = _load_structured_classes()

print("Models loaded successfully.\n")


# ==========================================================
# 2) IMAGE PREPROCESSING
# ==========================================================


def preprocess_image(image_path):
    IMG_SIZE = 224

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found or invalid path")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)

    return img


# ==========================================================
# 3) STRUCTURED DATA PREPROCESSING
# ==========================================================


def _expected_structured_shape():
    shape = bilstm_model.input_shape
    if isinstance(shape, (list, tuple)) and shape and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    if shape is None or len(shape) < 3:
        return (5, 10)
    return tuple(shape[1:])


def _shape_matches(actual, expected):
    if expected is None:
        return True
    if len(actual) != len(expected):
        return False
    for a, e in zip(actual, expected):
        if e is not None and a != e:
            return False
    return True


def preprocess_structured(structured_input):
    structured_input = np.array(structured_input, dtype="float32")
    expected = _expected_structured_shape()
    if structured_input.ndim == 1:
        if None in expected:
            raise ValueError(
                f"structured_features must be 2D or 3D, got shape {structured_input.shape}"
            )
        if structured_input.size != int(np.prod(expected)):
            raise ValueError(
                f"structured_features must have size {int(np.prod(expected))}, got {structured_input.size}"
            )
        structured_input = structured_input.reshape(expected)
        structured_input = np.expand_dims(structured_input, axis=0)
    elif structured_input.ndim == 2:
        if _shape_matches(structured_input.shape, expected):
            structured_input = np.expand_dims(structured_input, axis=0)
        elif _shape_matches(structured_input.shape, expected[::-1]):
            structured_input = np.expand_dims(structured_input.T, axis=0)
        else:
            raise ValueError(
                f"structured_features must have shape {expected}, got {structured_input.shape}"
            )
    elif structured_input.ndim == 3:
        if _shape_matches(structured_input.shape[1:], expected):
            pass
        elif _shape_matches(structured_input.shape[1:], expected[::-1]):
            structured_input = structured_input.transpose(0, 2, 1)
        else:
            raise ValueError(
                f"structured_features must have shape (batch,{expected}), got {structured_input.shape}"
            )
    else:
        raise ValueError(
            f"structured_features must be 1D, 2D, or 3D, got shape {structured_input.shape}"
        )
    return structured_input


# ==========================================================
# 4) CNN EMERGENCY DETECTION
# ==========================================================


def _to_label(value):
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def detect_emergency(image_input):

    prediction = cnn_model.predict(image_input, verbose=0)

    predicted_class = int(np.argmax(prediction, axis=-1)[0])
    predicted_label = _to_label(image_classes[predicted_class])
    confidence = float(prediction[0][predicted_class])

    emergency_flag = 1 if "emergency" in predicted_label.lower() else 0

    return predicted_label, emergency_flag, confidence


# ==========================================================
# 5) Bi-LSTM CONGESTION PREDICTION
# ==========================================================


def predict_congestion(structured_input):

    prediction = bilstm_model.predict(structured_input, verbose=0)
    prediction = np.array(prediction)

    if prediction.ndim == 2 and prediction.shape[1] == 1:
        score = float(prediction[0][0])
        predicted_class = 1 if score >= 0.5 else 0
        if len(structured_classes) < 2:
            raise ValueError("structured_classes must have at least 2 labels for binary output.")
        predicted_label = _to_label(structured_classes[predicted_class])
        confidence = score if predicted_class == 1 else 1.0 - score
    else:
        predicted_class = int(np.argmax(prediction, axis=-1)[0])
        predicted_label = _to_label(structured_classes[predicted_class])
        confidence = float(prediction[0][predicted_class])

    return predicted_label, confidence


# ==========================================================
# 6) ONTOLOGY REASONING ENGINE (ADVANCED)
# ==========================================================


def ontology_reasoning(congestion_level, emergency_flag):

    # Rule 1
    if congestion_level == "High" and emergency_flag == 1:
        return {
            "Action": "Activate Green Wave",
            "Signal Timing": "Extended (30 sec)",
            "Priority": "Critical",
            "Explanation": "Emergency detected in high congestion zone"
        }

    # Rule 2
    elif congestion_level == "High":
        return {
            "Action": "Traffic Diversion + Signal Optimization",
            "Signal Timing": "Adaptive (25 sec)",
            "Priority": "High",
            "Explanation": "High congestion without emergency"
        }

    # Rule 3
    elif congestion_level == "Medium":
        return {
            "Action": "Adjust Signal Phase",
            "Signal Timing": "Normal (20 sec)",
            "Priority": "Moderate",
            "Explanation": "Moderate congestion detected"
        }

    # Rule 4
    else:
        return {
            "Action": "Normal Monitoring",
            "Signal Timing": "Standard (15 sec)",
            "Priority": "Low",
            "Explanation": "Traffic operating normally"
        }

def run_traffic_system(image_path, structured_features):

    # Step 1: Preprocess
    image_input = preprocess_image(image_path)
    structured_input = preprocess_structured(structured_features)

    # Step 2: CNN
    image_label, emergency_flag, image_conf = detect_emergency(image_input)

    # Step 3: Bi-LSTM
    congestion_label, congestion_conf = predict_congestion(structured_input)

    # Step 4: Ontology Reasoning
    decision = ontology_reasoning(congestion_label, emergency_flag)

    # Step 5: Final Output
    output = {
        "Emergency Detection (CNN)": image_label,
        "Emergency Confidence": round(float(image_conf), 4),
        "Congestion Prediction (Bi-LSTM)": congestion_label,
        "Congestion Confidence": round(float(congestion_conf), 4),
        "Final Decision": decision["Action"],
        "Signal Timing": decision["Signal Timing"],
        "Priority Level": decision["Priority"],
        "Explanation": decision["Explanation"]
    }

    return output


# ==========================================================
# 8) TEST RUN
# ==========================================================

if __name__ == "__main__":

    print("Running Traffic AI System...\n")

    data_root = Path("processed_images") / "test"
    candidates = list(data_root.rglob("*.jpg")) + list(data_root.rglob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No test images found under {data_root}")
    test_image = str(candidates[0])

    # Example structured input (time steps, features)
    structured_example = np.random.rand(*_expected_structured_shape())

    results = run_traffic_system(test_image, structured_example)

    print("========== FINAL SYSTEM OUTPUT ==========\n")

    for key, value in results.items():
        print(f"{key}: {value}")
