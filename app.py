import sys
import subprocess
from pathlib import Path
from uuid import uuid4

from flask import Flask, Response, render_template, request, url_for
from asgiref.wsgi import WsgiToAsgi
from werkzeug.utils import secure_filename

from ExAi import analyze_image


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ANALYSIS_DIR = BASE_DIR / "static" / "analysis"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


def format_subprocess_output(result):
    parts = [f"Exit code: {result.returncode}"]

    if result.stdout:
        parts.append(f"STDOUT:\n{result.stdout}")

    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")

    if len(parts) == 1:
        parts.append("No output produced.")

    return "\n\n".join(parts)


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def static_url(path):
    relative_path = Path(path).resolve().relative_to((BASE_DIR / "static").resolve())
    return url_for("static", filename=str(relative_path).replace("\\", "/"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/analyze_image", methods=["POST"])
def analyze_uploaded_image():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    uploaded_file = request.files.get("image_file")
    if uploaded_file is None or uploaded_file.filename == "":
        return render_template("index.html", upload_error="Please choose an image file.")

    if not allowed_file(uploaded_file.filename):
        return render_template("index.html", upload_error="Supported formats: JPG, JPEG, PNG, BMP, WEBP.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id = uuid4().hex
    safe_name = secure_filename(uploaded_file.filename) or f"{file_id}.png"
    upload_path = UPLOAD_DIR / f"{file_id}_{safe_name}"
    uploaded_file.save(upload_path)

    try:
        analysis_output = analyze_image(upload_path, ANALYSIS_DIR / file_id)
    except Exception as exc:
        return render_template("index.html", upload_error=f"Analysis failed: {exc}")

    analysis_result = {
        "uploaded_image_url": static_url(upload_path),
        "prediction": analysis_output["prediction"],
        "prediction_confidence": analysis_output["prediction_confidence"],
        "congestion_level": analysis_output["congestion_level"],
        "emergency_detected": analysis_output["emergency_detected"],
        "recommended_action": analysis_output["recommended_action"],
        "priority": analysis_output["priority"],
        "reason": analysis_output["reason"],
        "reference_context": analysis_output["reference_context"],
        "image_feature_summary": analysis_output["image_feature_summary"],
        "shap_text_explainer": analysis_output["shap_text_explainer"],
        "text_explainer": analysis_output["text_explainer"],
        "gradcam_heatmap_url": static_url(analysis_output["files"]["gradcam_heatmap"]),
        "gradcam_overlay_url": static_url(analysis_output["files"]["gradcam_overlay"]),
    }

    return render_template("index.html", analysis_result=analysis_result)


@app.route("/run_system", methods=["POST"])
def run_system():
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "main_system.py")],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )
    output = format_subprocess_output(result)
    return render_template("index.html", system_output=output)


@app.route("/run_explain", methods=["POST"])
def run_explain():
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "ExAi.py")],
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
    )
    output = format_subprocess_output(result)
    return render_template("index.html", explain_output=output)


asgi_app = WsgiToAsgi(app)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
