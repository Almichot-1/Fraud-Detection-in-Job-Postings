import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from src.training import load_artifacts
from src.inference import predict_single

# Initialize Flask application and enable CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)

# Global variables to cache the model pipeline and threshold
MODEL_PIPELINE = None
DECISION_THRESHOLD = 0.5

def get_model():
    """Loads and caches the trained pipeline and threshold in memory."""
    global MODEL_PIPELINE, DECISION_THRESHOLD
    if MODEL_PIPELINE is None:
        logging.info("Loading trained ML pipeline and threshold...")
        try:
            MODEL_PIPELINE, DECISION_THRESHOLD = load_artifacts()
            logging.info(f"Model loaded successfully with calibrated threshold: {DECISION_THRESHOLD:.4f}")
        except Exception as e:
            logging.error(f"Failed to load ML artifacts: {e}")
            raise e
    return MODEL_PIPELINE, DECISION_THRESHOLD

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts job posting details in JSON, performs feature engineering,
    runs the ML inference pipeline, and returns the risk level and signals.
    """
    try:
        pipeline, threshold = get_model()
    except Exception:
        return jsonify({
            "status": "error",
            "message": "Model artifacts not available on the server. Please run the training pipeline first."
        }), 500

    data = request.get_json()
    if not data:
        return jsonify({
            "status": "error",
            "message": "Missing JSON request body."
        }), 400

    # Extract original EMSCAD columns from input, falling back to empty/default values if missing
    input_dict = {
        'title': str(data.get('title', '')),
        'company_profile': str(data.get('company_profile', '')),
        'description': str(data.get('description', '')),
        'requirements': str(data.get('requirements', '')),
        'benefits': str(data.get('benefits', '')),
        'employment_type': str(data.get('employment_type', 'unspecified')),
        'required_experience': str(data.get('required_experience', 'unspecified')),
        'required_education': str(data.get('required_education', 'unspecified')),
        'industry': str(data.get('industry', 'unspecified')),
        'function': str(data.get('function', 'unspecified')),
        'salary_range': np.nan,  # Checked dynamically for salary_missing feature
        'telecommuting': int(data.get('telecommuting', 0)),
        'has_company_logo': int(data.get('has_company_logo', 0)),
        'has_questions': int(data.get('has_questions', 0))
    }

    try:
        # Run the standard prediction and signal extraction pipeline
        result = predict_single(input_dict, pipeline, threshold)
        
        return jsonify({
            "status": "success",
            "label": result['label'],
            "probability": result['probability'],
            "signals": result['signals']
        })
    except Exception as e:
        logging.error(f"Error during single prediction workflow: {e}")
        return jsonify({
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }), 500

@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check endpoint."""
    try:
        get_model()
        return jsonify({"status": "healthy", "model_loaded": True})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

if __name__ == "__main__":
    logging.info("Starting Flask REST API backend server...")
    app.run(debug=True, host='0.0.0.0', port=5000)