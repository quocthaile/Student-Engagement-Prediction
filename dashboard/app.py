import os
import json
import pandas as pd
import joblib
import numpy as np
from flask import Flask, render_template, jsonify, request
from pathlib import Path
from trainer import train_model

app = Flask(__name__)

# --- Configuration Paths ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_BUNDLE_FILE = MODEL_DIR / "deployment_bundle.pkl"
METADATA_FILE = MODEL_DIR / "metadata.json"
SAMPLES_FILE = MODEL_DIR / "sample_predictions.json"
BENCHMARK_FILE = MODEL_DIR / "benchmarks.json"

# Global Cache
cache = {
    "metadata": None,
    "all_models": None,
    "test_samples": None,
    "last_updated": 0
}

def get_cached_data():
    """Load data from disk only if cache is empty."""
    if cache["metadata"] is None or cache["test_samples"] is None:
        print("📂 Cache empty. Loading data from disk...")
        
        # Trigger training if model bundle doesn't exist
        if not MODEL_BUNDLE_FILE.exists():
            print("⚠️ Model bundle not found. Starting training...")
            train_model()

        # Load Metadata
        if METADATA_FILE.exists():
            try:
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    cache["metadata"] = json.load(f)
            except Exception as e:
                print(f"❌ Error loading metadata: {e}")
        
        # Load Sample Predictions (Entire set for server-side pagination)
        if SAMPLES_FILE.exists():
            try:
                with open(SAMPLES_FILE, 'r', encoding='utf-8') as f:
                    cache["test_samples"] = json.load(f)
            except Exception as e:
                print(f"❌ Error loading samples: {e}")
                
        # Load Benchmarks
        if BENCHMARK_FILE.exists():
            try:
                with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
                    cache["all_models"] = list(json.load(f).values())
            except Exception as e:
                print(f"❌ Error loading benchmarks: {e}")
                
    return cache

def clear_cache():
    global cache
    cache = {"metadata": None, "all_models": None, "test_samples": None, "last_updated": 0}

def initialize():
    global model_bundle
    if MODEL_BUNDLE_FILE.exists():
        try:
            model_bundle = joblib.load(MODEL_BUNDLE_FILE)
            print("✅ Loaded model bundle.")
        except Exception as e:
            print(f"❌ Error loading model bundle: {e}")
    get_cached_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    data = get_cached_data()
    # Trả về metadata và benchmarks, KHÔNG trả về test_samples (vì quá lớn)
    return jsonify({
        "metadata": data["metadata"],
        "all_models": data["all_models"]
    })

@app.route('/api/test-data')
def get_test_data():
    data = get_cached_data()
    samples = data["test_samples"] or []
    
    # Pagination parameters
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 10))
    search = request.args.get('search', '').lower()
    
    # Filter
    filtered = samples
    if search:
        filtered = [s for s in samples if search in str(s.get('row_id', '')).lower() or search in str(s.get('school_name', '')).lower()]
    
    # Paginate
    start = (page - 1) * page_size
    end = start + page_size
    paginated = filtered[start:end]
    
    return jsonify({
        "total": len(filtered),
        "page": page,
        "page_size": page_size,
        "data": paginated
    })

@app.route('/api/train', methods=['POST'])
def run_train():
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'svc')
        
        # Run training
        success = train_model(model_type)
        
        if success:
            # Reload global data and clear cache
            global model_bundle
            model_bundle = joblib.load(MODEL_BUNDLE_FILE)
            clear_cache()
            get_cached_data()
            return jsonify({"status": "success", "message": f"Mô hình {model_type} đã được huấn luyện thành công!"})
        else:
            return jsonify({"status": "error", "message": "Huấn luyện thất bại."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    global model_bundle
    if model_bundle is None:
        if MODEL_BUNDLE_FILE.exists():
            model_bundle = joblib.load(MODEL_BUNDLE_FILE)
        else:
            return jsonify({"error": "Model not found"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
            
        # Prepare features in correct order
        features = []
        for col in model_bundle['feature_columns']:
            val = data.get(col, 0)
            
            # Special handling for school name mapping
            if col == 'school_encoded' and 'school_name' in data:
                school_name = str(data['school_name']).strip()
                encoder = model_bundle.get('school_encoder')
                if encoder:
                    try:
                        val = encoder.transform([[school_name]])[0][0]
                    except:
                        val = -1
            
            # Map values correctly based on expected types
            if col in ['gender_encoded', 'num_courses', 'attempts_4w', 'rep_counts']:
                val = int(val)
            else:
                val = float(val)
                
            features.append(val)
        
        X = np.array(features).reshape(1, -1)
        X_imputed = model_bundle['imputer'].transform(X)
        X_scaled = model_bundle['scaler'].transform(X_imputed)
        
        pred_idx = model_bundle['model'].predict(X_scaled)[0]
        prediction = model_bundle['target_labels'][pred_idx]
        
        return jsonify({
            "status": "success", 
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    initialize()
    print("🚀 Dashboard is running at http://localhost:8080")
    app.run(debug=True, port=8080)
