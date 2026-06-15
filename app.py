from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import sqlite3
import datetime
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hematology.db")
MODEL_PATH = os.path.join(BASE_DIR, "psmear_model.onnx")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS patients (id INTEGER PRIMARY KEY AUTOINCREMENT, hospital_no TEXT UNIQUE, age INTEGER, dob TEXT, address TEXT)')
    cursor.execute('CREATE TABLE IF NOT EXISTS diagnostics (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER, test_type TEXT, prediction TEXT, confidence TEXT, timestamp TEXT, FOREIGN KEY (patient_id) REFERENCES patients (id))')
    conn.commit()
    conn.close()

init_db()

# Load class names
try:
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
except:
    class_names = ["Leukemia", "Lymphoma", "Myeloma", "Normal"]

# Initialize ONNX Session
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    print("ONNX Model loaded successfully.")
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    ort_session = None

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((224, 224))
    img_data = np.array(img).astype('float32') / 255.0
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = (img_data - mean) / std
    img_data = np.transpose(img_data, (2, 0, 1)) # HWC to CHW
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32) # Add batch dim
    return img_data

@app.route('/predict_blood', methods=['POST'])
@app.route('/predict_tissue', methods=['POST'])
@app.route('/predict_myeloma', methods=['POST'])
def predict():
    if ort_session is None:
        return jsonify({"error": "Model not loaded"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        input_data = preprocess_image(file.read())
    except Exception as e:
        return jsonify({"error": f"Invalid or corrupted image file."}), 400
    
    # Run Inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Softmax and Filtering
    logits = ort_outs[0][0]
    
    # Determine allowed classes based on the endpoint
    try:
        leukemia_idx = class_names.index("Leukemia")
    except ValueError:
        leukemia_idx = 0
        
    try:
        lymphoma_idx = class_names.index("Lymphoma")
    except ValueError:
        lymphoma_idx = 1
        
    try:
        myeloma_idx = class_names.index("Myeloma")
    except ValueError:
        myeloma_idx = 2
        
    try:
        normal_idx = class_names.index("Normal")
    except ValueError:
        normal_idx = 3

    # Remove masking to prevent the system from forcing Myeloma into Leukemia (viewing them as one)
    allowed_indices = list(range(len(class_names)))
    
    # Mask unallowed logits with -inf
    masked_logits = np.full_like(logits, -np.inf)
    for idx in allowed_indices:
        if idx < len(logits):
            masked_logits[idx] = logits[idx]
            
    exp_logits = np.exp(masked_logits - np.max(masked_logits))
    probs = exp_logits / exp_logits.sum()
    
    idx = np.argmax(probs)
    prediction = class_names[idx]
    confidence = probs[idx] * 100
    
    return jsonify({
        "prediction": prediction,
        "confidence": f"{confidence:.2f}%",
        "classes": class_names
    })

@app.route('/save_record', methods=['POST'])
def save_record():
    data = request.json
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO patients (hospital_no, age, dob, address) VALUES (?, ?, ?, ?) ON CONFLICT(hospital_no) DO UPDATE SET age=excluded.age, dob=excluded.dob, address=excluded.address', (data['hospital_no'], data['age'], data['dob'], data['address']))
        cursor.execute('SELECT id FROM patients WHERE hospital_no = ?', (data['hospital_no'],))
        p_id = cursor.fetchone()[0]
        cursor.execute('INSERT INTO diagnostics (patient_id, test_type, prediction, confidence, timestamp) VALUES (?, ?, ?, ?, ?)', (p_id, data['test_type'], data['prediction'], data['confidence'], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_patient', methods=['POST'])
def save_patient():
    data = request.json
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO patients (hospital_no, age, dob, address) VALUES (?, ?, ?, ?) ON CONFLICT(hospital_no) DO UPDATE SET age=excluded.age, dob=excluded.dob, address=excluded.address', (data['hospital_no'], data['age'], data['dob'], data.get('address', '')))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT p.hospital_no, p.age, d.test_type, d.prediction, d.confidence, d.timestamp FROM diagnostics d JOIN patients p ON d.patient_id = p.id ORDER BY d.timestamp DESC').fetchall()
    res = [dict(r) for r in rows]; conn.close()
    return jsonify(res)

@app.route('/')
def home():
    return send_file('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
