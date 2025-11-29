from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import json
import os

app = Flask(__name__, static_folder=os.path.dirname(__file__))
CORS(app)
with open('data.json', 'r') as f:
    predictions_db = [json.loads(line) for line in f]

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'home.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.dirname(__file__), filename)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects: multipart/form-data with 'image' file
    Returns: JSON with { "label": "...", "confidence": 0.xx }
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and validate image
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        
        # TODO: Replace with your actual model prediction
        # predictions = model.predict(img_array)
        # For now, return a random prediction from data.json
        import random
        prediction = random.choice(predictions_db)
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
