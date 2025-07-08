"""Vehicleverse Flask Application - Professional Vehicle Classification Web Interface"""
from flask import Flask, request, jsonify, render_template
import torch
import torch.serialization
from PIL import Image
import os
import logging
from datetime import datetime
import numpy as np
from werkzeug.utils import secure_filename
import base64
import uuid
from pathlib import Path
import gdown
from config import *
from model_architecture import ModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fix PyTorch serialization issue
torch.serialization.add_safe_globals([
    'numpy.core.multiarray.scalar',
    'numpy._core.multiarray.scalar'
])

# Initialize Flask app
app = Flask(__name__)
app.config.update(FLASK_CONFIG)
app.config['MAX_CONTENT_LENGTH'] = FLASK_CONFIG['max_content_length']
app.config['UPLOAD_FOLDER'] = FLASK_CONFIG['upload_folder']

# Download model from Google Drive if not exists
if not os.path.exists(BEST_MODEL_PATH):
    print("\U0001F4E6 Model not found. Downloading from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=1Bt6x2zuuli5TZ0EC67HmweyrBclESRD9",
        output=str(BEST_MODEL_PATH),
        quiet=False,
        fuzzy=True
    )
    print("\u2705 Model downloaded successfully.")

class VehicleversePredictor:
    """Vehicleverse prediction engine"""

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = None
        self.classes = {
            'vehicle': VEHICLE_CLASSES,
            'size': SIZE_CLASSES
        }

        self.load_model()
        self.setup_transforms()

    def load_model(self):
        try:
            model_paths = [BEST_MODEL_PATH, self.model_path]

            for path in model_paths:
                if os.path.exists(path):
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    model_config = checkpoint.get('model_config', MODEL_CONFIG)
                    self.model = ModelFactory.create_model(model_config)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"✅ Vehicleverse model loaded successfully from {path}!")
                    logger.info(f"   Device: {self.device}")
                    logger.info(f"   Model accuracy: {checkpoint.get('val_accuracy', 'Unknown')}")
                    return True

            logger.warning(f"❌ No model file found at {model_paths}")
            return False

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False

    def setup_transforms(self):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(IMAGE_CONFIG['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGE_CONFIG['mean'],
                std=IMAGE_CONFIG['std']
            )
        ])

    def predict(self, image):
        if self.model is None:
            return None

        try:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                return None

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)

            predictions = {}
            confidences = {}

            for task in ['vehicle', 'size']:
                probs = torch.softmax(outputs[task], dim=1)
                confidence, predicted = torch.max(probs, 1)

                predictions[task] = {
                    'class_idx': predicted.item(),
                    'class_name': self.classes[task][predicted.item()],
                    'confidence': confidence.item() * 100,
                    'all_probabilities': probs.cpu().numpy().tolist()[0]
                }
                confidences[task] = confidence.item() * 100

            size_name = predictions['size']['class_name']
            predictions['size']['description'] = SIZE_DESCRIPTIONS.get(
                size_name, 'Standard vehicle size'
            )

            overall_confidence = np.mean(list(confidences.values()))

            result = {
                'predictions': predictions,
                'overall_confidence': overall_confidence,
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'architecture': 'ResNet-18',
                    'features': ['Vehicle Type', 'Size Category'],
                    'classes': {
                        'vehicle': len(self.classes['vehicle']),
                        'size': len(self.classes['size'])
                    }
                }
            }

            return result

        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return None

predictor = VehicleversePredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = predictor.predict(filepath)
        if result is None:
            return jsonify({'error': 'Prediction failed. Please try again.'}), 500

        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')

        result['image_data'] = f"data:image/jpeg;base64,{img_data}"
        result['filename'] = file.filename

        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Error in predict route: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/model-info')
def model_info():
    if predictor.model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        info = ModelFactory.get_model_info(predictor.model_path)
        return jsonify(info)
    except:
        return jsonify({'error': 'Could not retrieve model info'}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'bmp', 'gif'}

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Vehicleverse Web Application...")
    print(f"🌐 URL: http://localhost:{FLASK_CONFIG['port']}")
    print(f"📱 Features: Professional vehicle classification")
    print(f"🎯 Ready to analyze: Type + Size")
    print("=" * 60)

    model_exists = os.path.exists(BEST_MODEL_PATH) or os.path.exists(MODEL_PATH)
    if not model_exists:
        print("⚠️ Warning: No trained model found!")
        print("   Please run training first: python run_pipeline.py")
        print("   The web app will still start but predictions will fail.")

    app.run(
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug']
    )
