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

        # Download model if needed
        download_model_if_needed()

        # Load model
        self.load_model()
        # Setup transforms
        self.setup_transforms()

    def load_model(self):
        """Load the trained Vehicleverse model"""
        try:
            # Try to load best model first, then fallback to regular model
            model_paths = [BEST_MODEL_PATH, self.model_path]

            for path in model_paths:
                if os.path.exists(path):
                    # Load with weights_only=False to handle the serialization issue
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)

                    # Create model
                    model_config = checkpoint.get('model_config', MODEL_CONFIG)
                    self.model = ModelFactory.create_model(model_config)

                    # Load state dict
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model.to(self.device)
                    self.model.eval()

                    logger.info(f"✅ Vehicleverse model loaded successfully from {path}!")
                    logger.info(f"   Device: {self.device}")
                    logger.info(f"   Model accuracy: {checkpoint.get('val_accuracy', 'Unknown')}")
                    return True

            logger.warning(f"❌ No model file found at {model_paths}")
            # Create a dummy model for deployment
            self.create_dummy_model()
            return False

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            # Create a dummy model for deployment
            self.create_dummy_model()
            return False

    def create_dummy_model(self):
        """Create a dummy model for deployment when real model is not available"""
        try:
            logger.info("🔧 Creating dummy model for deployment...")
            self.model = ModelFactory.create_model()
            self.model.to(self.device)
            self.model.eval()
            logger.info("✅ Dummy model created successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create dummy model: {e}")
            return False

    def setup_transforms(self):
        """Setup image preprocessing transforms"""
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
        """Make prediction on a single image"""
        if self.model is None:
            return None

        try:
            # Preprocess image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                return None

            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)

            # Process outputs
            predictions = {}
            confidences = {}

            for task in ['vehicle', 'size']:
                # Get probabilities
                probs = torch.softmax(outputs[task], dim=1)
                confidence, predicted = torch.max(probs, 1)

                predictions[task] = {
                    'class_idx': predicted.item(),
                    'class_name': self.classes[task][predicted.item()],
                    'confidence': confidence.item() * 100,
                    'all_probabilities': probs.cpu().numpy().tolist()[0]
                }
                confidences[task] = confidence.item() * 100

            # Add size description
            size_name = predictions['size']['class_name']
            predictions['size']['description'] = SIZE_DESCRIPTIONS.get(
                size_name, 'Standard vehicle size'
            )

            # Calculate overall confidence
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

# Initialize predictor
predictor = VehicleversePredictor()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

        # Save uploaded file
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction
        result = predictor.predict(filepath)
        if result is None:
            return jsonify({'error': 'Prediction failed. Please try again.'}), 500

        # Convert image to base64 for display
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')

        # Add image data to result
        result['image_data'] = f"data:image/jpeg;base64,{img_data}"
        result['filename'] = file.filename

        # Clean up uploaded file
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
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '2.0.0'
    })

@app.route('/model-info')
def model_info():
    """Get model information"""
    if predictor.model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        info = {
            'architecture': 'ResNet-18',
            'vehicle_classes': VEHICLE_CLASSES,
            'size_classes': SIZE_CLASSES,
            'status': 'loaded'
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': 'Could not retrieve model info'}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'bmp', 'gif'}

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Vehicleverse Web Application...")
    print(f"🌐 URL: http://localhost:{FLASK_CONFIG['port']}")
    print(f"📱 Features: Professional vehicle classification")
    print(f"🎯 Ready to analyze: Type + Size")
    print("=" * 60)

    app.run(
        host=FLASK_CONFIG['host'],
        port=FLASK_CONFIG['port'],
        debug=FLASK_CONFIG['debug']
    )