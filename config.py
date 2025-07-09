"""Vehicleverse Configuration - Memory Optimized for Deployment"""
import os
from pathlib import Path

# Project Structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "vehicle" / "images"
MODEL_DIR = PROJECT_ROOT / "model"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
for directory in [MODEL_DIR, STATIC_DIR, TEMPLATES_DIR, LOGS_DIR, RESULTS_DIR]:
    directory.mkdir(exist_ok=True)

def get_actual_vehicle_folders():
    """Get actual vehicle folders from the data directory"""
    if not DATA_DIR.exists():
        return VEHICLE_CLASSES

    actual_folders = []
    for item in DATA_DIR.iterdir():
        if item.is_dir() and any(item.name.lower().startswith(v.split()[0].lower()) for v in VEHICLE_CLASSES):
            actual_folders.append(item.name)

    return actual_folders if actual_folders else VEHICLE_CLASSES

# Vehicle Classification Categories
VEHICLE_CLASSES = ['bicycle', 'car', 'motor bike', 'truck']
ACTUAL_VEHICLE_CLASSES = get_actual_vehicle_folders()

# Vehicle Size Categories
SIZE_CLASSES = ['compact', 'mid-size', 'large', 'extra-large']

# Enhanced Size Descriptions
SIZE_DESCRIPTIONS = {
    'compact': 'Small and efficient - Perfect for city driving and parking',
    'mid-size': 'Balanced size - Good combination of space and efficiency',
    'large': 'Spacious and powerful - Ideal for families and long trips',
    'extra-large': 'Maximum capacity - Built for heavy-duty work and transport'
}

# Model Configuration
MODEL_CONFIG = {
    'architecture': 'resnet18',
    'pretrained': True,
    'num_classes': len(ACTUAL_VEHICLE_CLASSES),
    'num_sizes': len(SIZE_CLASSES),
    'hidden_size': 256,  # Reduced from 512
    'dropout_rate': 0.5,
    'feature_extract': False
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 8,  # Reduced from 16
    'num_epochs': 25,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'train_split': 0.8,
    'val_split': 0.15,
    'test_split': 0.05,
    'early_stopping_patience': 7,
    'lr_scheduler_patience': 3,
    'lr_scheduler_factor': 0.5,
    'min_lr': 1e-6
}

# Image Processing Configuration
IMAGE_CONFIG = {
    'input_size': (224, 224),
    'resize_size': (256, 256),
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'max_file_size': 10 * 1024 * 1024,  # Reduced to 10MB
    'allowed_extensions': {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'},
    'quality_threshold': 32
}

# Flask Configuration
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.environ.get('PORT', 5000)),
    'debug': False,
    'threaded': True,
    'max_content_length': IMAGE_CONFIG['max_file_size'],
    'upload_folder': STATIC_DIR / 'uploads',
    'secret_key': os.environ.get('SECRET_KEY', 'vehicleverse-secret-key-2024')
}

# Create upload folder
FLASK_CONFIG['upload_folder'].mkdir(exist_ok=True)

# Model Paths
MODEL_PATH = MODEL_DIR / "vehicleverse_model.pth"
BEST_MODEL_PATH = MODEL_DIR / "best_vehicleverse_model.pth"
TRAINING_HISTORY_PATH = MODEL_DIR / "training_history.json"
MODEL_METRICS_PATH = MODEL_DIR / "model_metrics.json"

# Google Drive Model Configuration
GOOGLE_DRIVE_MODEL_ID = "1Bt6x2zuuli5TZ0EC67HmweyrBclESRD9"
GOOGLE_DRIVE_LINK = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_MODEL_ID}"

# Memory optimization flag
IS_PRODUCTION = os.environ.get('RENDER', False) or os.environ.get('RAILWAY_ENVIRONMENT', False)

def download_model_if_needed():
    """Download model from Google Drive if not present - Memory optimized"""
    if not BEST_MODEL_PATH.exists() and not IS_PRODUCTION:
        try:
            import gdown
            print("📦 Model not found. Downloading from Google Drive...")
            gdown.download(GOOGLE_DRIVE_LINK, output=str(BEST_MODEL_PATH), quiet=False, fuzzy=True)
            print("✅ Model downloaded successfully.")
            return True
        except Exception as e:
            print(f"❌ Error downloading model: {e}")
            return False
    elif IS_PRODUCTION:
        print("🚀 Production mode: Skipping model download to save memory")
        return False
    else:
        print("✅ Model already exists.")
        return True

print("✅ Vehicleverse Configuration Loaded Successfully!")
print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"🚗 Vehicle Classes: {len(ACTUAL_VEHICLE_CLASSES)} types")
print(f"📏 Size Classes: {len(SIZE_CLASSES)} categories")
print(f"💾 Production Mode: {IS_PRODUCTION}")