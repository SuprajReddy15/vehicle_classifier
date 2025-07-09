"""Vehicleverse Model Architecture - Ultra Lightweight"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import logging
from typing import Dict, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleClassifier(nn.Module):
    """Ultra-lightweight Vehicle Classifier"""
    def __init__(self,
                 architecture: str = 'resnet18',
                 pretrained: bool = True,
                 num_classes: int = 4,
                 num_sizes: int = 4,
                 hidden_size: int = 128,
                 dropout_rate: float = 0.5):
        super(VehicleClassifier, self).__init__()

        self.architecture = architecture
        self.num_classes = num_classes
        self.num_sizes = num_sizes
        self.hidden_size = hidden_size

        # Lightweight backbone
        if architecture == 'resnet18':
            if pretrained:
                model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.feature_dim = 512
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Ultra-lightweight feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, hidden_size),
            nn.ReLU(inplace=True)
        )

        # Minimal classifiers
        self.vehicle_classifier = nn.Linear(hidden_size, num_classes)
        self.size_classifier = nn.Linear(hidden_size, num_sizes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        shared_features = self.feature_extractor(features)

        return {
            'vehicle': self.vehicle_classifier(shared_features),
            'size': self.size_classifier(shared_features),
            'features': shared_features
        }

class ModelFactory:
    """Factory class for creating models"""

    @staticmethod
    def create_model(config: Optional[Dict] = None) -> VehicleClassifier:
        """Create a new model"""
        if config is None:
            from config import MODEL_CONFIG, VEHICLE_CLASSES
            config = MODEL_CONFIG.copy()
            config['num_classes'] = len(VEHICLE_CLASSES)

        return VehicleClassifier(
            architecture=config['architecture'],
            pretrained=config['pretrained'],
            num_classes=config['num_classes'],
            num_sizes=config['num_sizes'],
            hidden_size=config['hidden_size'],
            dropout_rate=config['dropout_rate']
        )

    @staticmethod
    def save_model(model: VehicleClassifier,
                   path,
                   epoch: int,
                   train_loss: float,
                   val_loss: float,
                   val_accuracy: float,
                   optimizer=None,
                   scheduler=None,
                   metrics=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'timestamp': datetime.now().isoformat()
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if metrics is not None:
            checkpoint['metrics'] = metrics

        torch.save(checkpoint, path)
        logger.info(f"💾 Model saved to: {path}")

    @staticmethod
    def load_model(path, device=None, map_location=None):
        """Load model from checkpoint"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if map_location is None:
            map_location = str(device)

        try:
            checkpoint = torch.load(path, map_location=map_location, weights_only=False)
            model = ModelFactory.create_model()
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            logger.info(f"✅ Model loaded from: {path}")
            return model, checkpoint
        except Exception as e:
            logger.error(f"❌ Failed to load model from {path}: {e}")
            raise
