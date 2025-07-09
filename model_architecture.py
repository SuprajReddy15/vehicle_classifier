"""Vehicleverse Model Architecture - FINAL EXACT MATCH"""
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
    """Vehicle Classifier - FINAL EXACT MATCH for your saved model"""
    def __init__(self,
                 architecture: str = 'resnet18',
                 pretrained: bool = True,
                 num_classes: int = 4,
                 num_sizes: int = 4,
                 hidden_size: int = 32,  # EXACT: Your model uses 32, not 256!
                 dropout_rate: float = 0.5):
        super(VehicleClassifier, self).__init__()

        self.architecture = architecture
        self.num_classes = num_classes
        self.num_sizes = num_sizes
        self.hidden_size = hidden_size

        # Load backbone model
        if architecture == 'resnet18':
            if pretrained:
                model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # Use only the feature extraction part
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        self.feature_dim = 512

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Attention mechanism - EXACT match (512→32→512)
        self.attention = nn.Sequential()
        self.attention.add_module('fc1', nn.Linear(self.feature_dim, hidden_size))  # 512→32
        self.attention.add_module('relu1', nn.ReLU(inplace=True))
        self.attention.add_module('fc2', nn.Linear(hidden_size, self.feature_dim))  # 32→512
        self.attention.add_module('sigmoid', nn.Sigmoid())

        # Feature extractor - EXACT match (512→256 with BatchNorm)
        self.feature_extractor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 256),  # 512→256 (EXACT from logs)
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, 256),  # 256→256
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        # Vehicle classifier - EXACT match (256→128→4)
        self.vehicle_classifier = nn.Sequential(
            nn.Linear(256, 256),              # 0: 256→256
            nn.BatchNorm1d(256),              # 1: BatchNorm(256)
            nn.ReLU(inplace=True),            # 2: ReLU
            nn.Dropout(0.3),                 # 3: Dropout
            nn.Linear(256, 128),             # 4: 256→128
            nn.BatchNorm1d(128),             # 5: BatchNorm(128)
            nn.ReLU(inplace=True),           # 6: ReLU
            nn.Dropout(0.2),                 # 7: Dropout
            nn.Linear(128, num_classes)      # 8: 128→4
        )

        # Size classifier - EXACT match (256→128→4)
        self.size_classifier = nn.Sequential(
            nn.Linear(256, 256),              # 0: 256→256
            nn.BatchNorm1d(256),              # 1: BatchNorm(256)
            nn.ReLU(inplace=True),            # 2: ReLU
            nn.Dropout(0.3),                 # 3: Dropout
            nn.Linear(256, 128),             # 4: 256→128
            nn.BatchNorm1d(128),             # 5: BatchNorm(128)
            nn.ReLU(inplace=True),           # 6: ReLU
            nn.Dropout(0.2),                 # 7: Dropout
            nn.Linear(128, num_sizes)        # 8: 128→4
        )

    def forward(self, x):
        """Forward pass with attention mechanism"""
        # Extract features
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(features.size(0), -1)

        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights

        # Extract shared features
        shared_features = self.feature_extractor(attended_features)

        # Get predictions
        vehicle_output = self.vehicle_classifier(shared_features)
        size_output = self.size_classifier(shared_features)

        return {
            'vehicle': vehicle_output,
            'size': size_output,
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
            hidden_size=config.get('hidden_size', 32),  # EXACT: 32 not 256!
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
            'model_config': {
                'architecture': model.architecture,
                'num_classes': model.num_classes,
                'num_sizes': model.num_sizes,
                'hidden_size': model.hidden_size
            },
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

            # Get model config from checkpoint or use default
            model_config = checkpoint.get('model_config', {
                'architecture': 'resnet18',
                'num_classes': 4,
                'num_sizes': 4,
                'hidden_size': 32  # EXACT match
            })

            model = VehicleClassifier(
                architecture=model_config.get('architecture', 'resnet18'),
                num_classes=model_config.get('num_classes', 4),
                num_sizes=model_config.get('num_sizes', 4),
                hidden_size=model_config.get('hidden_size', 32)  # EXACT: 32
            )

            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            logger.info(f"✅ Model loaded from: {path}")
            return model, checkpoint
        except Exception as e:
            logger.error(f"❌ Failed to load model from {path}: {e}")
            raise
