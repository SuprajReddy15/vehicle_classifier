"""Vehicleverse Model Architecture - Simplified Professional Classification"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime
from config import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionModule(nn.Module):
    """Attention mechanism for feature enhancement"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super(AttentionModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(batch_size, channels, 1, 1)
        return x * y

class VehicleClassifier(nn.Module):
    """Professional Vehicle Classifier - Type + Size"""
    def __init__(self,
                 architecture: str = 'resnet18',
                 pretrained: bool = True,
                 num_classes: int = 4,
                 num_sizes: int = 4,
                 hidden_size: int = 512,
                 dropout_rate: float = 0.5,
                 use_attention: bool = True):
        super(VehicleClassifier, self).__init__()

        self.architecture = architecture
        self.num_classes = num_classes
        self.num_sizes = num_sizes
        self.hidden_size = hidden_size
        self.use_attention = use_attention

        # Load backbone model
        self.backbone = self._create_backbone(architecture, pretrained)
        self.feature_dim = self._get_feature_dim()

        # Attention module
        if use_attention:
            self.attention = AttentionModule(self.feature_dim)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2)
        )

        # Task-specific classifiers
        self.vehicle_classifier = self._create_classifier(hidden_size, num_classes)
        self.size_classifier = self._create_classifier(hidden_size, num_sizes)

        # Initialize weights
        self._initialize_weights()

    def _create_backbone(self, architecture: str, pretrained: bool):
        """Create backbone network"""
        if architecture == 'resnet18':
            if pretrained:
                model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet18(weights=None)
        elif architecture == 'resnet50':
            if pretrained:
                model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        return nn.Sequential(*list(model.children())[:-2])

    def _get_feature_dim(self):
        """Get feature dimension from backbone"""
        if self.architecture == 'resnet18':
            return 512
        elif self.architecture == 'resnet50':
            return 2048
        else:
            raise ValueError(f"Unknown feature dimension for {self.architecture}")

    def _create_classifier(self, input_size: int, num_classes: int):
        """Create task-specific classifier"""
        return nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.BatchNorm1d(input_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(input_size // 2, input_size // 4),
            nn.BatchNorm1d(input_size // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(input_size // 4, num_classes)
        )

    def _initialize_weights(self):
        """Initialize weights for new layers"""
        for module in [self.feature_extractor, self.vehicle_classifier, self.size_classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)

        if self.use_attention:
            features = self.attention(features)

        features = self.global_pool(features)
        features = features.view(features.size(0), -1)
        shared_features = self.feature_extractor(features)

        vehicle_output = self.vehicle_classifier(shared_features)
        size_output = self.size_classifier(shared_features)

        return {
            'vehicle': vehicle_output,
            'size': size_output,
            'features': shared_features
        }

class ModelFactory:
    """Factory class for creating and managing Vehicleverse models"""

    @staticmethod
    def create_model(config: Optional[Dict] = None) -> VehicleClassifier:
        """Create a new Vehicleverse model"""
        if config is None:
            config = MODEL_CONFIG.copy()

        # Update num_classes based on actual vehicle classes found
        config['num_classes'] = len(VEHICLE_CLASSES)

        model = VehicleClassifier(
            architecture=config['architecture'],
            pretrained=config['pretrained'],
            num_classes=config['num_classes'],
            num_sizes=config['num_sizes'],
            hidden_size=config['hidden_size'],
            dropout_rate=config['dropout_rate'],
            use_attention=True
        )
        return model

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
            'model_config': MODEL_CONFIG,
            'training_config': TRAINING_CONFIG,
            'vehicle_classes': VEHICLE_CLASSES,
            'size_classes': SIZE_CLASSES,
            'size_descriptions': SIZE_DESCRIPTIONS,
            'architecture': model.architecture,
            'feature_dim': model.feature_dim,
            'hidden_size': model.hidden_size,
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
            model_config = checkpoint.get('model_config', MODEL_CONFIG)
            model = ModelFactory.create_model(model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()

            logger.info(f"✅ Model loaded from: {path}")
            return model, checkpoint
        except Exception as e:
            logger.error(f"❌ Failed to load model from {path}: {e}")
            raise

    @staticmethod
    def get_model_info(model_path):
        """Get model information"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            return {
                'epoch': checkpoint.get('epoch', 'Unknown'),
                'val_accuracy': checkpoint.get('val_accuracy', 'Unknown'),
                'architecture': checkpoint.get('architecture', 'Unknown'),
                'vehicle_classes': checkpoint.get('vehicle_classes', []),
                'size_classes': checkpoint.get('size_classes', [])
            }
        except Exception as e:
            return {'error': str(e)}