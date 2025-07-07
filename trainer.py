"""Vehicleverse Trainer - Simplified for Type + Size Classification"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from config import *
from model_architecture import ModelFactory, VehicleClassifier
from dataset_manager import VehicleverseDatasetManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""
    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False

    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint"""
        self.best_weights = {k: v.clone().detach() for k, v in model.state_dict().items()}

class MetricsCalculator:
    """Metrics calculation for multi-task learning"""
    def __init__(self, num_classes: Dict[str, int]):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.predictions = {task: [] for task in self.num_classes.keys()}
        self.targets = {task: [] for task in self.num_classes.keys()}
        self.losses = {task: [] for task in self.num_classes.keys()}
        self.total_loss = []

    def update(self, outputs: Dict[str, torch.Tensor],
               targets: Dict[str, torch.Tensor],
               losses: Dict[str, torch.Tensor]):
        """Update metrics with batch results"""
        for task in self.num_classes.keys():
            # Properly detach and convert to numpy
            preds = torch.argmax(outputs[task], dim=1).detach().cpu().numpy()
            targs = targets[task].detach().cpu().numpy()

            self.predictions[task].extend(preds.tolist())
            self.targets[task].extend(targs.tolist())

            # Convert tensor loss to float
            loss_value = float(losses[task].detach().cpu().item())
            self.losses[task].append(loss_value)

        # Calculate total loss as sum of individual losses
        total_loss_value = sum(float(losses[task].detach().cpu().item()) for task in self.num_classes.keys())
        self.total_loss.append(total_loss_value)

    def compute_metrics(self) -> Dict:
        """Compute comprehensive metrics"""
        metrics = {}
        for task in self.num_classes.keys():
            if len(self.predictions[task]) == 0:
                continue

            try:
                accuracy = accuracy_score(self.targets[task], self.predictions[task])
                metrics[task] = {
                    'accuracy': float(accuracy * 100),
                    'loss': float(np.mean(self.losses[task])) if self.losses[task] else 0.0
                }
            except Exception as e:
                logger.warning(f"Error computing metrics for {task}: {e}")
                metrics[task] = {'accuracy': 0.0, 'loss': 0.0}

        # Calculate overall metrics
        if metrics:
            combined_accuracy = np.mean([metrics[task]['accuracy'] for task in metrics.keys()])
            total_loss = float(np.mean(self.total_loss)) if self.total_loss else 0.0
        else:
            combined_accuracy = 0.0
            total_loss = 0.0

        metrics['overall'] = {
            'total_loss': total_loss,
            'combined_accuracy': float(combined_accuracy)
        }
        return metrics

class VehicleverseTrainer:
    """Simplified trainer for vehicle classification"""

    def __init__(self, model=None, device=None, config=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or TRAINING_CONFIG

        if model is None:
            self.model = ModelFactory.create_model()
        else:
            self.model = model
        self.model.to(self.device)

        # Loss functions
        self.criterion = {
            'vehicle': nn.CrossEntropyLoss(),
            'size': nn.CrossEntropyLoss()
        }

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=self.config['lr_scheduler_factor'],
            patience=self.config['lr_scheduler_patience'], min_lr=self.config['min_lr'],
            verbose=True
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience']
        )

        # Metrics calculator
        self.metrics_calc = MetricsCalculator({
            'vehicle': len(VEHICLE_CLASSES),
            'size': MODEL_CONFIG['num_sizes']
        })

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
            'epoch_times': []
        }
        self.best_val_score = 0.0

        logger.info(f"🚀 VehicleverseTrainer initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train_epoch(self, train_loader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        self.metrics_calc.reset()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)

        for batch_idx, (images, labels) in enumerate(pbar):
            try:
                # Move data to device
                images = images.to(self.device, non_blocking=True)
                targets = {
                    'vehicle': labels['vehicle'].to(self.device, non_blocking=True),
                    'size': labels['size'].to(self.device, non_blocking=True)
                }

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)

                # Calculate losses
                losses = {}
                for task in ['vehicle', 'size']:
                    losses[task] = self.criterion[task](outputs[task], targets[task])

                # Backward pass
                total_loss = sum(losses.values())
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Update metrics
                self.metrics_calc.update(outputs, targets, losses)

                # Update progress bar
                if batch_idx % 10 == 0:
                    try:
                        current_metrics = self.metrics_calc.compute_metrics()
                        pbar.set_postfix({
                            'Loss': f"{current_metrics['overall']['total_loss']:.4f}",
                            'Acc': f"{current_metrics['overall']['combined_accuracy']:.1f}%"
                        })
                    except:
                        pass

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue

        return self.metrics_calc.compute_metrics()

    def validate_epoch(self, val_loader, epoch: int) -> Dict:
        """Validate for one epoch"""
        self.model.eval()
        self.metrics_calc.reset()

        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]', leave=False)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(pbar):
                try:
                    # Move data to device
                    images = images.to(self.device, non_blocking=True)
                    targets = {
                        'vehicle': labels['vehicle'].to(self.device, non_blocking=True),
                        'size': labels['size'].to(self.device, non_blocking=True)
                    }

                    # Forward pass
                    outputs = self.model(images)

                    # Calculate losses
                    losses = {}
                    for task in ['vehicle', 'size']:
                        losses[task] = self.criterion[task](outputs[task], targets[task])

                    # Update metrics
                    self.metrics_calc.update(outputs, targets, losses)

                    # Update progress bar
                    if batch_idx % 5 == 0:
                        try:
                            current_metrics = self.metrics_calc.compute_metrics()
                            pbar.set_postfix({
                                'Loss': f"{current_metrics['overall']['total_loss']:.4f}",
                                'Acc': f"{current_metrics['overall']['combined_accuracy']:.1f}%"
                            })
                        except:
                            pass

                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue

        return self.metrics_calc.compute_metrics()

    def train(self, train_loader, val_loader, num_epochs=None) -> Dict:
        """Complete training loop"""
        num_epochs = num_epochs or self.config['num_epochs']

        logger.info(f"🎯 Starting Vehicleverse Training...")
        logger.info(f"   Epochs: {num_epochs}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Training samples: {len(train_loader.dataset)}")
        logger.info(f"   Validation samples: {len(val_loader.dataset)}")
        logger.info("=" * 80)

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            try:
                # Training phase
                train_metrics = self.train_epoch(train_loader, epoch)

                # Validation phase
                val_metrics = self.validate_epoch(val_loader, epoch)

                # Update learning rate scheduler
                current_val_score = val_metrics['overall']['combined_accuracy']
                self.scheduler.step(current_val_score)

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time

                # Store history
                self.history['train_loss'].append(train_metrics['overall']['total_loss'])
                self.history['val_loss'].append(val_metrics['overall']['total_loss'])
                self.history['train_metrics'].append(train_metrics)
                self.history['val_metrics'].append(val_metrics)
                self.history['learning_rates'].append(current_lr)
                self.history['epoch_times'].append(epoch_time)

                # Print epoch results
                logger.info(f"\nEpoch {epoch + 1}/{num_epochs} Results:")
                logger.info(f"  Train - Loss: {train_metrics['overall']['total_loss']:.4f}")
                if 'vehicle' in train_metrics:
                    logger.info(f"    Vehicle: {train_metrics['vehicle']['accuracy']:.1f}% | "
                               f"Size: {train_metrics['size']['accuracy']:.1f}%")
                logger.info(f"  Val   - Loss: {val_metrics['overall']['total_loss']:.4f}")
                if 'vehicle' in val_metrics:
                    logger.info(f"    Vehicle: {val_metrics['vehicle']['accuracy']:.1f}% | "
                               f"Size: {val_metrics['size']['accuracy']:.1f}%")
                logger.info(f"    Combined Accuracy: {current_val_score:.1f}%")
                logger.info(f"  Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")

                # Save best model
                if current_val_score > self.best_val_score:
                    self.best_val_score = current_val_score
                    try:
                        ModelFactory.save_model(
                            self.model, BEST_MODEL_PATH, epoch + 1,
                            train_metrics['overall']['total_loss'],
                            val_metrics['overall']['total_loss'],
                            current_val_score, self.optimizer, self.scheduler, val_metrics
                        )
                        logger.info(f"  ✅ New best model saved! Combined Accuracy: {current_val_score:.1f}%")
                    except Exception as e:
                        logger.error(f"  ❌ Failed to save model: {e}")

                # Early stopping check
                if self.early_stopping(current_val_score, self.model):
                    logger.info(f"  ⏹️ Early stopping triggered at epoch {epoch + 1}")
                    break

            except Exception as e:
                logger.error(f"❌ Error in epoch {epoch + 1}: {e}")
                continue

        # Training completed
        total_time = time.time() - start_time
        logger.info(f"\n🎉 Vehicleverse Training Completed!")
        logger.info(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        logger.info(f"   Best combined accuracy: {self.best_val_score:.1f}%")
        logger.info(f"   Best model saved to: {BEST_MODEL_PATH}")

        # Save training history
        self._save_training_history()

        return self.history

    def _save_training_history(self):
        """Save comprehensive training history"""
        try:
            history_data = {
                'training_config': self.config,
                'model_config': MODEL_CONFIG,
                'best_val_score': float(self.best_val_score),
                'total_epochs': len(self.history['train_loss']),
                'total_training_time': float(sum(self.history['epoch_times'])),
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'history': {
                    'train_loss': [float(x) for x in self.history['train_loss']],
                    'val_loss': [float(x) for x in self.history['val_loss']],
                    'learning_rates': [float(x) for x in self.history['learning_rates']],
                    'epoch_times': [float(x) for x in self.history['epoch_times']]
                }
            }

            with open(TRAINING_HISTORY_PATH, 'w') as f:
                json.dump(history_data, f, indent=2)

            logger.info(f"💾 Training history saved to: {TRAINING_HISTORY_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to save training history: {e}")
