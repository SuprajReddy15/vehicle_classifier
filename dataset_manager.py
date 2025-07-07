"""Vehicleverse Dataset Manager - Simplified for Type + Size Classification"""
import os
import json
import random
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFile
import numpy as np
from sklearn.model_selection import train_test_split
from config import *

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VehicleverseDataset(Dataset):
    """Enhanced dataset class for Vehicleverse classification"""

    def __init__(self, image_paths: List[str], labels: List[Dict],
                 transform=None, augment: bool = False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment

        assert len(image_paths) == len(labels), "Mismatch between images and labels"
        logger.info(f"📊 VehicleverseDataset created with {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        try:
            image = Image.open(image_path).convert('RGB')

            # Validate image size
            if min(image.size) < 32:  # Minimum size check
                logger.warning(f"Small image detected: {image_path} - {image.size}")

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Create a black fallback image
            image = Image.new('RGB', IMAGE_CONFIG['input_size'], color='black')

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                logger.error(f"Transform failed for {image_path}: {e}")
                # Create tensor fallback
                image = torch.zeros(3, *IMAGE_CONFIG['input_size'])

        # Get labels and ensure they are proper dictionaries
        label = self.labels[idx]

        # Convert to tensors and ensure correct types
        label_tensors = {
            'vehicle': torch.tensor(label['vehicle'], dtype=torch.long),
            'size': torch.tensor(label['size'], dtype=torch.long)
        }

        return image, label_tensors

class VehicleverseDatasetManager:
    """Advanced dataset manager with comprehensive data handling capabilities"""

    def __init__(self, data_dir=None, seed: int = 42):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.seed = seed
        self.image_paths = []
        self.labels = []
        self.dataset_stats = {}

        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.train_transform = self._create_train_transforms()
        self.val_transform = self._create_val_transforms()

        logger.info(f"🔧 VehicleverseDatasetManager initialized")
        logger.info(f"   Data directory: {self.data_dir}")

    def _create_train_transforms(self):
        """Create training transforms with augmentation"""
        return transforms.Compose([
            transforms.Resize(IMAGE_CONFIG['resize_size']),
            transforms.RandomCrop(IMAGE_CONFIG['input_size']),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_CONFIG['mean'], std=IMAGE_CONFIG['std']),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])

    def _create_val_transforms(self):
        """Create validation transforms without augmentation"""
        return transforms.Compose([
            transforms.Resize(IMAGE_CONFIG['resize_size']),
            transforms.CenterCrop(IMAGE_CONFIG['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_CONFIG['mean'], std=IMAGE_CONFIG['std'])
        ])

    def load_dataset(self, validate_images: bool = True) -> bool:
        """Load and organize the complete dataset"""
        logger.info("📊 Loading Vehicleverse dataset...")

        if not self.data_dir.exists():
            logger.error(f"❌ Data directory not found: {self.data_dir}")
            return False

        # Get actual vehicle folders from the directory
        actual_vehicle_folders = []
        vehicle_class_mapping = {}

        for item in self.data_dir.iterdir():
            if item.is_dir():
                folder_name = item.name
                # Check if folder has images
                image_count = len([f for f in item.iterdir()
                                  if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']])
                if image_count > 0:
                    actual_vehicle_folders.append(folder_name)
                    vehicle_class_mapping[folder_name] = len(actual_vehicle_folders) - 1

        if not actual_vehicle_folders:
            logger.error("❌ No vehicle folders with images found!")
            return False

        logger.info(f"📁 Detected vehicle types: {actual_vehicle_folders}")

        # Update global vehicle classes
        global VEHICLE_CLASSES
        VEHICLE_CLASSES = actual_vehicle_folders

        self.image_paths = []
        self.labels = []

        vehicle_counts = Counter()
        size_counts = Counter()
        corrupted_images = []

        for vehicle_idx, vehicle_type in enumerate(actual_vehicle_folders):
            vehicle_folder = self.data_dir / vehicle_type

            # Get all image files
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                image_files.extend(list(vehicle_folder.glob(f"*{ext}")))
                image_files.extend(list(vehicle_folder.glob(f"*{ext.upper()}")))

            logger.info(f"   {vehicle_type}: {len(image_files)} images found")

            for image_path in image_files:
                if validate_images and not self._validate_image(image_path):
                    corrupted_images.append(str(image_path))
                    continue

                # Generate size labels based on vehicle type and filename
                size_idx = self._predict_size_from_vehicle_and_filename(vehicle_type, image_path.name)

                label = {
                    'vehicle': vehicle_idx,
                    'size': size_idx,
                    'vehicle_name': vehicle_type,
                    'size_name': SIZE_CLASSES[size_idx],
                    'image_path': str(image_path)
                }

                self.image_paths.append(str(image_path))
                self.labels.append(label)

                vehicle_counts[vehicle_type] += 1
                size_counts[SIZE_CLASSES[size_idx]] += 1

        self.dataset_stats = {
            'total_images': len(self.image_paths),
            'corrupted_images': len(corrupted_images),
            'vehicle_distribution': dict(vehicle_counts),
            'size_distribution': dict(size_counts),
            'vehicle_classes': actual_vehicle_folders,
            'size_classes': SIZE_CLASSES
        }

        if len(self.image_paths) == 0:
            logger.error("❌ No valid images found in dataset!")
            return False

        logger.info(f"✅ Dataset loaded successfully!")
        logger.info(f"   Total images: {len(self.image_paths)}")
        logger.info(f"   Corrupted images: {len(corrupted_images)}")

        return True

    def _validate_image(self, image_path: Path) -> bool:
        """Validate image file"""
        try:
            with Image.open(image_path) as img:
                img.verify()

            # Check file size (max 16MB)
            if image_path.stat().st_size > IMAGE_CONFIG['max_file_size']:
                return False

            # Re-open to check dimensions
            with Image.open(image_path) as img:
                if min(img.size) < IMAGE_CONFIG['quality_threshold']:
                    return False

            return True
        except Exception:
            return False

    def _predict_size_from_vehicle_and_filename(self, vehicle_type: str, filename: str) -> int:
        """Predict size based on vehicle type and filename"""
        filename_lower = filename.lower()
        vehicle_lower = vehicle_type.lower()

        # Size keywords in filename
        if any(word in filename_lower for word in ['compact', 'small', 'mini', 'city']):
            return SIZE_CLASSES.index('compact')
        elif any(word in filename_lower for word in ['large', 'big', 'xl', 'extended']):
            return SIZE_CLASSES.index('large')
        elif any(word in filename_lower for word in ['extra', 'xxl', 'heavy', 'commercial']):
            return SIZE_CLASSES.index('extra-large')

        # Default size based on vehicle type
        if 'bicycle' in vehicle_lower:
            return SIZE_CLASSES.index('compact')
        elif 'car' in vehicle_lower:
            return random.choice([SIZE_CLASSES.index('compact'), SIZE_CLASSES.index('mid-size')])
        elif 'motor' in vehicle_lower or 'bike' in vehicle_lower:
            return SIZE_CLASSES.index('compact')
        elif 'truck' in vehicle_lower:
            return random.choice([SIZE_CLASSES.index('large'), SIZE_CLASSES.index('extra-large')])
        else:
            return SIZE_CLASSES.index('mid-size')

    def create_data_loaders(self, train_split: float = 0.8, val_split: float = 0.15,
                           test_split: float = 0.05, batch_size: int = 16,
                           num_workers: int = 0):  # Changed default to 0 for Windows
        """Create comprehensive data loaders with train/val/test splits"""
        if len(self.image_paths) == 0:
            logger.error("❌ No dataset loaded. Call load_dataset() first.")
            return None, None, None

        if abs(train_split + val_split + test_split - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")

        logger.info(f"🔄 Creating data loaders...")
        logger.info(f"   Train split: {train_split:.1%}")
        logger.info(f"   Validation split: {val_split:.1%}")
        logger.info(f"   Test split: {test_split:.1%}")

        indices = list(range(len(self.image_paths)))
        vehicle_labels = [self.labels[i]['vehicle'] for i in indices]

        # Create stratified splits
        train_indices, temp_indices = train_test_split(
            indices, test_size=(val_split + test_split),
            stratify=vehicle_labels, random_state=self.seed
        )

        if test_split > 0:
            temp_vehicle_labels = [vehicle_labels[i] for i in temp_indices]
            val_indices, test_indices = train_test_split(
                temp_indices, test_size=test_split / (val_split + test_split),
                stratify=temp_vehicle_labels, random_state=self.seed
            )
        else:
            val_indices = temp_indices
            test_indices = []

        # Create datasets
        train_paths = [self.image_paths[i] for i in train_indices]
        train_labels = [self.labels[i] for i in train_indices]
        val_paths = [self.image_paths[i] for i in val_indices]
        val_labels = [self.labels[i] for i in val_indices]

        train_dataset = VehicleverseDataset(
            train_paths, train_labels, transform=self.train_transform, augment=True
        )
        val_dataset = VehicleverseDataset(
            val_paths, val_labels, transform=self.val_transform, augment=False
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=True, persistent_workers=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
            drop_last=False, persistent_workers=False
        )

        test_loader = None
        if test_indices:
            test_paths = [self.image_paths[i] for i in test_indices]
            test_labels = [self.labels[i] for i in test_indices]
            test_dataset = VehicleverseDataset(
                test_paths, test_labels, transform=self.val_transform, augment=False
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                drop_last=False, persistent_workers=False
            )

        logger.info(f"✅ Data loaders created!")
        logger.info(f"   Training samples: {len(train_dataset)}")
        logger.info(f"   Validation samples: {len(val_dataset)}")
        if test_loader:
            logger.info(f"   Test samples: {len(test_loader.dataset)}")

        return train_loader, val_loader, test_loader
