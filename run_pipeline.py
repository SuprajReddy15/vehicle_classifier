"""Vehicleverse Complete Training Pipeline - Updated Version"""
import logging
import sys
import os
from pathlib import Path
from config import *
from dataset_manager import VehicleverseDatasetManager
from trainer import VehicleverseTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all requirements are met"""
    logger.info("🔍 Checking requirements...")

    # Check if data directory exists
    if not DATA_DIR.exists():
        logger.error(f"❌ Data directory not found: {DATA_DIR}")
        return False

    # Get actual folders in the data directory
    actual_folders = []
    if DATA_DIR.exists():
        for item in DATA_DIR.iterdir():
            if item.is_dir():
                actual_folders.append(item.name)

    logger.info(f"📁 Found folders: {actual_folders}")

    # Check if we have at least some vehicle folders
    if not actual_folders:
        logger.error("❌ No folders found in data directory!")
        logger.error("   Please create folders and add vehicle images:")
        logger.error(f"     Example: {DATA_DIR}/car/")
        logger.error(f"     Example: {DATA_DIR}/truck/")
        return False

    # Check if there are images in folders
    total_images = 0
    for folder_name in actual_folders:
        folder_path = DATA_DIR / folder_name
        if folder_path.is_dir():
            image_count = len([f for f in folder_path.iterdir()
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']])
            total_images += image_count
            logger.info(f"   {folder_name}: {image_count} images")

    if total_images == 0:
        logger.error("❌ No images found in vehicle folders!")
        return False

    logger.info(f"✅ Requirements check passed! Total images: {total_images}")
    return True

def main():
    """Complete Vehicleverse training pipeline"""
    logger.info("🚀 Starting Vehicleverse Complete Training Pipeline...")
    logger.info("=" * 80)

    try:
        # Step 0: Check requirements
        logger.info("🔍 Step 0: Checking Requirements...")
        if not check_requirements():
            return False

        # Step 1: Initialize dataset manager
        logger.info("📊 Step 1: Initializing Dataset Manager...")
        dataset_manager = VehicleverseDatasetManager()

        # Step 2: Load and validate dataset
        logger.info("📂 Step 2: Loading Dataset...")
        if not dataset_manager.load_dataset(validate_images=True):
            logger.error("❌ Failed to load dataset. Please check your data directory.")
            logger.error(f"   Expected data directory: {DATA_DIR}")
            logger.error("   Make sure you have vehicle images organized in folders:")
            for vehicle_type in VEHICLE_CLASSES:
                logger.error(f"     - {DATA_DIR}/{vehicle_type}/")
            return False

        # Step 3: Create data loaders
        logger.info("🔄 Step 3: Creating Data Loaders...")
        train_loader, val_loader, test_loader = dataset_manager.create_data_loaders(
            train_split=TRAINING_CONFIG['train_split'],
            val_split=TRAINING_CONFIG['val_split'],
            test_split=TRAINING_CONFIG['test_split'],
            batch_size=TRAINING_CONFIG['batch_size'],
            num_workers=0  # Set to 0 for Windows compatibility
        )

        if train_loader is None or val_loader is None:
            logger.error("❌ Failed to create data loaders.")
            return False

        # Step 4: Initialize trainer
        logger.info("🎯 Step 4: Initializing Trainer...")
        trainer = VehicleverseTrainer()

        # Step 5: Start training
        logger.info("🏋️ Step 5: Starting Training...")
        history = trainer.train(train_loader, val_loader)

        # Step 6: Final model check
        logger.info("🧪 Step 6: Final Model Check...")
        if os.path.exists(BEST_MODEL_PATH):
            logger.info(f"✅ Best model saved successfully: {BEST_MODEL_PATH}")
        else:
            logger.warning("⚠️ Best model file not found, but training completed")

        logger.info("🎉 Vehicleverse Training Pipeline Completed Successfully!")
        logger.info("=" * 80)
        logger.info("📋 Summary:")
        logger.info(f"   Best Combined Accuracy: {trainer.best_val_score:.1f}%")
        logger.info(f"   Model saved to: {BEST_MODEL_PATH}")
        logger.info(f"   Training history: {TRAINING_HISTORY_PATH}")
        logger.info("   Ready to run web application: python app.py")

        return True

    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("❌ Training pipeline failed!")
        sys.exit(1)
    else:
        logger.info("✅ Training pipeline completed successfully!")
