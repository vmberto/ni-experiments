"""
Quick test to verify CIFAR-100 setup is working correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all CIFAR-100 modules can be imported."""
    print("Testing imports...")
    
    try:
        from dataset.cifar100dataset import Cifar100Dataset
        print("✅ Cifar100Dataset imported")
    except ImportError as e:
        print(f"❌ Failed to import Cifar100Dataset: {e}")
        return False
    
    try:
        from lib.consts import CIFAR100_CORRUPTIONS
        print(f"✅ CIFAR100_CORRUPTIONS imported ({len(CIFAR100_CORRUPTIONS)} corruptions)")
    except ImportError as e:
        print(f"❌ Failed to import CIFAR100_CORRUPTIONS: {e}")
        return False
    
    return True


def test_dataset():
    """Test CIFAR-100 dataset loading."""
    print("\nTesting dataset loading...")
    
    try:
        from keras import datasets
        
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
        
        print(f"✅ CIFAR-100 loaded successfully")
        print(f"   - Train: {x_train.shape}, {y_train.shape}")
        print(f"   - Test: {x_test.shape}, {y_test.shape}")
        print(f"   - Classes: {len(set(y_train.flatten()))} unique classes")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load CIFAR-100: {e}")
        return False


def test_corrupted():
    """Test CIFAR-100-C availability."""
    print("\nTesting CIFAR-100-C corrupted dataset...")
    
    try:
        import tensorflow_datasets as tfds
        
        # Try loading one corruption
        ds = tfds.load('cifar100_corrupted/gaussian_noise_1', split='test', as_supervised=True)
        
        # Count samples
        count = sum(1 for _ in ds.take(10))
        
        print(f"✅ CIFAR-100-C available (tested gaussian_noise_1)")
        print(f"   - Successfully loaded {count} samples")
        
        return True
        
    except Exception as e:
        print(f"⚠️  CIFAR-100-C not available (will download on first use)")
        print(f"   Error: {e}")
        return False


def test_config():
    """Test configuration file."""
    print("\nTesting configuration...")
    
    try:
        import cifar100_experiments_config
        
        print(f"✅ Configuration loaded")
        print(f"   - Input shape: {cifar100_experiments_config.INPUT_SHAPE}")
        print(f"   - Batch size: {cifar100_experiments_config.BATCH_SIZE}")
        print(f"   - K-folds: {cifar100_experiments_config.KFOLD_N_SPLITS}")
        print(f"   - Strategies: {len(cifar100_experiments_config.CONFIGS)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False


def main():
    print("=" * 70)
    print("CIFAR-100 Setup Verification")
    print("=" * 70)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Dataset Loading", test_dataset()))
    results.append(("Corrupted Dataset", test_corrupted()))
    results.append(("Configuration", test_config()))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! CIFAR-100 is ready to use.")
        print("\nNext steps:")
        print("  1. Run KL divergence: python scripts/cifar100_cifar100c_kldiv.py")
        print("  2. Run experiments: python cifar100_experiments_config.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

