#!/usr/bin/env python3
"""
Download and setup CIFAR-100-C dataset.

CIFAR-100-C is not available in TensorFlow Datasets.
This script downloads it from Zenodo and extracts it to the correct location.
"""

import os
import sys
import tarfile
import urllib.request
from pathlib import Path

# Configuration
ZENODO_URL = "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar"
PROJECT_ROOT = Path(__file__).parent
DATASET_DIR = PROJECT_ROOT / "dataset"
CIFAR100C_DIR = DATASET_DIR / "CIFAR-100-C"
TAR_FILE = DATASET_DIR / "CIFAR-100-C.tar"

# Expected files
EXPECTED_FILES = [
    "labels.npy",
    "brightness.npy",
    "contrast.npy",
    "defocus_blur.npy",
    "elastic_transform.npy",
    "fog.npy",
    "frost.npy",
    "gaussian_blur.npy",
    "gaussian_noise.npy",
    "glass_blur.npy",
    "impulse_noise.npy",
    "jpeg_compression.npy",
    "motion_blur.npy",
    "pixelate.npy",
    "saturate.npy",
    "shot_noise.npy",
    "snow.npy",
    "spatter.npy",
    "speckle_noise.npy",
    "zoom_blur.npy",
]


def check_already_downloaded():
    """Check if CIFAR-100-C is already downloaded and extracted."""
    if not CIFAR100C_DIR.exists():
        return False
    
    missing_files = []
    for filename in EXPECTED_FILES:
        if not (CIFAR100C_DIR / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"⚠️  CIFAR-100-C directory exists but {len(missing_files)} files are missing:")
        for f in missing_files[:5]:
            print(f"   - {f}")
        if len(missing_files) > 5:
            print(f"   ... and {len(missing_files) - 5} more")
        return False
    
    print("✅ CIFAR-100-C already downloaded and complete!")
    print(f"   Location: {CIFAR100C_DIR}")
    print(f"   Files: {len(EXPECTED_FILES)} corruption files + labels")
    return True


def download_file(url, destination):
    """Download file with progress bar."""
    print(f"\n📥 Downloading CIFAR-100-C...")
    print(f"   From: {url}")
    print(f"   To: {destination}")
    print(f"   Size: ~2.7 GB (this may take a while)")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        
        bar_length = 50
        filled = int(bar_length * downloaded / total_size)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f'\r   [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='')
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, show_progress)
        print("\n✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def extract_tar(tar_path, dest_dir):
    """Extract tar file."""
    print(f"\n📦 Extracting CIFAR-100-C...")
    print(f"   This may take a few minutes...")
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(dest_dir)
        print("✅ Extraction complete!")
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def verify_installation():
    """Verify all expected files are present."""
    print(f"\n🔍 Verifying installation...")
    
    if not CIFAR100C_DIR.exists():
        print(f"❌ Directory not found: {CIFAR100C_DIR}")
        return False
    
    missing_files = []
    for filename in EXPECTED_FILES:
        filepath = CIFAR100C_DIR / filename
        if not filepath.exists():
            missing_files.append(filename)
        else:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"   ✓ {filename} ({size_mb:.1f} MB)")
    
    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} files:")
        for f in missing_files:
            print(f"   - {f}")
        return False
    
    print(f"\n✅ All {len(EXPECTED_FILES)} files verified!")
    return True


def cleanup_tar():
    """Remove tar file after extraction."""
    if TAR_FILE.exists():
        print(f"\n🧹 Cleaning up...")
        print(f"   Removing: {TAR_FILE}")
        TAR_FILE.unlink()
        print("✅ Cleanup complete!")


def main():
    print("=" * 70)
    print("CIFAR-100-C Download and Setup")
    print("=" * 70)
    
    # Check if already downloaded
    if check_already_downloaded():
        response = input("\n🤔 CIFAR-100-C already exists. Re-download? (y/N): ")
        if response.lower() != 'y':
            print("\n👍 Using existing CIFAR-100-C installation.")
            return 0
        print("\n🔄 Re-downloading CIFAR-100-C...")
    
    # Create dataset directory if needed
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download
    if not download_file(ZENODO_URL, TAR_FILE):
        return 1
    
    # Extract
    if not extract_tar(TAR_FILE, DATASET_DIR):
        return 1
    
    # Verify
    if not verify_installation():
        print("\n⚠️  Installation may be incomplete. Please check the files manually.")
        return 1
    
    # Cleanup
    cleanup_tar()
    
    # Success message
    print("\n" + "=" * 70)
    print("🎉 CIFAR-100-C Setup Complete!")
    print("=" * 70)
    print(f"\n📁 Location: {CIFAR100C_DIR}")
    print(f"📊 Files: {len(EXPECTED_FILES)} corruption types + labels")
    print(f"💾 Total size: ~13 GB")
    
    print("\n🚀 Next steps:")
    print("   1. Run experiments: python cifar100_experiments_config.py")
    print("   2. Run KL divergence: python scripts/cifar100_cifar100c_kldiv.py")
    print("   3. See guide: CIFAR100C_DOWNLOAD_GUIDE.md")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)

