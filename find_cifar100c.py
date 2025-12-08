#!/usr/bin/env python3
"""
Quick script to find where CIFAR-100-C is located.
"""

import os
from pathlib import Path

# Possible locations
locations = [
    './dataset/CIFAR-100-C',
    '../dataset/CIFAR-100-C',
    '../../dataset/CIFAR-100-C',
    Path(__file__).parent / 'dataset' / 'CIFAR-100-C',
    Path(__file__).parent.parent / 'dataset' / 'CIFAR-100-C',
    Path.home() / 'CIFAR-100-C',
    '/tmp/CIFAR-100-C',
]

print("Searching for CIFAR-100-C...")
print(f"Current working directory: {os.getcwd()}")
print(f"Script location: {Path(__file__).parent}")
print("\nChecking locations:")

found = False
for loc in locations:
    path = Path(loc).resolve()
    exists = path.exists()
    print(f"  {'✓' if exists else '✗'} {path}")
    if exists and not found:
        print(f"\n✅ Found CIFAR-100-C at: {path}")
        # List files
        files = list(path.glob('*.npy'))
        print(f"   Files: {len(files)} .npy files")
        if files:
            print(f"   Sample files:")
            for f in sorted(files)[:5]:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"     - {f.name} ({size_mb:.1f} MB)")
        found = True

if not found:
    print("\n❌ CIFAR-100-C not found in any standard location")
    print("\nPlease tell me where it is, or download it:")
    print("  python download_cifar100c.py")

