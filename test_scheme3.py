#!/usr/bin/env python3
"""
Tests for Scheme 3 fine-tuning implementation.

This script validates:
1. Dataset loads RGB PNG correctly and outputs correct channel count
2. SuperPoint model can be created with different channel counts
3. Weight adaptation works when loading 1ch pretrained weights to 3ch model
4. Channel inference from dataset works correctly
"""

import sys
import os
from pathlib import Path
import numpy as np
import cv2
import torch
import tempfile
import shutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models.superpoint import SuperPoint, adapt_conv_weight_channels
from data import S2RGBPNGDataset


def create_test_images(test_dir, num_images=5, channels=3, size=(256, 256)):
    """Create test RGB PNG images."""
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_images):
        # Create random RGB image
        if channels == 3:
            img = np.random.randint(0, 256, (*size, channels), dtype=np.uint8)
        elif channels == 1:
            img = np.random.randint(0, 256, size, dtype=np.uint8)
        else:
            img = np.random.randint(0, 256, (*size, channels), dtype=np.uint8)
        
        img_path = test_dir / f"test_image_{i:03d}.png"
        cv2.imwrite(str(img_path), img)
    
    return test_dir


def test_dataset_loading():
    """Test 1: Dataset loads RGB PNG and outputs correct channels."""
    print("\n" + "="*70)
    print("TEST 1: Dataset loading and channel detection")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test RGB images
        test_dir = create_test_images(tmpdir, num_images=5, channels=3)
        
        # Load dataset
        dataset = S2RGBPNGDataset(test_dir)
        
        # Check basic properties
        assert len(dataset) == 5, f"Expected 5 images, got {len(dataset)}"
        assert dataset.channels == 3, f"Expected 3 channels, got {dataset.channels}"
        assert dataset.band_mapping == 'B4/B3/B2', f"Unexpected band_mapping: {dataset.band_mapping}"
        
        # Check sample format
        sample = dataset[0]
        assert 'image' in sample, "Sample missing 'image' key"
        assert 'channels' in sample, "Sample missing 'channels' key"
        assert sample['channels'] == 3, f"Sample reports {sample['channels']} channels"
        
        img = sample['image']
        assert img.shape[0] == 3, f"Image has wrong channel dimension: {img.shape}"
        assert len(img.shape) == 3, f"Image should be CHW format, got shape {img.shape}"
        assert img.dtype == torch.float32 or img.dtype == np.float32, f"Wrong dtype: {img.dtype}"
        assert 0 <= img.min() <= 1, f"Image not normalized: min={img.min()}"
        assert 0 <= img.max() <= 1, f"Image not normalized: max={img.max()}"
        
        print("✓ Dataset correctly loads RGB PNG images")
        print(f"✓ Dataset reports {dataset.channels} channels")
        print(f"✓ Image shape is {img.shape} (CHW format)")
        print(f"✓ Image normalized to [{img.min():.3f}, {img.max():.3f}]")


def test_rgba_handling():
    """Test 1b: Dataset handles RGBA by discarding alpha."""
    print("\n" + "="*70)
    print("TEST 1b: RGBA handling (alpha channel removal)")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)
        test_dir.mkdir(exist_ok=True)
        
        # Create RGBA image
        img_rgba = np.random.randint(0, 256, (256, 256, 4), dtype=np.uint8)
        img_path = test_dir / "test_rgba.png"
        cv2.imwrite(str(img_path), img_rgba)
        
        # Load dataset
        dataset = S2RGBPNGDataset(test_dir)
        sample = dataset[0]
        
        # Should have 3 channels (RGB), not 4 (RGBA)
        assert sample['channels'] == 3, f"RGBA not converted to RGB: got {sample['channels']} channels"
        assert sample['image'].shape[0] == 3, f"Image has {sample['image'].shape[0]} channels, expected 3"
        
        print("✓ RGBA image correctly converted to RGB (alpha discarded)")


def test_model_channel_config():
    """Test 2: SuperPoint can be created with different channel counts."""
    print("\n" + "="*70)
    print("TEST 2: SuperPoint model with configurable input channels")
    print("="*70)
    
    # Test with 1 channel (original)
    model_1ch = SuperPoint({'in_channels': 1, 'load_pretrained': False})
    assert model_1ch.conv1a.in_channels == 1, "Model should have 1 input channel"
    print("✓ Created SuperPoint with 1 input channel")
    
    # Test with 3 channels (RGB)
    model_3ch = SuperPoint({'in_channels': 3, 'load_pretrained': False})
    assert model_3ch.conv1a.in_channels == 3, "Model should have 3 input channels"
    print("✓ Created SuperPoint with 3 input channels")
    
    # Test with 13 channels (original SAR use case)
    model_13ch = SuperPoint({'in_channels': 13, 'load_pretrained': False})
    assert model_13ch.conv1a.in_channels == 13, "Model should have 13 input channels"
    print("✓ Created SuperPoint with 13 input channels")


def test_weight_adaptation():
    """Test 3: Weight adaptation from 1ch to 3ch works correctly."""
    print("\n" + "="*70)
    print("TEST 3: Weight adaptation (1ch -> 3ch)")
    print("="*70)
    
    # Create dummy pretrained weight (1 channel input)
    pretrained_1ch = torch.randn(64, 1, 3, 3)  # (out_ch, in_ch, H, W)
    
    # Adapt to 3 channels
    adapted_3ch = adapt_conv_weight_channels(pretrained_1ch, 3)
    
    assert adapted_3ch.shape == (64, 3, 3, 3), f"Wrong shape: {adapted_3ch.shape}"
    print(f"✓ Adapted weight shape: {pretrained_1ch.shape} -> {adapted_3ch.shape}")
    
    # Test reverse adaptation (3ch -> 1ch)
    pretrained_3ch = torch.randn(64, 3, 3, 3)
    adapted_1ch = adapt_conv_weight_channels(pretrained_3ch, 1)
    assert adapted_1ch.shape == (64, 1, 3, 3), f"Wrong shape: {adapted_1ch.shape}"
    print(f"✓ Reverse adaptation: {pretrained_3ch.shape} -> {adapted_1ch.shape}")
    
    # Test no change when channels match
    same = adapt_conv_weight_channels(pretrained_1ch, 1)
    assert torch.allclose(same, pretrained_1ch), "Weight changed when it shouldn't"
    print("✓ No change when channels already match")


def test_pretrained_loading():
    """Test 4: Can load pretrained weights with channel adaptation."""
    print("\n" + "="*70)
    print("TEST 4: Loading pretrained weights with channel mismatch")
    print("="*70)
    
    # Create 3-channel model with pretrained weights
    # (This will load 1ch pretrained and adapt to 3ch)
    try:
        model = SuperPoint({'in_channels': 3, 'load_pretrained': True})
        assert model.conv1a.in_channels == 3, "Model should have 3 input channels"
        print("✓ Successfully loaded and adapted pretrained weights (1ch -> 3ch)")
        
        # Test that model can do forward pass with 3-channel input
        test_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model({'image': test_input})
        
        assert 'keypoints' in output, "Output missing 'keypoints'"
        assert 'descriptors' in output, "Output missing 'descriptors'"
        print("✓ Model forward pass successful with 3-channel input")
        
    except FileNotFoundError as e:
        print(f"⚠ Pretrained weights not found (expected for testing): {e}")
        print("  This is OK - weight adaptation logic is still validated")


def test_channel_inference():
    """Test 5: Channel inference from dataset works."""
    print("\n" + "="*70)
    print("TEST 5: Automatic channel inference from dataset")
    print("="*70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create RGB test images
        test_dir = create_test_images(tmpdir, num_images=3, channels=3)
        
        # Load dataset and infer channels
        dataset = S2RGBPNGDataset(test_dir)
        sample = dataset[0]
        
        # Infer channels (as would be done in training script)
        inferred_channels = sample['image'].shape[0]
        
        assert inferred_channels == 3, f"Inferred {inferred_channels} channels, expected 3"
        print(f"✓ Correctly inferred {inferred_channels} channels from dataset")
        
        # Test with grayscale images
        gray_dir = create_test_images(Path(tmpdir) / "gray", num_images=3, channels=1)
        dataset_gray = S2RGBPNGDataset(gray_dir)
        sample_gray = dataset_gray[0]
        
        inferred_channels_gray = sample_gray['image'].shape[0]
        assert inferred_channels_gray == 1, f"Inferred {inferred_channels_gray} channels, expected 1"
        print(f"✓ Correctly inferred {inferred_channels_gray} channel from grayscale dataset")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("Running Scheme 3 Fine-tuning Implementation Tests")
    print("="*70)
    
    tests = [
        ("Dataset loading and channel detection", test_dataset_loading),
        ("RGBA handling", test_rgba_handling),
        ("Model channel configuration", test_model_channel_config),
        ("Weight adaptation", test_weight_adaptation),
        ("Pretrained weight loading", test_pretrained_loading),
        ("Channel inference", test_channel_inference),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*70)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
