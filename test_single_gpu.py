#!/usr/bin/env python3
"""
Test script for single GPU mode
"""
import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_single_gpu():
    print("Testing single GPU mode...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return False
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Test basic tensor operations
    try:
        x = torch.randn(2, 3, 64, 64).cuda()
        y = torch.randn(2, 3, 64, 64).cuda()
        z = x + y
        print("Basic tensor operations on GPU: OK")
        return True
    except Exception as e:
        print(f"Error in tensor operations: {e}")
        return False

if __name__ == "__main__":
    success = test_single_gpu()
    if success:
        print("Single GPU test passed!")
    else:
        print("Single GPU test failed!") 