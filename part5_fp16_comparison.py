"""
Part 5: FP16 vs FP32 Precision Comparison

This script compares the performance and efficiency of CLIP model
in FP32 (32-bit floating point) and FP16 (16-bit floating point) precision.

Key Metrics:
- Inference time (speed)
- Memory usage
- Accuracy (top-5 predictions)

Benefits of FP16:
- ~2x reduction in memory usage
- Faster inference on modern GPUs (Tensor Cores)
- Minimal accuracy loss for most tasks
"""

import torch
import torch.nn.functional as F
import clip
import time
import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from utils import get_device


def benchmark_inference_time(model, image_tensor, num_runs=100):
    """
    Benchmark inference time for a model.
    
    Args:
        model: Model to benchmark
        image_tensor: Input image tensor
        num_runs: Number of inference runs
        
    Returns:
        mean_time: Mean inference time in seconds
        std_time: Standard deviation of inference time
    """
    times = []
    
    # Warmup
    with torch.no_grad():
        _ = model(image_tensor)
    
    # Benchmark
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(image_tensor)
        times.append(time.time() - start_time)
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time


def compare_inference_times(clip_visual_fp32, clip_visual_fp16, image_tensor, num_runs=100):
    """
    Compare inference times between FP32 and FP16 models.
    
    Args:
        clip_visual_fp32: FP32 CLIP visual encoder
        clip_visual_fp16: FP16 CLIP visual encoder
        image_tensor: Input image tensor (FP32)
        num_runs: Number of benchmark runs
        
    Returns:
        fp32_mean, fp32_std: FP32 timing statistics
        fp16_mean, fp16_std: FP16 timing statistics
    """
    print("Benchmarking FP32 model...")
    fp32_mean, fp32_std = benchmark_inference_time(clip_visual_fp32, image_tensor, num_runs)
    
    print("Benchmarking FP16 model...")
    # Convert input to FP16 for FP16 model
    image_tensor_fp16 = image_tensor.half()
    fp16_mean, fp16_std = benchmark_inference_time(clip_visual_fp16, image_tensor_fp16, num_runs)
    
    print("\n" + "="*60)
    print("INFERENCE TIME COMPARISON")
    print("="*60)
    print(f"FP32 mean inference time: {fp32_mean:.4f} s, std dev: {fp32_std:.4f} s")
    print(f"FP16 mean inference time: {fp16_mean:.4f} s, std dev: {fp16_std:.4f} s")
    print(f"Speedup: {fp32_mean / fp16_mean:.2f}x")
    print("="*60)
    
    return fp32_mean, fp32_std, fp16_mean, fp16_std


def compare_predictions(clip_visual_fp32, clip_visual_fp16, image_tensor, 
                       text_features, labels_ten, device):
    """
    Compare predictions between FP32 and FP16 models.
    
    Args:
        clip_visual_fp32: FP32 CLIP visual encoder
        clip_visual_fp16: FP16 CLIP visual encoder
        image_tensor: Input image tensor
        text_features: Encoded text features
        labels_ten: List of class labels
        device: Computation device
    """
    # FP32 predictions
    with torch.no_grad():
        image_features_fp32 = clip_visual_fp32(image_tensor)
        cosine_sim_fp32 = F.cosine_similarity(
            image_features_fp32.unsqueeze(1),
            text_features.unsqueeze(0),
            dim=2
        )
        probs_fp32 = F.softmax(cosine_sim_fp32, dim=1)
        top5_fp32 = torch.topk(probs_fp32, k=5, dim=1).indices.squeeze().cpu().numpy()
    
    # FP16 predictions
    with torch.no_grad():
        image_tensor_fp16 = image_tensor.half()
        image_features_fp16 = clip_visual_fp16(image_tensor_fp16)
        # Convert back to FP32 for comparison
        image_features_fp16 = image_features_fp16.float()
        cosine_sim_fp16 = F.cosine_similarity(
            image_features_fp16.unsqueeze(1),
            text_features.unsqueeze(0),
            dim=2
        )
        probs_fp16 = F.softmax(cosine_sim_fp16, dim=1)
        top5_fp16 = torch.topk(probs_fp16, k=5, dim=1).indices.squeeze().cpu().numpy()
    
    print("\n" + "="*60)
    print("PREDICTION COMPARISON")
    print("="*60)
    print("\nFP32 Top-5 Predictions:")
    for idx in top5_fp32:
        print(f"  Class: {labels_ten[idx]}, Probability: {probs_fp32[0, idx]:.4f}")
    
    print("\nFP16 Top-5 Predictions:")
    for idx in top5_fp16:
        print(f"  Class: {labels_ten[idx]}, Probability: {probs_fp16[0, idx]:.4f}")
    
    # Check if rankings match
    rankings_match = np.array_equal(top5_fp32, top5_fp16)
    print(f"\nTop-5 rankings match: {rankings_match}")
    print("="*60)
    
    return top5_fp32, top5_fp16, probs_fp32, probs_fp16


def compare_memory_usage(clip_visual_fp32, clip_visual_fp16, image_tensor, device):
    """
    Compare memory usage between FP32 and FP16 models.
    
    Args:
        clip_visual_fp32: FP32 CLIP visual encoder
        clip_visual_fp16: FP16 CLIP visual encoder
        image_tensor: Input image tensor
        device: Computation device
    """
    if device != 'cuda':
        print("Memory comparison requires CUDA device")
        return None, None
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    
    # FP32 forward pass
    with torch.no_grad():
        _ = clip_visual_fp32(image_tensor)
    fp32_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Reset and test FP16
    torch.cuda.reset_peak_memory_stats()
    image_tensor_fp16 = image_tensor.half()
    
    with torch.no_grad():
        _ = clip_visual_fp16(image_tensor_fp16)
    fp16_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print("\n" + "="*60)
    print("MEMORY USAGE COMPARISON")
    print("="*60)
    print(f"FP32 peak memory: {fp32_memory:.2f} MB")
    print(f"FP16 peak memory: {fp16_memory:.2f} MB")
    print(f"Memory reduction: {(1 - fp16_memory/fp32_memory)*100:.1f}%")
    print("="*60)
    
    return fp32_memory, fp16_memory


def main(image_path, categories_path=None, labels_ten=None, device=None):
    """
    Main function to compare FP32 and FP16 models.
    
    Args:
        image_path: Path to test image
        categories_path: Path to ImageNet labels (optional)
        labels_ten: List of 10 labels to test (optional)
        device: Computation device (auto-detected if None)
    """
    if device is None:
        device = get_device()
    
    if device != 'cuda':
        print("Warning: FP16 benefits are most apparent on CUDA devices")
    
    print(f"Using device: {device}\n")
    
    # Load CLIP model
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load("RN50", device=device)
    clip_model.eval()
    clip_visual_fp32 = clip_model.visual
    
    # Create FP16 version
    print("Creating FP16 model...")
    clip_visual_fp16 = clip_visual_fp32.half()
    clip_visual_fp16.eval()
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title("Test Image")
    plt.axis('off')
    plt.show()
    
    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    
    # Compare inference times
    print("\n" + "="*80)
    print("PART 5: FP16 vs FP32 COMPARISON")
    print("="*80)
    
    fp32_mean, fp32_std, fp16_mean, fp16_std = compare_inference_times(
        clip_visual_fp32, clip_visual_fp16, image_tensor, num_runs=100
    )
    
    # Compare predictions (if labels provided)
    if labels_ten is not None:
        # Encode text labels
        text_features = []
        for category in labels_ten:
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {category}")])
            text_inputs = text_inputs.to(device)
            text_feature = clip_model.encode_text(text_inputs)
            text_features.append(text_feature)
        text_features = torch.cat(text_features)
        
        compare_predictions(
            clip_visual_fp32, clip_visual_fp16, image_tensor,
            text_features, labels_ten, device
        )
    
    # Compare memory usage (CUDA only)
    if device == 'cuda':
        compare_memory_usage(clip_visual_fp32, clip_visual_fp16, image_tensor, device)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Key Observations:")
    print("1. FP16 provides faster inference with lower standard deviation")
    print("2. Top-5 rankings typically match between FP32 and FP16")
    print("3. Probability scores may differ slightly (FP32 is more accurate)")
    print("4. FP16 uses significantly less memory (~50% reduction)")
    print("5. Good trade-off between memory/speed and accuracy")
    print("="*80)
    
    return {
        'fp32_mean': fp32_mean,
        'fp32_std': fp32_std,
        'fp16_mean': fp16_mean,
        'fp16_std': fp16_std
    }


if __name__ == "__main__":
    print("Part 5: FP16 vs FP32 Comparison")
    print("="*60)
    print("To use this script, provide:")
    print("1. Path to test image")
    print("2. (Optional) List of 10 labels to test against")
    print("\nExample:")
    print("  image_path = '/path/to/test_image.jpg'")
    print("  labels_ten = ['bald eagle', 'axolotl', ...]")
    print("  main(image_path, labels_ten=labels_ten)")

