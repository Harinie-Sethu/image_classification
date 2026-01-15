"""
Part 1: Model Loading and Parameter Comparison

This script loads ResNet-50 (ImageNet pretrained) and CLIP models,
and compares their architectures and parameter counts.

- ResNet-50 has been pretrained on ImageNet dataset
- CLIP uses ResNet-50 as its visual encoder but with different architecture
- CLIP's visual encoder is trained jointly with a text encoder
- The architectures differ, which is reflected in parameter counts
CLIP = 38316896 <br>
R-50 = 25557032

"""

import torch
import torchvision.models as models
import clip
from utils import get_device


def load_models(device):
    """
    Load ResNet-50 (ImageNet) and CLIP models.
    
    Args:
        device: Computation device ('cuda' or 'cpu')
        
    Returns:
        resnet50_imagenet: ResNet-50 model pretrained on ImageNet
        clip_model: Full CLIP model
        clip_visual_encoder: CLIP's visual encoder (ResNet-50 based)
    """
    print(f"Loading models on device: {device}")
    
    # Load ResNet-50 model with ImageNet pretraining
    print("Loading ResNet-50 (ImageNet pretrained)...")
    resnet50_imagenet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet50_imagenet.eval()
    resnet50_imagenet.to(device)
    
    # Load CLIP model
    print("Loading CLIP model (RN50)...")
    clip_model, _ = clip.load("RN50")
    clip_model.eval()
    clip_model.to(device)
    clip_visual_encoder = clip_model.visual
    
    return resnet50_imagenet, clip_model, clip_visual_encoder


def count_parameters(model):
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        num_params: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def compare_models(resnet50_imagenet, clip_visual_encoder):
    """
    Compare ResNet-50 and CLIP visual encoder parameter counts.
    
    Args:
        resnet50_imagenet: ResNet-50 model
        clip_visual_encoder: CLIP's visual encoder
    """
    clip_params = count_parameters(clip_visual_encoder)
    r50_params = count_parameters(resnet50_imagenet)
    
    print("\n" + "="*60)
    print("MODEL PARAMETER COMPARISON")
    print("="*60)
    print(f"CLIP Visual Encoder (RN50): {clip_params:,} parameters")
    print(f"ResNet-50 (ImageNet):       {r50_params:,} parameters")
    print(f"Difference:                 {clip_params - r50_params:,} parameters")
    print("="*60)
    
    return clip_params, r50_params


def main():
    """Main function to run Part 1."""
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Load models
    resnet50_imagenet, clip_model, clip_visual_encoder = load_models(device)
    
    # Compare parameters
    clip_params, r50_params = compare_models(resnet50_imagenet, clip_visual_encoder)
    
    print("\nKey Observations:")
    print("- CLIP has more parameters than standard ResNet-50")
    print("- This is because CLIP's architecture is optimized for joint image-text learning")
    print("- CLIP's visual encoder is trained to work with text encodings")
    print("- The architecture differences reflect the different training objectives")
    
    return resnet50_imagenet, clip_model, clip_visual_encoder


if __name__ == "__main__":
    resnet50_imagenet, clip_model, clip_visual_encoder = main()


