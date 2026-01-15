"""
Part 4: ResNet-50 vs CLIP Model Comparison

This script compares the performance of ResNet-50 (ImageNet pretrained) 
and CLIP on various image types including:
- Original ImageNet images
- ImageNet Sketch (sketch/drawing versions)
- Artistic/stylized versions
- Other distribution shifts

Key Observations:
- CLIP performs better on distribution shifts (sketches, art)
- ResNet-50 performs better on original ImageNet images (slight advantage)
- CLIP's zero-shot learning provides robustness to domain shifts
"""

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import clip
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
from utils import get_device, load_imagenet_labels, get_resnet_transform


def evaluate_resnet(resnet_model, image_paths, text_labels, device):
    """
    Evaluate ResNet-50 on images and return top-5 predictions.
    
    Args:
        resnet_model: ResNet-50 model
        image_paths: List of image paths
        text_labels: List of ImageNet category labels
        device: Computation device
        
    Returns:
        all_top5: List of top-5 predictions for each image
        all_probs: List of probability distributions for each image
    """
    transform = get_resnet_transform()
    all_top5 = []
    all_probs = []
    
    plt.figure(figsize=(15, 5))
    
    for i, image_path in enumerate(image_paths):
        # Load and display image
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(image)
        plt.title(f"Image {i+1}")
        plt.axis('off')
        
        # Preprocess for ResNet
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get predictions
        with torch.no_grad():
            resnet_outputs = resnet_model(image_tensor)
            probs = F.softmax(resnet_outputs, dim=1)
            _, top5_indices = torch.topk(resnet_outputs, k=5, dim=1)
            top5_indices = top5_indices.squeeze().cpu().numpy()
        
        all_top5.append(top5_indices)
        all_probs.append(probs.squeeze().cpu())
        
        # Print top-5 predictions
        print(f"\nTop 5 matches with ResNet for image {i+1}:")
        for idx in top5_indices:
            print(f"  Class: {text_labels[idx]}, Probability: {probs.squeeze()[idx]:.4f}")
    
    plt.tight_layout()
    plt.show()
    
    return all_top5, all_probs


def evaluate_clip(clip_model, image_paths, text_labels, labels_ten, device):
    """
    Evaluate CLIP on images and return top-5 predictions.
    
    Args:
        clip_model: CLIP model
        image_paths: List of image paths
        text_labels: Full list of ImageNet labels (for reference)
        labels_ten: List of 10 specific labels to test against
        device: Computation device
        
    Returns:
        all_top5: List of top-5 predictions for each image
        all_probs: List of probability distributions for each image
    """
    # Get CLIP preprocessing
    _, preprocess = clip.load("RN50", device=device)
    
    # Encode text labels
    text_features = []
    for category in labels_ten:
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {category}")])
        text_inputs = text_inputs.to(device)
        text_feature = clip_model.encode_text(text_inputs)
        text_features.append(text_feature)
    text_features = torch.cat(text_features)
    
    # Process images
    transformed_images = []
    plt.figure(figsize=(15, 5))
    
    for i, image_path in enumerate(image_paths):
        # Load and display image
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(image)
        plt.title(f"Image {i+1}")
        plt.axis('off')
        
        # Preprocess for CLIP
        image_pil = Image.fromarray(image)
        image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
        transformed_images.append(image_tensor)
    
    plt.tight_layout()
    plt.show()
    
    # Encode all images
    images_tensor = torch.cat(transformed_images)
    with torch.no_grad():
        image_features = clip_model.encode_image(images_tensor)
    
    # Compute similarities and probabilities
    cosine_similarities = F.cosine_similarity(
        image_features.unsqueeze(1), 
        text_features.unsqueeze(0), 
        dim=2
    )
    probabilities = F.softmax(cosine_similarities, dim=1)
    
    # Get top-5 for each image
    all_top5 = []
    all_probs = []
    
    for i in range(len(image_paths)):
        top5_indices = torch.topk(probabilities[i], k=5).indices.cpu().numpy()
        all_top5.append(top5_indices)
        all_probs.append(probabilities[i].cpu())
        
        print(f"\nTop 5 matches with CLIP for image {i+1}:")
        for idx in top5_indices:
            print(f"  Class: {labels_ten[idx]}, Similarity: {probabilities[i, idx]:.4f}")
    
    return all_top5, all_probs


def compare_class_performance(class_name, image_paths, resnet_model, clip_model, 
                              text_labels, labels_ten, device):
    """
    Compare ResNet and CLIP performance on a specific class.
    
    Args:
        class_name: Name of the class being tested
        image_paths: List of image paths (sketch, art, original)
        resnet_model: ResNet-50 model
        clip_model: CLIP model
        text_labels: Full ImageNet labels
        labels_ten: 10 specific labels being tested
        device: Computation device
    """
    print("\n" + "="*80)
    print(f"COMPARING PERFORMANCE: {class_name.upper()}")
    print("="*80)
    
    print("\n--- ResNet-50 Results ---")
    resnet_top5, resnet_probs = evaluate_resnet(
        resnet_model, image_paths, text_labels, device
    )
    
    print("\n--- CLIP Results ---")
    clip_top5, clip_probs = evaluate_clip(
        clip_model, image_paths, text_labels, labels_ten, device
    )
    
    return resnet_top5, resnet_probs, clip_top5, clip_probs


def main():
    """
    Main function to compare ResNet and CLIP on multiple classes.
    
    Note: Update image_paths with your actual data paths.
    The structure should be: [sketch_path, art_path, original_path]
    """
    device = get_device()
    print(f"Using device: {device}\n")
    
    # Load models
    print("Loading models...")
    resnet50_imagenet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet50_imagenet.eval()
    resnet50_imagenet.to(device)
    
    clip_model, _ = clip.load("RN50", device=device)
    clip_model.eval()
    
    # Define 10 classes to test
    # These indices correspond to specific ImageNet classes
    # Update based on your label file structure
    labels_ten = [
        'bald eagle', 'axolotl', 'common iguana', 'Indian cobra', 'peacock',
        'scorpion', 'centipede', 'black swan', 'French bulldog', 'leopard'
    ]
    
    print("\n" + "="*80)
    print("RESNET-50 vs CLIP COMPARISON")
    print("="*80)
    print("\nTesting on 10 ImageNet classes:")
    for i, label in enumerate(labels_ten):
        print(f"  {i+1}. {label}")
    
    print("\n" + "="*80)
    print("NOTE: Update image_paths in the code with your actual data paths")
    print("Expected structure: [sketch_image, art_image, original_image]")
    print("="*80)
    
    # Example structure (commented out - update with your paths):
    # classes_to_test = {
    #     'bald eagle': [
    #         '/path/to/sketch/n01614925/sketch_2.JPEG',
    #         '/path/to/art/art_6.jpg',
    #         '/path/to/original/n01614925/n01614925_10607.JPEG'
    #     ],
    #     # Add more classes...
    # }
    # 
    # for class_name, image_paths in classes_to_test.items():
    #     compare_class_performance(
    #         class_name, image_paths, resnet50_imagenet, 
    #         clip_model, text_labels, labels_ten, device
    #     )


if __name__ == "__main__":
    main()

