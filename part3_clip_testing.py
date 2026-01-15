"""
Part 3: Basic CLIP Testing

This script tests the CLIP model on sample images from ImageNet.
It demonstrates how CLIP performs zero-shot classification by:
1. Encoding text labels into feature vectors
2. Encoding images into feature vectors
3. Computing cosine similarity between image and text features
4. Using softmax to get probability distribution over classes

Key Concepts:
- Zero-shot learning: Classifying images without training on specific classes
- Contrastive learning: Learning by comparing image-text pairs
- Cosine similarity: Measuring alignment between feature vectors
"""

import torch
import torch.nn.functional as F
import clip
from utils import get_device, load_imagenet_labels, load_image_cv
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv


def encode_text_labels(clip_model, text_labels, device, num_labels=10):
    """
    Encode text labels into feature vectors using CLIP's text encoder.
    
    Args:
        clip_model: CLIP model
        text_labels: List of text labels
        device: Computation device
        num_labels: Number of labels to encode (default: first 10)
        
    Returns:
        text_features: Tensor of encoded text features
    """
    text_features = []
    
    for i in range(num_labels):
        category = text_labels[i]
        # Create text prompt: "a photo of a {category}"
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {category}")])
        text_inputs = text_inputs.to(device)
        text_feature = clip_model.encode_text(text_inputs)
        text_features.append(text_feature)
    
    text_features = torch.cat(text_features)
    return text_features


def encode_images(clip_model, image_paths, device, preprocess=None):
    """
    Encode images into feature vectors using CLIP's image encoder.
    
    Args:
        clip_model: CLIP model
        image_paths: List of image file paths
        device: Computation device
        preprocess: CLIP preprocessing function (will load if None)
        
    Returns:
        image_features: Tensor of encoded image features
        transformed_images: List of preprocessed image tensors
    """
    # Get CLIP's preprocessing function if not provided
    if preprocess is None:
        _, preprocess = clip.load("RN50", device=device)
    
    transformed_images = []
    
    # Visualize images
    plt.figure(figsize=(10, 5))
    for i, image_path in enumerate(image_paths):
        # Load and display image
        image = load_image_cv(image_path)
        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(image)
        plt.title(f"Image {i+1}")
        plt.axis('off')
        
        # Preprocess for CLIP
        image_tensor = preprocess(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        transformed_images.append(image_tensor)
    
    plt.tight_layout()
    plt.show()
    
    # Encode all images
    images_tensor = torch.cat(transformed_images)
    image_features = clip_model.encode_image(images_tensor)
    
    return image_features, transformed_images


def compute_predictions(image_features, text_features, text_labels):
    """
    Compute predictions by comparing image and text features.
    
    Args:
        image_features: Encoded image features
        text_features: Encoded text features
        text_labels: List of text labels
        
    Returns:
        predictions: List of predicted category names
        probabilities: Probability distributions for each image
    """
    # Compute cosine similarity between image and text features
    # image_features: [num_images, feature_dim]
    # text_features: [num_labels, feature_dim]
    # Result: [num_images, num_labels]
    cosine_similarities = F.cosine_similarity(
        image_features.unsqueeze(1), 
        text_features.unsqueeze(0), 
        dim=2
    )
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(cosine_similarities, dim=1)
    
    # Get predictions
    predictions = []
    for i, prob in enumerate(probabilities):
        max_index = prob.argmax().item()
        predicted_category = text_labels[max_index]
        predictions.append(predicted_category)
        print(f"Image {i + 1} is predicted to be in category: {predicted_category}")
    
    return predictions, probabilities


def main(image_paths, categories_path, device=None):
    """
    Main function to test CLIP on sample images.
    
    Args:
        image_paths: List of image file paths to test
        categories_path: Path to ImageNet labels file
        device: Computation device (auto-detected if None)
    """
    if device is None:
        device = get_device()
    
    print(f"Using device: {device}")
    print(f"Testing CLIP on {len(image_paths)} images\n")
    
    # Load CLIP model and preprocessing function
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load("RN50", device=device)
    clip_model.eval()
    
    # Load ImageNet labels
    print("Loading ImageNet labels...")
    text_labels = load_imagenet_labels(categories_path)
    print(f"Loaded {len(text_labels)} ImageNet categories")
    
    # Encode text labels (first 10 for testing)
    print("\nEncoding text labels...")
    text_features = encode_text_labels(clip_model, text_labels, device, num_labels=10)
    print(f"Text features shape: {text_features.shape}")
    
    # Encode images
    print("\nEncoding images...")
    image_features, _ = encode_images(clip_model, image_paths, device, preprocess=preprocess)
    print(f"Image features shape: {image_features.shape}")
    
    # Compute predictions
    print("\nComputing predictions...")
    print("="*60)
    predictions, probabilities = compute_predictions(image_features, text_features, text_labels[:10])
    print("="*60)
    
    return predictions, probabilities, clip_model


if __name__ == "__main__":
    # Example usage - update these paths with your data
    # image_paths = [
    #     "/path/to/image1.jpg",
    #     "/path/to/image2.jpg"
    # ]
    # categories_path = "/path/to/imagenet1000_clsidx_to_labels.txt"
    # 
    # predictions, probabilities, clip_model = main(image_paths, categories_path)
    
    print("Part 3: CLIP Testing")
    print("="*60)
    print("To use this script, provide:")
    print("1. List of image paths to test")
    print("2. Path to ImageNet labels file")
    print("\nExample:")
    print("  image_paths = ['/path/to/image1.jpg', '/path/to/image2.jpg']")
    print("  categories_path = '/path/to/imagenet1000_clsidx_to_labels.txt'")
    print("  main(image_paths, categories_path)")

