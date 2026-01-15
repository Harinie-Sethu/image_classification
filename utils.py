"""
Utility functions for image classification with CLIP and ResNet.
Common functions for image loading, preprocessing, and visualization.
"""

import torch
import torchvision.transforms as transforms
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def get_device():
    """Get the computation device (CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_imagenet_labels(categories_path):
    """
    Load ImageNet category labels from a text file.
    
    Args:
        categories_path: Path to the ImageNet labels file
        
    Returns:
        text_labels: List of category names (first word of each label)
    """
    with open(categories_path, 'r') as file:
        imagenet_categories = [line.strip() for line in file]

    text_labels = []
    for text in imagenet_categories:
        text = text.strip().strip('{}').strip(",")
        key, value = text.split(': ')
        value = value.strip().strip("'")
        first_word = value.split(',')[0]
        text_labels.append(first_word)
    
    return text_labels


def load_image_cv(image_path):
    """
    Load an image using OpenCV and convert to RGB.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        image: PIL Image in RGB format
    """
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


def load_image_pil(image_path):
    """
    Load an image using PIL.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        image: PIL Image
    """
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def get_resnet_transform():
    """
    Get the standard ImageNet preprocessing transform for ResNet.
    
    Returns:
        transform: torchvision transform pipeline
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def visualize_images(image_paths, titles=None, rows=1, cols=None):
    """
    Visualize multiple images in a grid.
    
    Args:
        image_paths: List of image paths
        titles: List of titles for each image (optional)
        rows: Number of rows in the grid
        cols: Number of columns (auto-calculated if None)
    """
    if cols is None:
        cols = len(image_paths)
    
    plt.figure(figsize=(cols * 3, rows * 3))
    
    for i, image_path in enumerate(image_paths):
        plt.subplot(rows, cols, i + 1)
        
        # Try loading with PIL first, then OpenCV
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except:
            image = load_image_cv(image_path)
        
        plt.imshow(image)
        if titles:
            plt.title(titles[i] if i < len(titles) else f"Image {i+1}")
        else:
            plt.title(f"Image {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


