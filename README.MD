# Image Classification: CLIP vs ResNet-50 Comparison

## Overview

This project compares two state-of-the-art image classification approaches:
1. **ResNet-50** - A traditional CNN pretrained on ImageNet
2. **CLIP (Contrastive Language-Image Pre-training)** - A vision-language model that learns from image-text pairs

The comparison explores:
- Model architectures and parameter counts
- Zero-shot learning capabilities
- Performance on distribution shifts (sketches, artistic images)
- Computational efficiency (FP16 vs FP32 precision)

**Reference Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (CLIP)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Part 1: Model Loading](#part-1-model-loading)
4. [Part 2: ImageNet Synsets](#part-2-imagenet-synsets)
5. [Part 3: CLIP Testing](#part-3-clip-testing)
6. [Part 4: Model Comparison](#part-4-model-comparison)
7. [Part 5: FP16 vs FP32](#part-5-fp16-vs-fp32)
8. [Setup and Installation](#setup-and-installation)
9. [Data Requirements](#data-requirements)
10. [Usage](#usage)

---

## Introduction

### What is CLIP?

**CLIP (Contrastive Language-Image Pre-training)** is a neural network trained on a massive dataset of image-text pairs from the internet. Unlike traditional image classification models that learn to predict fixed categories, CLIP learns to understand the relationship between images and their textual descriptions.

**Key Innovation**: CLIP uses **contrastive learning** to align image and text representations in a shared embedding space. This allows it to:
- Perform **zero-shot classification** (classify images without training on specific classes)
- Generalize to new tasks and domains
- Understand semantic relationships between concepts

### What is ResNet-50?

**ResNet-50** is a deep convolutional neural network with 50 layers, introduced in 2015. It uses residual connections (skip connections) to enable training of very deep networks. The version used here is pretrained on ImageNet, a dataset of 1.2 million images across 1000 categories.

**Key Characteristics**:
- Supervised learning (requires labeled training data)
- Fixed output classes (1000 ImageNet categories)
- Excellent performance on ImageNet test set
- May struggle with distribution shifts

### Why Compare Them?

This comparison demonstrates:
1. **Different learning paradigms**: Supervised (ResNet) vs. Self-supervised (CLIP)
2. **Generalization**: How well models handle distribution shifts
3. **Flexibility**: Zero-shot vs. fixed-class classification
4. **Efficiency**: Trade-offs between precision and performance

---

## Project Structure

```
image_classification/
├── part1_model_loading.py      # Load and compare models
├── part2_imagenet_synsets.py   # ImageNet structure explanation
├── part3_clip_testing.py        # Basic CLIP zero-shot testing
├── part4_model_comparison.py    # ResNet vs CLIP comparison
├── part5_fp16_comparison.py     # FP16 vs FP32 efficiency
├── utils.py                     # Common utility functions
├── requirements.txt             # Python dependencies
├── README.MD                    # This file
└── assignment4-cv.ipynb         # Original notebook (reference)
```

---

## Part 1: Model Loading

**File**: `part1_model_loading.py`

### What This Part Does

Loads both ResNet-50 (ImageNet pretrained) and CLIP models, then compares their architectures and parameter counts.

### Theory

#### ResNet-50 Architecture

ResNet-50 consists of:
- **Initial Convolution**: 7×7 conv, stride 2
- **Max Pooling**: 3×3, stride 2
- **4 Residual Blocks**: Each with multiple bottleneck layers
- **Average Pooling**: Global average pooling
- **Fully Connected Layer**: 1000 outputs (ImageNet classes)

**Total Parameters**: ~25.5 million

#### CLIP Architecture

CLIP uses a dual-encoder architecture:
- **Visual Encoder**: ResNet-50 based (but modified)
- **Text Encoder**: Transformer-based
- **Joint Training**: Both encoders trained together

**Key Differences from Standard ResNet-50**:
- Modified input/output layers for CLIP compatibility
- Trained jointly with text encoder (not just images)
- Optimized for contrastive learning objective
- More parameters (~38.3 million) due to architectural modifications

### Why Different Parameter Counts?

1. **Architectural Modifications**: CLIP's ResNet-50 has different layer configurations
2. **Training Objective**: Optimized for image-text alignment, not just classification
3. **Feature Dimensions**: Different embedding dimensions for contrastive learning

### Code Explanation

```python
# Load ResNet-50 with ImageNet weights
resnet50_imagenet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Load CLIP model (RN50 variant)
clip_model, _ = clip.load("RN50")
clip_visual_encoder = clip_model.visual  # Extract visual encoder

# Count parameters
clip_params = sum(p.numel() for p in clip_visual_encoder.parameters())
r50_params = sum(p.numel() for p in resnet50_imagenet.parameters())
```

### Usage

```python
python part1_model_loading.py
```


---

## Part 2: ImageNet Synsets

**File**: `part2_imagenet_synsets.py`

### What This Part Does

Explains the ImageNet dataset structure and the concept of synsets (synonym sets).

### Theory

#### WordNet Hierarchy

ImageNet organizes its 1000 classes using the **WordNet** hierarchy:

- **Synset**: A set of synonyms representing a concept
- **Hypernyms**: Broader categories (e.g., "animal" is a hypernym of "dog")
- **Hyponyms**: More specific categories (e.g., "poodle" is a hyponym of "dog")

**Example Hierarchy**:
```
entity
  └── physical entity
      └── object
          └── living thing
              └── organism
                  └── animal
                      └── chordate
                          └── vertebrate
                              └── mammal
                                  └── carnivore
                                      └── dog
                                          └── poodle
```

#### Challenges with Synset-Based Classification

1. **Intra-class Variability**:
   - Objects in the same synset can look very different
   - Example: "Car" includes sedans, trucks, sports cars, etc.
   - Models must learn robust features invariant to these variations

2. **Inter-class Similarity**:
   - Different synsets can be visually similar
   - Example: Different dog breeds share many characteristics
   - Can lead to confusion between classes

3. **Visual Variations**:
   - Same object in different orientations
   - Varying lighting conditions
   - Different camera perspectives
   - Background variations
   - Scale differences

### Why This Matters

Understanding synsets helps explain:
- Why classification is challenging
- Why models need to learn robust features
- How CLIP's semantic understanding helps with generalization

### Usage

```python
python part2_imagenet_synsets.py
```

This will print an explanation of ImageNet synsets.

---

## Part 3: CLIP Testing

**File**: `part3_clip_testing.py`

### What This Part Does

Demonstrates CLIP's zero-shot classification capabilities by:
1. Encoding text labels into feature vectors
2. Encoding images into feature vectors
3. Computing similarity between image and text features
4. Predicting the most likely class

### Theory

#### Zero-Shot Learning

**Zero-shot learning** is the ability to recognize objects/classes that were not seen during training. CLIP achieves this through:

1. **Joint Embedding Space**: Images and text are mapped to the same feature space
2. **Semantic Understanding**: The model learns semantic relationships, not just visual patterns
3. **Flexible Classification**: Any text description can be used as a "class"

#### Contrastive Learning

CLIP is trained using **contrastive learning**:

**Training Process**:
1. Sample image-text pairs from the internet
2. Encode images and text separately
3. Maximize similarity for matching pairs
4. Minimize similarity for non-matching pairs

**Mathematical Formulation**:
- For a batch of N image-text pairs:
  - Create similarity matrix: `S[i,j] = similarity(image_i, text_j)`
  - Positive pairs: `(image_i, text_i)` (diagonal)
  - Negative pairs: All other combinations
  - Loss: Contrastive loss that pulls positive pairs together, pushes negatives apart

#### Cosine Similarity

After encoding, CLIP uses **cosine similarity** to measure alignment:

```
similarity = (image_features · text_features) / (||image_features|| × ||text_features||)
```

This measures the angle between vectors in the embedding space, not their magnitude.

#### Softmax for Probabilities

The similarities are converted to probabilities using softmax:

```
P(class_i | image) = exp(similarity_i) / Σ exp(similarity_j)
```

This gives a probability distribution over all possible classes.

### Code Explanation

```python
# 1. Encode text labels
text_inputs = clip.tokenize(f"a photo of a {category}")
text_features = clip_model.encode_text(text_inputs)

# 2. Encode images
image_tensor = preprocess(image).unsqueeze(0)
image_features = clip_model.encode_image(image_tensor)

# 3. Compute cosine similarity
cosine_similarities = F.cosine_similarity(
    image_features.unsqueeze(1),  # [num_images, 1, feature_dim]
    text_features.unsqueeze(0),    # [1, num_labels, feature_dim]
    dim=2                          # Compare along feature dimension
)

# 4. Convert to probabilities
probabilities = F.softmax(cosine_similarities, dim=1)

# 5. Get prediction
predicted_class = probabilities.argmax(dim=1)
```

### Why "a photo of a {category}"?

The prompt template matters! CLIP was trained on image-text pairs, and many descriptions follow this pattern. Using consistent prompts improves performance.

### Usage

```python
from part3_clip_testing import main

# Provide image paths and labels file
image_paths = [
    "/path/to/image1.jpg",
    "/path/to/image2.jpg"
]
categories_path = "/path/to/imagenet1000_clsidx_to_labels.txt"

predictions, probabilities, clip_model = main(image_paths, categories_path)
```

---

## Part 4: Model Comparison

**File**: `part4_model_comparison.py`

### What This Part Does

Compares ResNet-50 and CLIP performance on:
- **Original ImageNet images**: Standard photos
- **ImageNet Sketch**: Hand-drawn sketches
- **Artistic images**: Stylized/artistic versions
- **Other distribution shifts**: Images from different domains

### Theory

#### Distribution Shift

**Distribution shift** occurs when test data differs from training data. Types include:
- **Covariate shift**: Different input distributions (e.g., sketches vs. photos)
- **Label shift**: Different class distributions
- **Concept drift**: Relationship between inputs and outputs changes

#### Why CLIP is More Robust

1. **Diverse Training Data**: Trained on internet images (not just ImageNet)
2. **Semantic Understanding**: Learns concepts, not just visual patterns
3. **Text Guidance**: Text descriptions provide semantic context
4. **Contrastive Learning**: More robust feature representations

#### Why ResNet Struggles with Distribution Shifts

1. **Narrow Training**: Only trained on ImageNet photos
2. **Visual Patterns**: Learns specific visual features (textures, colors)
3. **No Semantic Understanding**: Doesn't understand "what" objects are conceptually
4. **Fixed Classes**: Can only predict 1000 predefined categories

### Key Observations

From the experiments:

1. **Original ImageNet Images**:
   - ResNet-50: Slightly better (trained specifically for this)
   - CLIP: Very close performance (zero-shot!)

2. **ImageNet Sketch**:
   - ResNet-50: Poor performance (confused by sketch style)
   - CLIP: Much better (understands semantic content)

3. **Artistic Images**:
   - ResNet-50: Struggles (different visual style)
   - CLIP: Better (semantic understanding helps)

### Code Structure

The comparison tests 10 classes:
- Bald eagle, Axolotl, Common iguana, Indian cobra, Peacock
- Scorpion, Centipede, Black swan, French bulldog, Leopard

For each class, three image types are tested:
1. Sketch version
2. Artistic/stylized version
3. Original ImageNet photo

### Usage

```python
from part4_model_comparison import compare_class_performance

# Define image paths for a class
image_paths = [
    "/path/to/sketch_image.jpg",    # Sketch version
    "/path/to/art_image.jpg",       # Artistic version
    "/path/to/original_image.jpg"   # Original ImageNet image
]

# Compare performance
compare_class_performance(
    class_name="bald eagle",
    image_paths=image_paths,
    resnet_model=resnet50_imagenet,
    clip_model=clip_model,
    text_labels=text_labels,
    labels_ten=labels_ten,
    device=device
)
```

### Output

For each class, you'll see:
- ResNet-50 top-5 predictions with probabilities
- CLIP top-5 predictions with similarities
- Comparison showing CLIP's robustness to distribution shifts

---

## Part 5: FP16 vs FP32

**File**: `part5_fp16_comparison.py`

### What This Part Does

Compares model performance and efficiency using:
- **FP32 (32-bit floating point)**: Standard precision
- **FP16 (16-bit floating point)**: Half precision

### Theory

#### Floating Point Precision

**FP32 (Single Precision)**:
- 32 bits per number
- ~7 decimal digits of precision
- Standard for most deep learning

**FP16 (Half Precision)**:
- 16 bits per number
- ~3 decimal digits of precision
- 2× less memory
- Faster on modern GPUs (Tensor Cores)

#### Why FP16 Works

1. **Neural Network Robustness**: Models are often robust to small numerical errors
2. **Gradient Scaling**: Training can use gradient scaling to prevent underflow
3. **Mixed Precision**: Often used in training (FP16 forward, FP32 backward)

#### Benefits of FP16

1. **Memory Reduction**: ~50% less GPU memory
2. **Speed**: Faster on GPUs with Tensor Cores (NVIDIA V100, A100, etc.)
3. **Batch Size**: Can use larger batches with same memory
4. **Throughput**: More images processed per second

#### Trade-offs

1. **Accuracy**: Slight reduction in numerical precision
2. **Underflow Risk**: Very small numbers may become zero
3. **Overflow Risk**: Very large numbers may become infinity
4. **Compatibility**: Not all operations support FP16

### Code Explanation

```python
# Convert model to FP16
clip_visual_fp16 = clip_visual_fp32.half()

# Convert input to FP16
image_tensor_fp16 = image_tensor.half()

# Inference (same as FP32)
output_fp16 = clip_visual_fp16(image_tensor_fp16)

# Convert back to FP32 for comparison
output_fp16 = output_fp16.float()
```

### Benchmarking

The script measures:
1. **Inference Time**: Average and standard deviation over 100 runs
2. **Memory Usage**: Peak GPU memory allocation
3. **Accuracy**: Top-5 predictions (should match FP32)

### Usage

```python
from part5_fp16_comparison import main

# Test on a single image
image_path = "/path/to/test_image.jpg"
labels_ten = ['bald eagle', 'axolotl', ...]  # 10 labels to test

results = main(image_path, labels_ten=labels_ten)
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for best performance)
- 8GB+ GPU memory (for FP32, less for FP16)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Harinie-Sethu/image_classification.git
cd image_classification
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and torchvision
- CLIP (from OpenAI's GitHub)
- OpenCV, PIL, matplotlib, numpy

3. **Verify installation**:
```python
import torch
import clip
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Data Requirements

### ImageNet Labels File

You need the ImageNet class labels file:
- **File**: `imagenet1000_clsidx_to_labels.txt`
- **Format**: Each line contains `{index}: '{label1', 'label2', ...}`
- **Source**: Can be downloaded from ImageNet website or various repositories

### Image Data

For full functionality, you'll need:

1. **ImageNet Images**: Original ImageNet validation/test images
   - Can use ImageNet-mini for testing
   - Or download specific classes from ImageNet

2. **ImageNet Sketch**: Sketch versions of ImageNet images
   - Available from ImageNet-Sketch dataset
   - Or create your own sketches

3. **Artistic Images**: Stylized versions
   - Can use artistic filters
   - Or download from art datasets

### Data Organization

Organize your data as:
```
data/
├── imagenet/
│   ├── train/
│   │   ├── n01440764/  # Class folders
│   │   └── ...
│   └── val/
├── imagenet_sketch/
│   └── sketch/
│       ├── n01440764/
│       └── ...
└── artistic/
    └── ...
```

### Downloading Data

**Note**: If downloading data will take a long time, you can:
1. Start with small subsets for testing
2. Use the code structure as-is (it's ready to run once data is available)
3. Test with your own images (modify paths accordingly)

---

## Usage

### Running Individual Parts

**Part 1: Model Loading**
```bash
python part1_model_loading.py
```

**Part 2: ImageNet Synsets (Documentation)**
```bash
python part2_imagenet_synsets.py
```

**Part 3: CLIP Testing**
```python
from part3_clip_testing import main

image_paths = ["/path/to/image1.jpg", "/path/to/image2.jpg"]
categories_path = "/path/to/imagenet1000_clsidx_to_labels.txt"
main(image_paths, categories_path)
```

**Part 4: Model Comparison**
```python
from part4_model_comparison import main
# Update image paths in the script
main()
```

**Part 5: FP16 Comparison**
```python
from part5_fp16_comparison import main

image_path = "/path/to/test_image.jpg"
labels_ten = ['bald eagle', 'axolotl', 'common iguana', 'Indian cobra', 
              'peacock', 'scorpion', 'centipede', 'black swan', 
              'French bulldog', 'leopard']
main(image_path, labels_ten=labels_ten)
```

### Running All Parts Sequentially

Create a main script that runs all parts:

```python
# run_all.py
from part1_model_loading import main as part1
from part2_imagenet_synsets import main as part2
# ... etc

if __name__ == "__main__":
    print("Running Part 1...")
    resnet, clip_model, clip_visual = part1()
    
    print("\nRunning Part 2...")
    part2()
    
    # Continue with other parts...
```

---

## Key Concepts Explained

### Zero-Shot Learning

**Zero-shot learning** allows a model to recognize classes it hasn't seen during training. CLIP achieves this by:
- Learning a shared image-text embedding space
- Understanding semantic relationships
- Using text descriptions as flexible "classes"

**Example**: If CLIP understands "dog" and "cat", it can likely understand "puppy" and "kitten" without explicit training.

### Contrastive Learning

**Contrastive learning** trains models by comparing similar and dissimilar pairs:
- **Positive pairs**: Matching image-text pairs (should be similar)
- **Negative pairs**: Non-matching pairs (should be different)
- **Objective**: Maximize similarity for positives, minimize for negatives

This creates rich, generalizable representations.

### Distribution Shift Robustness

**Distribution shift** occurs when test data differs from training data. CLIP is more robust because:
- Trained on diverse internet data (not just ImageNet)
- Learns semantic concepts (not just visual patterns)
- Text guidance provides context

### Mixed Precision Training

**Mixed precision** uses FP16 for forward pass, FP32 for backward pass:
- **Forward**: FP16 (faster, less memory)
- **Backward**: FP32 (more accurate gradients)
- **Gradient Scaling**: Prevents underflow in FP16

---

## Results Summary

### Model Comparison Findings

1. **Parameter Counts**:
   - CLIP: 38.3M parameters
   - ResNet-50: 25.5M parameters
   - Difference: CLIP has more parameters due to architectural modifications

2. **Performance on Original Images**:
   - ResNet-50: Slightly better (trained specifically for ImageNet)
   - CLIP: Very close (remarkable for zero-shot)

3. **Performance on Distribution Shifts**:
   - **Sketches**: CLIP significantly better
   - **Artistic Images**: CLIP better
   - **Other Shifts**: CLIP more robust

4. **FP16 Efficiency**:
   - **Speed**: ~2% faster, more stable
   - **Memory**: ~50% reduction
   - **Accuracy**: Top-5 rankings match, slight probability differences

### Key Takeaways

1. **CLIP's Zero-Shot Ability**: Can classify images without training on specific classes
2. **Robustness**: Better generalization to distribution shifts
3. **Flexibility**: Can use any text description as a class
4. **Efficiency**: FP16 provides good trade-off between speed/memory and accuracy

---


## References

1. **CLIP Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
2. **ResNet Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
3. **ImageNet Dataset**: [ImageNet](https://www.image-net.org/)
4. **CLIP GitHub**: [OpenAI CLIP](https://github.com/openai/CLIP)
