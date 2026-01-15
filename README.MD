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

### Expected Output

```
Using device: cuda

Loading models on device: cuda
Loading ResNet-50 (ImageNet pretrained)...
Loading CLIP model (RN50)...

============================================================
MODEL PARAMETER COMPARISON
============================================================
CLIP Visual Encoder (RN50): 38,316,896 parameters
ResNet-50 (ImageNet):       25,557,032 parameters
Difference:                 12,759,864 parameters
============================================================
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

### Expected Output

```
Using device: cuda
Testing CLIP on 2 images

Loading CLIP model...
Loading ImageNet labels...
Loaded 1000 ImageNet categories

Encoding text labels...
Text features shape: torch.Size([10, 512])

Encoding images...
Image features shape: torch.Size([2, 512])

Computing predictions...
============================================================
Image 1 is predicted to be in category: tench
Image 2 is predicted to be in category: stingray
============================================================
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

### Expected Output

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

### Expected Output

```
============================================================
INFERENCE TIME COMPARISON
============================================================
FP32 mean inference time: 0.0087 s, std dev: 0.0008 s
FP16 mean inference time: 0.0085 s, std dev: 0.0003 s
Speedup: 1.02x
============================================================

============================================================
PREDICTION COMPARISON
============================================================
FP32 Top-5 Predictions:
  Class: scorpion, Probability: 0.1110
  Class: centipede, Probability: 0.1085
  ...

FP16 Top-5 Predictions:
  Class: scorpion, Probability: 0.1107
  Class: centipede, Probability: 0.1082
  ...

Top-5 rankings match: True
============================================================

============================================================
MEMORY USAGE COMPARISON
============================================================
FP32 peak memory: 245.32 MB
FP16 peak memory: 122.66 MB
Memory reduction: 50.0%
============================================================
```

### Key Observations

1. **Speed**: FP16 is slightly faster with lower variance (more stable)
2. **Memory**: ~50% reduction (allows larger batches)
3. **Accuracy**: Top-5 rankings typically match, probabilities may differ slightly
4. **Trade-off**: Good balance between efficiency and accuracy

---

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

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Use FP16 model
   - Reduce batch size
   - Use smaller images

2. **CLIP Installation Issues**:
   - Ensure you have git installed
   - Try: `pip install git+https://github.com/openai/CLIP.git --upgrade`

3. **Image Loading Errors**:
   - Check image paths are correct
   - Ensure images are valid (not corrupted)
   - Try different image loading methods (PIL vs OpenCV)

4. **Label File Format**:
   - Ensure labels file matches expected format
   - Check file encoding (should be UTF-8)

---

## References

1. **CLIP Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
2. **ResNet Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
3. **ImageNet Dataset**: [ImageNet](https://www.image-net.org/)
4. **CLIP GitHub**: [OpenAI CLIP](https://github.com/openai/CLIP)

---

## License

This implementation is for educational purposes as part of a computer vision course assignment.

---

---

## Interview Preparation Guide

This section provides detailed explanations of every component, written for interview preparation. It explains the code, models, datasets, and design choices as if you're explaining to someone new to computer vision.

### Complete Code Walkthrough

#### Part 1: Model Loading - Detailed Explanation

**What Models Are We Using?**

1. **ResNet-50 (ImageNet Pretrained)**
   - **What it is**: A deep convolutional neural network with 50 layers
   - **Why ResNet-50?**: It's a standard, well-established architecture that's been proven effective. The "50" means 50 layers deep.
   - **What is ImageNet?**: A massive dataset with 1.2 million images across 1000 categories (dogs, cats, cars, etc.)
   - **Why pretrained?**: Training from scratch takes days/weeks. Using pretrained weights means the model already learned to recognize common objects, so we can use it immediately.
   - **What does "pretrained on ImageNet" mean?**: The model was trained to classify images into 1000 ImageNet categories. It learned features like edges, textures, shapes, and object parts.

2. **CLIP (RN50 variant)**
   - **What it is**: A vision-language model that understands both images and text
   - **Why CLIP?**: It can classify images using text descriptions (zero-shot learning) - you don't need to train it on specific classes
   - **What does "RN50" mean?**: "ResNet-50" - CLIP uses ResNet-50 as its visual encoder (the part that processes images)
   - **Why different from standard ResNet?**: CLIP's ResNet is modified to work with text. It's trained to match images with text descriptions, not just classify into fixed categories.

**Code Breakdown (Line by Line):**

```python
# Line 36: Load ResNet-50 with ImageNet weights
resnet50_imagenet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
```
- `models.resnet50()`: Creates ResNet-50 architecture
- `weights=IMAGENET1K_V1`: Downloads pretrained weights from ImageNet
- This gives us a model that can classify 1000 ImageNet classes

```python
# Line 37: Set to evaluation mode
resnet50_imagenet.eval()
```
- `.eval()`: Turns off training features (like dropout, batch normalization updates)
- Important for inference - we're not training, just using the model

```python
# Line 38: Move to GPU
resnet50_imagenet.to(device)
```
- Moves model to GPU (if available) for faster computation
- GPUs are much faster than CPUs for neural networks

```python
# Line 42: Load CLIP model
clip_model, _ = clip.load("RN50")
```
- `clip.load("RN50")`: Loads CLIP with ResNet-50 visual encoder
- Returns: (model, preprocess_function)
- We use `_` to ignore the preprocess function (we'll get it separately)

```python
# Line 45: Extract visual encoder
clip_visual_encoder = clip_model.visual
```
- CLIP has two parts: visual encoder (processes images) and text encoder (processes text)
- We extract just the visual part to compare with ResNet-50

**Why Compare Parameter Counts?**

- **Parameters**: The numbers the model learns (weights and biases)
- **More parameters**: Usually means more capacity to learn, but also more memory/computation
- **Different counts**: Shows the architectures are different, even though both use ResNet-50 as base
- **Interview question**: "Why does CLIP have more parameters?" 
  - Answer: "CLIP's architecture is modified for joint image-text learning. It has different layer configurations optimized for contrastive learning, which requires more parameters to align images and text in a shared space."

#### Part 2: ImageNet Synsets - Understanding the Dataset

**What is ImageNet?**

- **Dataset**: 1.2 million images, 1000 categories
- **Purpose**: Standard benchmark for image classification
- **Organization**: Uses WordNet hierarchy (a database of word relationships)

**What are Synsets?**

- **Synset = Synonym Set**: A group of words representing the same concept
- **Example**: "dog", "canine", "puppy" might be in the same synset
- **In ImageNet**: Each synset is a category (like "n01440764" = "tench" fish)

**Why This Organization Matters:**

1. **Hierarchical Structure**:
   - Broad categories → Specific categories
   - Example: Animal → Mammal → Dog → Poodle
   - Helps models understand relationships between classes

2. **Challenges for Models**:
   - **Intra-class variability**: Same class, very different looks (e.g., different dog breeds)
   - **Inter-class similarity**: Different classes that look similar (e.g., different bird species)
   - **Visual variations**: Same object in different poses, lighting, backgrounds

**Interview Question**: "What challenges does ImageNet present for classification?"
- **Answer**: "ImageNet has high intra-class variability (objects in same class look different) and inter-class similarity (different classes look similar). Models must learn robust features that capture semantic meaning, not just visual patterns. This is why CLIP's semantic understanding helps - it learns 'what' objects are conceptually, not just how they look."

#### Part 3: CLIP Testing - Zero-Shot Classification

**What is Zero-Shot Learning?**

- **Traditional approach**: Train model on specific classes (e.g., train on 1000 ImageNet classes)
- **Zero-shot**: Classify images into classes the model was never trained on
- **How CLIP does it**: Uses text descriptions as "classes" - if it understands the text, it can classify

**Step-by-Step Process:**

1. **Encode Text Labels**:
```python
text_inputs = clip.tokenize(f"a photo of a {category}")
text_features = clip_model.encode_text(text_inputs)
```
- **What happens**: Text is converted to tokens, then encoded into a feature vector
- **Why "a photo of a {category}"**: This prompt template matches how CLIP was trained. Using consistent prompts improves accuracy.
- **Output**: A vector representing the text in CLIP's embedding space

2. **Encode Images**:
```python
image_tensor = preprocess(image).unsqueeze(0)
image_features = clip_model.encode_image(image_tensor)
```
- **Preprocess**: Resizes, normalizes image to match CLIP's expected input
- **Encode**: Converts image to a feature vector in the same embedding space as text
- **Output**: A vector representing the image

3. **Compute Similarity**:
```python
cosine_similarities = F.cosine_similarity(
    image_features.unsqueeze(1),  # [2, 1, 512] - 2 images
    text_features.unsqueeze(0),    # [1, 10, 512] - 10 text labels
    dim=2
)
```
- **What is cosine similarity?**: Measures the angle between two vectors (not their length)
- **Why cosine?**: We care about direction (semantic meaning), not magnitude
- **Result**: A matrix showing how similar each image is to each text label

4. **Convert to Probabilities**:
```python
probabilities = F.softmax(cosine_similarities, dim=1)
```
- **Softmax**: Converts similarities to probabilities (sums to 1.0)
- **Why softmax?**: Gives us a probability distribution over classes
- **Result**: For each image, probabilities for each class

**Interview Questions and Answers:**

**Q: How does CLIP perform zero-shot classification?**
- **A**: "CLIP maps both images and text into a shared embedding space. During training, it learns to align matching image-text pairs close together. At inference, we encode text descriptions of classes and compare them with encoded images using cosine similarity. The closest text match is the predicted class. This works because CLIP learned semantic relationships, not just visual patterns."

**Q: Why use cosine similarity instead of Euclidean distance?**
- **A**: "Cosine similarity measures the angle between vectors, which captures semantic similarity regardless of vector magnitude. Euclidean distance would be affected by the length of feature vectors. In CLIP's embedding space, similar concepts point in similar directions, so cosine similarity is more appropriate for measuring semantic alignment."

**Q: Why the prompt template 'a photo of a {category}'?**
- **A**: "CLIP was trained on internet image-text pairs, and many descriptions follow this pattern. Using consistent prompts that match the training distribution improves performance. The model learned to associate this phrasing with image content, so using similar prompts at inference time leverages this learned association."

#### Part 4: Model Comparison - Distribution Shifts

**What is Distribution Shift?**

- **Definition**: When test data differs from training data
- **Example**: Model trained on photos, tested on sketches
- **Why it matters**: Real-world data often differs from training data

**Types of Distribution Shifts Tested:**

1. **ImageNet Sketch**: Hand-drawn sketches of objects
   - **Challenge**: Very different visual style (lines, no color/texture)
   - **Why ResNet struggles**: Learned specific visual patterns (textures, colors) that don't exist in sketches
   - **Why CLIP handles it**: Understands semantic content ("this is a bird") regardless of visual style

2. **Artistic Images**: Stylized, artistic versions
   - **Challenge**: Different artistic styles, colors, textures
   - **Why ResNet struggles**: Trained on realistic photos, not art
   - **Why CLIP handles it**: Semantic understanding transcends visual style

3. **Original ImageNet**: Standard photos
   - **ResNet advantage**: Trained specifically for this
   - **CLIP performance**: Very close (remarkable for zero-shot!)

**Code Explanation:**

```python
# ResNet evaluation
resnet_outputs = resnet50_imagenet(image_tensor)
probs = F.softmax(resnet_outputs, dim=1)
top5_indices = torch.topk(resnet_outputs, k=5, dim=1)
```
- **What happens**: Image goes through ResNet, outputs 1000 logits (one per ImageNet class)
- **Softmax**: Converts to probabilities
- **Top-5**: Gets 5 most likely classes

```python
# CLIP evaluation
image_features = clip_model.encode_image(images_tensor)
cosine_similarities = F.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
probabilities = F.softmax(cosine_similarities, dim=1)
```
- **What happens**: Image encoded, compared with 10 text labels (not 1000!)
- **Key difference**: CLIP compares with specific labels we provide, ResNet compares with all 1000 ImageNet classes

**Interview Questions:**

**Q: Why does CLIP perform better on distribution shifts?**
- **A**: "CLIP was trained on diverse internet data, not just ImageNet photos. It learned semantic concepts rather than specific visual patterns. When it sees a sketch of a bird, it recognizes the semantic concept 'bird' even though the visual style is different. ResNet learned specific textures and colors from ImageNet photos, so it struggles when those patterns aren't present."

**Q: Why does ResNet perform slightly better on original ImageNet images?**
- **A**: "ResNet was specifically trained and optimized for ImageNet classification. It learned the exact visual patterns, textures, and features that appear in ImageNet photos. CLIP, while trained on diverse data, wasn't fine-tuned specifically for ImageNet, so it performs slightly worse but still remarkably well for zero-shot learning."

**Q: What does this tell us about model generalization?**
- **A**: "It shows that learning semantic concepts (like CLIP) generalizes better than learning specific visual patterns (like ResNet). CLIP's joint image-text training creates more robust representations that transfer across domains. This is why vision-language models are becoming the standard for many tasks."

#### Part 5: FP16 vs FP32 - Precision and Efficiency

**What is Floating Point Precision?**

- **FP32 (32-bit)**: Standard precision, 32 bits per number
  - Range: Very large and very small numbers
  - Precision: ~7 decimal digits
  - Memory: 4 bytes per number
  
- **FP16 (16-bit)**: Half precision, 16 bits per number
  - Range: Smaller range
  - Precision: ~3 decimal digits
  - Memory: 2 bytes per number (half of FP32!)

**Why Use FP16?**

1. **Memory Savings**: 50% less GPU memory
   - Allows larger batch sizes
   - Can fit larger models
   - Important for deployment on edge devices

2. **Speed**: Faster on modern GPUs
   - GPUs have Tensor Cores optimized for FP16
   - Can process more operations per second
   - Lower latency

3. **Accuracy Trade-off**: Usually minimal loss
   - Neural networks are robust to small numerical errors
   - Top predictions usually match FP32
   - Probabilities may differ slightly

**Code Explanation:**

```python
# Convert model to FP16
clip_visual_fp16 = clip_visual_fp32.half()
```
- `.half()`: Converts all model parameters from FP32 to FP16
- Each weight uses half the memory

```python
# Convert input to FP16
image_tensor_fp16 = image_tensor.half()
```
- Input must also be FP16 to match model precision

```python
# Benchmark inference
for _ in range(100):
    start_time = time.time()
    with torch.no_grad():
        output = model(image_tensor)
    times.append(time.time() - start_time)
```
- **Warmup**: First run is slower (GPU initialization)
- **100 runs**: Get average time (more reliable than single run)
- **torch.no_grad()**: Disables gradient computation (faster, not needed for inference)

**Interview Questions:**

**Q: Why does FP16 work for neural networks when it has less precision?**
- **A**: "Neural networks are robust to small numerical errors. The learned weights and features are distributed in a way that small quantization errors don't significantly affect the output. Additionally, modern training often uses mixed precision (FP16 forward, FP32 backward) with gradient scaling, which helps maintain accuracy while gaining efficiency benefits."

**Q: When would you use FP16 vs FP32?**
- **A**: "Use FP16 when: memory is constrained, deploying to edge devices, or when slight accuracy loss is acceptable for speed gains. Use FP32 when: maximum accuracy is critical, doing research/development, or when memory isn't a concern. In practice, many production systems use FP16 for inference and FP32 for training."

**Q: What are the risks of FP16?**
- **A**: "Underflow (very small numbers become zero) and overflow (very large numbers become infinity) can occur. Gradient values during training can become zero. However, for inference with pretrained models, these risks are minimal because the weights are already stable. Modern frameworks handle these edge cases well."

### Dataset Details

**ImageNet Dataset:**
- **Size**: 1.2 million training images, 50,000 validation images
- **Classes**: 1000 object categories
- **Organization**: WordNet hierarchy (synsets)
- **Format**: Images organized by class folders (e.g., `n01440764/` for "tench")
- **Why ImageNet?**: Standard benchmark, well-established, allows fair comparison

**ImageNet Sketch:**
- **What it is**: Hand-drawn sketches of ImageNet objects
- **Purpose**: Tests robustness to distribution shift
- **Challenge**: Very different visual style from photos

**Artistic Images:**
- **What it is**: Stylized, artistic versions of objects
- **Purpose**: Tests semantic understanding vs. visual pattern matching
- **Challenge**: Different colors, textures, styles

### Model Architecture Details

**ResNet-50 Structure:**
```
Input (224×224×3)
  ↓
Conv1: 7×7, 64 filters, stride 2
  ↓
MaxPool: 3×3, stride 2
  ↓
Residual Block 1: 3 layers, 64 filters
  ↓
Residual Block 2: 4 layers, 128 filters
  ↓
Residual Block 3: 6 layers, 256 filters
  ↓
Residual Block 4: 3 layers, 512 filters
  ↓
Average Pooling
  ↓
Fully Connected: 1000 outputs
```

**Why Residual Connections?**
- **Problem**: Very deep networks are hard to train (vanishing gradients)
- **Solution**: Skip connections allow gradients to flow directly
- **Result**: Can train much deeper networks (50, 101, 152 layers)

**CLIP Architecture:**
```
Image Input → Visual Encoder (ResNet-50 modified) → Image Embedding (512-dim)
Text Input → Text Encoder (Transformer) → Text Embedding (512-dim)
  ↓
Both embeddings in same 512-dimensional space
  ↓
Contrastive Learning: Match image-text pairs
```

**Why 512 Dimensions?**
- Balance between capacity and efficiency
- Large enough to capture rich representations
- Small enough for fast computation
- Both image and text use same dimension for comparison

### Design Choices Explained

**Why ResNet-50 for CLIP's visual encoder?**
- **Proven architecture**: ResNet-50 is well-established and efficient
- **Good balance**: Not too small (lacks capacity) or too large (slow)
- **Modifications**: Adapted for contrastive learning (different output layer)

**Why compare with ImageNet pretrained ResNet?**
- **Fair comparison**: Both models can classify ImageNet classes
- **Standard benchmark**: ImageNet is the standard for image classification
- **Different approaches**: Shows supervised vs. zero-shot learning

**Why test on sketches and art?**
- **Real-world relevance**: Models encounter different visual styles
- **Robustness test**: Shows which model generalizes better
- **Distribution shift**: Important problem in real applications

**Why FP16 comparison?**
- **Practical importance**: Memory and speed matter in deployment
- **Trade-off analysis**: Shows accuracy vs. efficiency
- **Modern practice**: FP16 is commonly used in production

### Common Interview Questions and Answers

**Q: Explain the difference between ResNet and CLIP in simple terms.**
- **A**: "ResNet is like a student who memorized 1000 specific categories from a textbook. CLIP is like a student who learned to understand concepts by reading many different books. ResNet is great at the exact task it was trained on, but CLIP can handle new situations it hasn't seen before because it understands the meaning of things, not just how they look."

**Q: What is the main innovation of CLIP?**
- **A**: "CLIP's main innovation is learning from image-text pairs using contrastive learning. Instead of learning to predict fixed categories, it learns to align images and text in a shared space. This enables zero-shot learning - you can classify images using any text description, without training on specific classes."

**Q: Why does CLIP have more parameters than ResNet-50?**
- **A**: "CLIP's ResNet-50 is modified for joint image-text learning. It has different layer configurations optimized for contrastive learning. The architecture is adapted to work with text encodings, which requires additional parameters. Also, CLIP needs to learn richer representations to align with text semantics."

**Q: Can you explain contrastive learning?**
- **A**: "Contrastive learning teaches the model by comparison. For CLIP, matching image-text pairs should be similar in the embedding space, while non-matching pairs should be different. It's like teaching a child: 'This picture of a dog goes with the word dog' (positive pair) and 'This picture of a dog does NOT go with the word cat' (negative pair). By seeing many such examples, the model learns semantic relationships."

**Q: What are the limitations of CLIP?**
- **A**: "CLIP struggles with: 1) Fine-grained classification (distinguishing very similar classes), 2) Abstract concepts that are hard to describe in text, 3) Tasks requiring spatial reasoning, 4) Very domain-specific tasks where it hasn't seen similar data. Also, it requires text descriptions, so it's not purely visual like ResNet."

**Q: When would you choose ResNet over CLIP?**
- **A**: "Choose ResNet when: 1) You have a fixed set of classes and can train/fine-tune, 2) Maximum accuracy on a specific dataset is critical, 3) You don't need zero-shot capabilities, 4) Computational resources are very limited (ResNet is smaller). ResNet is also simpler to deploy since it doesn't need text processing."

**Q: How does zero-shot learning work?**
- **A**: "Zero-shot learning works by learning a shared representation space. CLIP learns to map both images and text into the same embedding space. At inference, you provide text descriptions of classes (even new ones), encode them, and compare with encoded images. The closest match is the prediction. This works because the model learned semantic relationships during training, so it can generalize to new classes described in text."

**Q: What is the difference between supervised and self-supervised learning?**
- **A**: "Supervised learning (like ResNet) requires labeled data - each image has a class label. Self-supervised learning (like CLIP) learns from unlabeled data by creating its own supervision signal. CLIP uses image-text pairs from the internet - the text provides 'free' labels. This allows learning from much more data without manual labeling."

**Q: Why is cosine similarity used instead of dot product?**
- **A**: "Cosine similarity measures the angle between vectors, which captures semantic similarity regardless of vector magnitude. Dot product would be affected by the length of feature vectors. In CLIP's embedding space, similar concepts point in similar directions, so cosine similarity better captures semantic alignment. It's also normalized, making comparisons more stable."

**Q: Explain the FP16 vs FP32 trade-off.**
- **A**: "FP16 uses half the memory and can be faster on modern GPUs, but has less numerical precision. For neural network inference, this is usually fine because models are robust to small errors. The top predictions typically match FP32, though probability values may differ slightly. It's a good trade-off for deployment where memory and speed matter more than perfect numerical accuracy."

### Key Takeaways for Interview

1. **Understand the models**: ResNet-50 (supervised, fixed classes) vs CLIP (self-supervised, flexible classes)
2. **Know the datasets**: ImageNet (1000 classes, 1.2M images), ImageNet Sketch, artistic images
3. **Explain the concepts**: Zero-shot learning, contrastive learning, distribution shift, cosine similarity
4. **Understand trade-offs**: Accuracy vs. flexibility, memory vs. precision, speed vs. accuracy
5. **Be able to explain code**: Know what each function does and why
6. **Know when to use what**: When ResNet is better, when CLIP is better

---

## Contact

For questions or issues, please refer to the assignment guidelines or course materials.
