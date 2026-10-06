# CNN From Scratch

This repository contains manual implementations of fundamental Convolutional Neural Network (CNN) operations using Python and NumPy.

The purpose of this repository is to understand the internal working of CNN layers by implementing the operations step-by-step without using high-level CNN layers such as `Conv2D`, `MaxPooling2D`, `Flatten`, or `ReLU` from TensorFlow/Keras.

## Concepts Covered

The repository currently includes:

* Image representation using NumPy
* 2D Convolution operation
* Kernel-based edge detection
* ReLU activation function
* 2x2 Max Pooling
* Flatten layer
* Fully Connected layer
* Manual calculation of final output

## CNN Flow

The implementations gradually demonstrate the following CNN pipeline:

```text
Input Image
     ↓
Convolution
     ↓
Feature Map
     ↓
ReLU
     ↓
Max Pooling
     ↓
Flatten
     ↓
Fully Connected Layer
     ↓
Final Output
```

## Repository Structure

```text
CNN-From-Scratch/
│
├── README.md
│
├── 02_Kernel_and_Convolution/
│   └── horizontal_edge_detection.py
│
├── 03_Activation_and_Pooling/
│   └── relu_and_max_pooling.py
│
└── 04_Flatten_and_Fully_Connected/
    └── flatten_and_fully_connected.py
```

## 1. Convolution and Edge Detection

### File

`02_Kernel_and_Convolution/horizontal_edge_detection.py`

### Objective

This program demonstrates the basic convolution operation using a manually created grayscale image and a 3x3 kernel.

### Operations Performed

1. Create a 5x5 grayscale image.
2. Create a 3x3 horizontal edge-detection kernel.
3. Extract 3x3 regions from the image.
4. Perform element-wise multiplication between the region and kernel.
5. Calculate the sum of the multiplied values.
6. Store the results in a feature map.

### Formula

```text
Output Size = (Input Size - Kernel Size) + 1
```

For a 5x5 image and a 3x3 kernel:

```text
(5 - 3 + 1) × (5 - 3 + 1)

= 3 × 3
```

### Concepts Demonstrated

* Grayscale image representation
* Convolution
* Kernel
* Feature extraction
* Feature map
* Horizontal edge detection

### Technologies

* Python
* NumPy

---

## 2. ReLU and Max Pooling

### File

`03_Activation_and_Pooling/relu_and_max_pooling.py`

### Objective

This program demonstrates how ReLU and Max Pooling are applied to a feature map.

### Operations Performed

1. Create a feature map containing positive and negative values.
2. Apply the ReLU activation function.
3. Apply 2x2 Max Pooling.
4. Display the output after each step.

### ReLU

The ReLU function is:

```text
ReLU(x) = max(0, x)
```

Therefore:

```text
Negative value → 0
Positive value → remains unchanged
```

Example:

```text
-3 → 0
 5 → 5
```

### Max Pooling

A 2x2 pooling window is used to select the maximum value from each region.

Example:

```text
[3  5]
[1  2]
```

Maximum value:

```text
5
```

### Why Pooling Reduces the Size

Pooling replaces a group of values with a single value.

For example, a 4x4 feature map using a 2x2 pooling window with stride 2 becomes a 2x2 feature map.

```text
4x4 → 2x2
```

This reduces the spatial dimensions of the feature map while retaining the strongest feature from each region.

### Concepts Demonstrated

* ReLU activation
* Negative value removal
* Max Pooling
* Pooling window
* Spatial size reduction
* Feature preservation

### Technologies

* Python
* NumPy

---

## 3. Flatten and Fully Connected Layer

### File

`04_Flatten_and_Fully_Connected/flatten_and_fully_connected.py`

### Objective

This program demonstrates how a 2D feature map is converted into a 1D vector and passed to a manually implemented Fully Connected layer.

### Input

```text
[[6, 4],
 [8, 6]]
```

### Flatten Output

```text
[6, 4, 8, 6]
```

### Operations Performed

1. Create a 2D feature map.
2. Convert the 2D matrix into a 1D vector using the Flatten operation.
3. Define weights and bias manually.
4. Calculate the weighted sum.
5. Calculate the final output.

### Fully Connected Layer Formula

```text
Output = Sum(Input × Weight) + Bias
```

Example:

```text
Input   = [6, 4, 8, 6]
Weights = [0.5, 0.2, 0.1, 0.4]
Bias    = 1
```

Calculation:

```text
(6 × 0.5) + (4 × 0.2) + (8 × 0.1) + (6 × 0.4) + 1

= 3.0 + 0.8 + 0.8 + 2.4 + 1

= 8.0
```

### Role of Flatten Layer

The convolution and pooling layers produce feature maps that retain spatial dimensions.

The Flatten layer converts these multidimensional feature maps into a one-dimensional vector so that the values can be provided as input to a Fully Connected layer.

```text
2D Feature Map
      ↓
   Flatten
      ↓
1D Vector
      ↓
Fully Connected Layer
      ↓
Final Output
```

The Flatten layer only changes the shape of the data. It does not perform feature extraction or modify the values.

### Concepts Demonstrated

* Flattening
* 2D to 1D conversion
* Weights
* Bias
* Weighted sum
* Fully Connected layer
* Manual output calculation

### Technologies

* Python
* NumPy

---

## Learning Approach

All the core CNN operations in these programs are implemented manually to understand the calculations involved in CNNs.

High-level deep learning layers are intentionally avoided in these basic implementations.

For example, instead of:

```python
Conv2D()
MaxPooling2D()
Flatten()
```

the operations are implemented using NumPy arrays, loops, mathematical calculations, and basic functions.

## Future Implementations

More CNN concepts will be added progressively, including:

* Vertical edge detection
* Different convolution kernels
* Average Pooling
* Convolution + ReLU
* Convolution + ReLU + Pooling
* Multiple feature maps
* Fully Connected layers with multiple neurons
* Softmax
* Complete CNN pipeline
* Manual forward propagation
* CNN classification example

## Requirements

Python 3.x and NumPy are required.

Install NumPy using:

```bash
pip install numpy
```

## Running the Programs

Clone the repository:

```bash
git clone https://github.com/aartiwamane/CNN-From-Scratch.git
```

Navigate into the repository:

```bash
cd CNN-From-Scratch
```

Run any program using:

```bash
python filename.py
```

Example:

```bash
python 02_Kernel_and_Convolution/horizontal_edge_detection.py
```

## Key Learning Outcome

This repository is focused on building a strong conceptual understanding of how CNNs process image data:

```text
Image
 ↓
Convolution
 ↓
Feature Map
 ↓
ReLU
 ↓
Pooling
 ↓
Flatten
 ↓
Fully Connected Layer
 ↓
Output
```

The implementations are designed for learning, experimentation, and understanding CNN fundamentals from the inside out.

## Author

**Aarti Wamane**

M.Tech Computer Engineering

GitHub: [aartiwamane](https://github.com/aartiwamane)

