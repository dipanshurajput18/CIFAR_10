## 

CIFAR-10 Image Classification using Convolutional Neural Networks
This repository contains Python code to train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset for image classification.

## Dataset
The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:

Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck


## Model Architecture
The CNN model is constructed using TensorFlow and Keras:

![Alt text]()


### Input Layer: 32x32x3 RGB image
### Convolutional Layers:
### 32 filters of size 3x3 with ReLU activation
### Max pooling with 2x2 window
### 64 filters of size 3x3 with ReLU activation
### Max pooling with 2x2 window
### 64 filters of size 3x3 with ReLU activation
### Fully Connected Layers:
### Dense layer with 64 units and ReLU activation
### Output layer with 10 units (corresponding to class labels)
### Training
### -->The model is trained with:

#### ----Optimizer: Adam
#### ----Loss function: Sparse Categorical Crossentropy
#### ----Metrics: Accuracy
#### ----Evaluation: After training for 10 epochs, the model achieves a test accuracy of 70% on unseen data.

## Requirements
```TensorFlow
-NumPy
-Matplotlib
-PIL (Python Imaging Library)
-Opencv
-Keras
```
## Usage
To predict the class of a new image, use the predict_image function defined in predict.py. Ensure the image is resized to 32x32 pixels and normalized before prediction.

python
```
image_path = "path_to_your_image.jpg"
predicted_class_name = predict_image(image_path)
print(f"Predicted class: {predicted_class_name}")
```