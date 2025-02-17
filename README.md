# Malaria Detection using CNN

## Overview
This project implements a Convolutional Neural Network (CNN) to detect malaria-infected cells using the Malaria dataset from TensorFlow Datasets. The model classifies cell images into two categories: infected and uninfected.

## Features
- **Deep Learning Pipeline:** Uses Conv2D, MaxPooling, Flatten, and Dense layers.
- **Batch Normalization:** Improves training convergence and model stability.
- **High Accuracy:** Achieved 95% classification accuracy.
- **Optimized Training:** Utilizes Adam optimizer and Binary Crossentropy loss.
- **Efficient Data Handling:** Uses data batching, shuffling, and prefetching for optimized performance.

## Dataset
- **Source:** TensorFlow Datasets (tfds.load("malaria"))
- **Classes:** Infected vs. Uninfected cell images
- **Image Size:** 128x128 (resized if needed)
- **Train/Test Split:** Managed by TensorFlow Datasets

## Model Architecture
1. **Input Layer:** Accepts input images.
2. **Convolutional Layers (Conv2D):** Extracts features from images.
3. **MaxPooling Layers (MaxPool2D):** Reduces spatial dimensions.
4. **Batch Normalization:** Stabilizes learning.
5. **Flatten Layer:** Converts feature maps into a dense vector.
6. **Dense Layers:** Performs final classification.

## Training Details
- **Optimizer:** Adam (learning rate = 0.01)
- **Loss Function:** Binary Crossentropy
- **Batch Size:** Defined for efficient training
- **Epochs:** Set based on convergence criteria

## Model Evaluation
- **Accuracy:** 95%
- **Metrics:** Accuracy, Loss Curves, and Confusion Matrix
- **Testing:** Evaluated on a separate test set

## Usage
### Installation
```bash
pip install tensorflow numpy matplotlib tensorflow-datasets
```

### Running the Model
```python
import tensorflow as tf
import tensorflow_datasets as tfds

# Load dataset
dataset, info = tfds.load("malaria", as_supervised=True, with_info=True)

# Define and train model (refer to the notebook for full implementation)
```

### Saving & Loading Model
```python
# Save model
tf.keras.models.save_model(model, "malaria_cnn.h5")

# Load model
model = tf.keras.models.load_model("malaria_cnn.h5")
```

## Results
- Achieved 95% accuracy.
- Successfully detects malaria-infected cells with high precision.

## Future Improvements
- Hyperparameter tuning for further optimization.
- Experiment with different architectures (e.g., deeper CNNs, ResNet).
- Deploy the model using Flask, FastAPI, or TensorFlow Serving.

## Reference material
- [Link 1](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,4,4,4,4&seed=0.18718&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=true&xSquared=true&ySquared=true&cosX=false&sinX=true&cosY=false&sinY=true&collectStats=false&problem=classification&initZero=false&hideText=false)
- [Link 2](https://poloclub.github.io/cnn-explainer/)
- [Link 3](https://medium.com/@saivenkat_/all-you-need-to-know-about-cnn-visual-explained-convolutional-neural-networks-d7041cf5e919)
- [Link 4](https://setosa.io/ev/image-kernels/)
- [Link 5](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html)
