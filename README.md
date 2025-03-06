# RNAAlpha: Advanced RNA 3D Folding and Structure Prediction Model

## Overview
RNAAlpha is a deep learning model designed for predicting RNA 3D structures and folding patterns. Developed by WolfTech Innovations, this model leverages convolutional layers, transformers, and LSTMs to enhance RNA structure prediction accuracy.

## Features
- **RNA 3D Structure Prediction**: Uses a transformer-based approach for precise 3D coordinate predictions.
- **RNA Folding Prediction**: Incorporates an LSTM-based module to predict base-pair probabilities.
- **One-Hot Encoding**: Converts RNA sequences into a format suitable for deep learning.
- **Synthetic Data Generation**: Creates simulated RNA sequences and structures for training.
- **Efficient Training**: Implements AdamW optimizer with Mean Squared Error (MSE) and Binary Cross-Entropy Loss (BCE) for training stability.

## Model Architecture
RNAAlpha consists of the following components:
- **Convolutional Layer**: Extracts feature representations from RNA sequences.
- **Transformer Encoder**: Captures complex dependencies within RNA structures.
- **Fully Connected Layers**: Maps encoded features to 3D coordinates.
- **LSTM Module**: Predicts RNA folding probabilities using bidirectional LSTMs.

## Installation & Usage
### Prerequisites
Ensure you have Python installed along with the following dependencies:
```bash
pip install numpy torch scikit-learn
```

### Running the Model
To train RNAAlpha on synthetic data:
```python
python train_model.py
```

### Loading the Model
To use a pre-trained RNAAlpha model:
```python
import torch
from RNAAlpha import RNAAlpha3DModel

model = RNAAlpha3DModel(seq_length=100)
model.load_state_dict(torch.load("RNAAlpha_Model.pth"))
model.eval()
```

## Training Details
- **Dataset**: Simulated RNA sequences with 3D coordinates and folding labels.
- **Batch Size**: 64
- **Learning Rate**: 1e-4
- **Weight Decay**: 5e-5
- **Epochs**: 30

## Results & Performance
RNAAlpha achieves high accuracy in predicting RNA structures and folding patterns, making it suitable for bioinformatics and structural biology research.

## License
RNAAlpha is developed by **WolfTech Innovations**, and released under the GPL2 License

## Contact
For inquiries or contributions, contact **WolfTech Innovations** at [spoinkosgithub@gmail.com](mailto:spoinkosgithub@gmail.com).
