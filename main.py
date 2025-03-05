import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# RNA Folding Prediction Model - Predict Secondary Structure
class RNA3DStructureModel(nn.Module):
    def __init__(self, seq_length):
        super(RNA3DStructureModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(64)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 4)  # Output size 4 for predicting secondary structure (base pairs)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch, channels, seq_length)
        x = torch.relu(self.conv1(x))
        x = self.batch_norm(x)
        x = x.permute(2, 0, 1)  # Convert to (seq_length, batch, channels)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Convert back to (batch, seq_length, channels)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output probabilities for base pairing
        return x

# Load the trained model
seq_length = 100  # Ensure this matches the model's trained sequence length
model = RNA3DStructureModel(seq_length)
model.load_state_dict(torch.load("RNAAlpha.pth"))
model.eval()

# One-hot encode function for RNA sequence
def one_hot_encode(seq):
    encoding = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    return np.array([encoding[nucleotide] for nucleotide in seq])

# Example RNA sequence to classify
rna_seq = "AUGC"  # Example sequence

# Encode the RNA sequence using one-hot encoding
encoded_seq = one_hot_encode(rna_seq)

# Add batch dimension (since models expect a batch of data)
encoded_seq_expanded = np.expand_dims(encoded_seq, axis=0)  # Shape (1, sequence_length, 4)

# Convert to PyTorch tensor
input_tensor = torch.tensor(encoded_seq_expanded, dtype=torch.float32)

# Make the prediction
with torch.no_grad():
    prediction = model(input_tensor)

# Interpret the secondary structure prediction
# Here we interpret the output as predicted base-pairing information
base_pair_probs = prediction.numpy()

# For simplicity, print out base-pairing probabilities (A-U, G-C)
print(f"Base Pair Probabilities for RNA Sequence: {base_pair_probs}")

# Further processing would be needed to convert this into a 3D structure (e.g., using ViennaRNA)

