import numpy as np
import torch
import torch.nn as nn
import RNA

# Define the RNA3DStructureModel (same as the one used for training)
class RNA3DStructureModel(nn.Module):
    def __init__(self, seq_length):
        super(RNA3DStructureModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm1d(64)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 3)  # Output 3D coordinates (X, Y, Z)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)  # Convert to (batch, channels, seq_length)
        x = torch.relu(self.conv1(x))
        x = self.batch_norm(x)
        x = x.permute(2, 0, 1)  # Convert to (seq_length, batch, channels)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # Convert back to (batch, seq_length, channels)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # Output the 3D coordinates (X, Y, Z)
        return x

# Set device to CPU
device = torch.device('cpu')

# Load the trained model (Make sure the model is compatible)
seq_length = 1000  # Make sure this matches the sequence length used during training
model = RNA3DStructureModel(seq_length)

# Load the model weights and explicitly map the model to CPU
model.load_state_dict(torch.load("RNAAlpha.pth", map_location=device))
model.eval()  # Set the model to evaluation mode

# One-hot encode function for RNA sequence
def one_hot_encode(seq):
    encoding = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    return np.array([encoding[nucleotide] for nucleotide in seq])

# Example RNA sequence to classify
rna_seq = "AUGCAGCAGACGAGACAUUCCCAGGA"  # Example RNA sequence
print("RNA Sequence:", rna_seq)

# Encode the RNA sequence using one-hot encoding
encoded_seq = one_hot_encode(rna_seq)

# Add batch dimension (models expect a batch of data)
encoded_seq_expanded = np.expand_dims(encoded_seq, axis=0)  # Shape (1, sequence_length, 4)

# Convert to PyTorch tensor and move it to CPU
input_tensor = torch.tensor(encoded_seq_expanded, dtype=torch.float32).to(device)

# Make the prediction
with torch.no_grad():
    prediction = model(input_tensor)

# Print the prediction (3D coordinates for each nucleotide)
print("Predicted 3D coordinates:")
print(prediction)

# Step 2: Use ViennaRNA to predict the secondary structure
fc = RNA.fold_compound(rna_seq)
structure, mfe = fc.mfe()

# Print secondary structure prediction
print(f"Predicted secondary structure: {structure}")
print(f"Minimum Free Energy (MFE): {mfe}")

# Step 3: Save the data to a .npy file
# Convert PyTorch tensor to NumPy array for compatibility
coords_3d_np = prediction.numpy()

# Create a dictionary to store both the sequence, structure, and 3D coordinates
data = {
    'sequence': rna_seq,
    'structure': structure,
    'mfe': mfe,
    'coords_3d': coords_3d_np
}

# Save the data to a .npy file
np.save('rna_prediction_with_3d_coords.npy', data)

print("Prediction and 3D coordinates saved in 'rna_prediction_with_3d_coords.npy'")
