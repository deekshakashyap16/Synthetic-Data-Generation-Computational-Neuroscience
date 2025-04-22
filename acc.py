import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

class EEGGenerator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, signal_length):
        super(EEGGenerator, self).__init__()
        self.lstm = nn.LSTM(noise_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, signal_length)

    def forward(self, noise):
        lstm_out, _ = self.lstm(noise)
        signal = self.fc(lstm_out)
        return torch.tanh(signal)  # Normalize between -1 and 1
        
class EEGDiscriminator(nn.Module):
    def __init__(self, signal_length, hidden_dim):
        super(EEGDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)  # Fixed input size
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, signal):
        lstm_out, _ = self.lstm(signal)  # Shape: (batch_size, signal_length, hidden_dim)
        validity = self.fc(lstm_out[:, -1, :])  # Use last time step's output
        return self.sigmoid(validity)  # Output probability
        
noise_dim = 100  # Dimension of noise vector
hidden_dim = 64  # Hidden size of LSTM
signal_length = 128  # Length of EEG signals (number of time steps)
batch_size = 32
num_epochs = 1000

generator = EEGGenerator(noise_dim, hidden_dim, signal_length)
discriminator = EEGDiscriminator(signal_length, hidden_dim)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)

def load_real_eeg_data(batch_size, signal_length):
    eeg_data = np.load("C:/Users/deeks/OneDrive/Desktop/Research/Prepro/segmented_eeg_data.npy")  # Load preprocessed EEG
    eeg_data = eeg_data[:, :signal_length]  # Ensure correct length
    eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
    indices = torch.randint(0, eeg_data.shape[0], (batch_size,))
    return eeg_data[indices].unsqueeze(-1)  # Add channel dimension
    
# Optional visualization code
def visualize_samples():
    batch_size = 5
    noise = torch.randn(batch_size, signal_length, noise_dim)
    fake_eeg_data = generator(noise).detach()

    plt.figure(figsize=(10, 5))

    for i in range(batch_size):
        # Remove extra dimensions like [128, 1] -> [128]
        signal = fake_eeg_data[i].squeeze().detach().cpu().numpy()
        plt.plot(signal, label=f"Sample {i+1}")

    plt.title("Generated Fake EEG Signals")
    plt.xlabel("Timestep")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Create output directory for generated files
output_dir = "generated_eeg_data"
os.makedirs(output_dir, exist_ok=True)

# Parameters for generating files - ADJUSTED FOR LARGER FILES
num_files = 10
samples_per_file = 5000  # Increased from 100 to 5000 to match original file size

print(f"Generating {num_files} synthetic EEG data files with {samples_per_file} samples each...")

# Generate files in smaller batches to avoid memory issues
batch_size_generation = 500  # Process 500 samples at a time
num_batches = samples_per_file // batch_size_generation

for file_idx in range(num_files):
    print(f"Generating file {file_idx+1}/{num_files}...")
    
    # Initialize array to hold all samples for this file
    all_samples = []
    
    # Generate in batches
    for batch_idx in range(num_batches):
        print(f"  Processing batch {batch_idx+1}/{num_batches}...")
        noise = torch.randn(batch_size_generation, signal_length, noise_dim)
        
        with torch.no_grad():
            fake_eeg_signals = generator(noise)
        
        # Add to our collection
        fake_eeg_np = fake_eeg_signals.squeeze(-1).cpu().numpy()
        all_samples.append(fake_eeg_np)
    
    # Combine all batches
    combined_samples = np.concatenate(all_samples, axis=0)
    
    # Save as .npy file
    file_path = os.path.join(output_dir, f"synthetic_eeg_data_{file_idx+1}.npy")
    np.save(file_path, combined_samples)
    
    actual_size_kb = os.path.getsize(file_path) / 1024
    print(f"File {file_idx+1}/{num_files} saved: {file_path} (Size: {actual_size_kb:.2f} KB)")

print("Generation complete!")

# Calculate total size of generated files
total_size = 0
for file_idx in range(num_files):
    file_path = os.path.join(output_dir, f"synthetic_eeg_data_{file_idx+1}.npy")
    file_size = os.path.getsize(file_path)
    total_size += file_size

print(f"Total size of all generated files: {total_size / (1024*1024):.2f} MB")