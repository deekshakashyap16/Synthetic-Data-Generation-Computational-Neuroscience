import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

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
    eeg_data = np.load("C:/Users/deeks/OneDrive/Desktop/Research/Preprocessed/merged_eeg_data.npy")  # Load preprocessed EEG
    eeg_data = eeg_data[:, :signal_length]  # Ensure correct length
    eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
    indices = torch.randint(0, eeg_data.shape[0], (batch_size,))
    return eeg_data[indices].unsqueeze(-1)  # Add channel dimension
batch_size = 5
noise = torch.randn(batch_size, signal_length, noise_dim)
fake_eeg_data = generator(noise).detach()

plt.figure(figsize=(10, 5))

batch_size = fake_eeg_data.shape[0]

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
