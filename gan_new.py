import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
import math

# Constants
SIGNAL_LENGTH = 128
NUM_CHANNELS = 128
BATCH_SIZE = 32
NOISE_DIM = 100
HIDDEN_DIM = 256  # Increased from 64
LEARNING_RATE_G = 0.00005  # Reduced learning rate for stability
LEARNING_RATE_D = 0.00005
BETA1 = 0.5
BETA2 = 0.999
NUM_EPOCHS = 3000  # More training epochs
CRITIC_ITERATIONS = 5  # Train discriminator more times per generator update
LAMBDA_GP = 10  # Gradient penalty coefficient
LAMBDA_SPECTRAL = 5.0  # Weight for spectral loss
LAMBDA_TEMPORAL = 2.0  # Weight for temporal coherence

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out_features, out_features),
        )
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.block(x)
        x += residual
        return self.activation(x)

class ImprovedEEGGenerator(nn.Module):
    def __init__(self, noise_dim, hidden_dim, signal_length, num_channels):
        super(ImprovedEEGGenerator, self).__init__()
        
        self.signal_length = signal_length
        self.num_channels = num_channels
        
        # Initial projection from noise
        self.fc_initial = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Bidirectional LSTM for better temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention mechanism for focusing on important time steps
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim*2, num_heads=4)
        
        # Residual blocks for deeper representation
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim*2, hidden_dim*2),
            ResidualBlock(hidden_dim*2, hidden_dim)
        )
        
        # Final mapping to output shape
        self.fc_out = nn.Linear(hidden_dim, num_channels)
        
        # Special frequency-aware layer
        self.freq_layer = nn.Sequential(
            nn.Linear(signal_length, signal_length),
            nn.Tanh()
        )
        
    def forward(self, noise):
        batch_size = noise.size(0)
        
        # Initial projection and reshape for LSTM
        x = self.fc_initial(noise)  # [batch, seq_len, hidden_dim]
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim*2]
        
        # Self-attention mechanism
        lstm_out_permuted = lstm_out.permute(1, 0, 2)  # [seq_len, batch, hidden_dim*2]
        attn_out, _ = self.attention(lstm_out_permuted, lstm_out_permuted, lstm_out_permuted)
        attn_out = attn_out.permute(1, 0, 2)  # [batch, seq_len, hidden_dim*2]
        
        # Add residual connection around attention
        x = lstm_out + attn_out
        
        # Residual blocks
        x = self.res_blocks(x)
        
        # Generate EEG samples
        eeg_out = self.fc_out(x)  # [batch, seq_len, num_channels]
        
        # Apply frequency-aware transformation across time dimension
        # This helps ensure appropriate spectral characteristics
        eeg_transposed = eeg_out.transpose(1, 2)  # [batch, num_channels, seq_len]
        eeg_freq = self.freq_layer(eeg_transposed)  # [batch, num_channels, seq_len]
        eeg_out = eeg_freq.transpose(1, 2)  # [batch, seq_len, num_channels]
        
        return torch.tanh(eeg_out)  # Ensure output is between -1 and 1

class ImprovedEEGDiscriminator(nn.Module):
    def __init__(self, signal_length, hidden_dim, num_channels):
        super(ImprovedEEGDiscriminator, self).__init__()
        
        self.signal_length = signal_length
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        
        # Convolutional layers with proper dimension calculations
        # After each conv with kernel=5, stride=2, padding=2:
        # output_size = floor((input_size + 2*padding - kernel) / stride) + 1
        self.conv1 = nn.Conv1d(num_channels, hidden_dim//2, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=5, stride=2, padding=2)
        
        # Calculate the final feature map size
        self.final_seq_len = signal_length // (2 ** 3)  # After 3 conv layers with stride 2
        
        # Spectral feature extractor
        self.spectral_extractor = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_dim*2,  # Output channels from conv3
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Final classification layers with proper input dimension
        # Input dim: (lstm_hidden*2 for bidirectional) + spectral_features
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim*2 + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
        # Activation and dropout
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.3)
        
    def extract_spectral_features(self, x):
        """Extract spectral features from input"""
        # Average across time dimension [batch, seq_len, channels] -> [batch, channels]
        x_spectral = x.mean(dim=1)
        return self.spectral_extractor(x_spectral)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract spectral features from original input
        spectral_features = self.extract_spectral_features(x)
        
        # Process through convolutional layers [batch, channels, seq_len]
        x_conv = x.transpose(1, 2)
        
        # Layer 1
        x = self.lrelu(self.conv1(x_conv))
        x = self.dropout(x)
        
        # Layer 2
        x = self.lrelu(self.conv2(x))
        x = self.dropout(x)
        
        # Layer 3
        x = self.lrelu(self.conv3(x))
        x = self.dropout(x)
        
        # Process temporal features with LSTM
        # Reshape: [batch, channels, seq_len] -> [batch, seq_len, channels]
        lstm_in = x.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)
        
        # Get final time step output from LSTM
        lstm_features = lstm_out[:, -1, :]
        
        # Combine features
        combined = torch.cat([lstm_features, spectral_features], dim=1)
        
        # Final classification
        validity = self.fc_layers(combined)
        
        return validity

def reshape_eeg_data(data, target_length=128, target_channels=128):
    """Intelligently reshape EEG data to target dimensions"""
    batch_size = data.shape[0]
    reshaped = np.zeros((batch_size, target_length, target_channels))
    
    # For each sample
    for i in range(batch_size):
        # Get original sample
        sample = data[i]
        
        # Reshape time dimension (take first 128 timepoints or pad if needed)
        time_points = min(sample.shape[0], target_length)
        
        # For channels, we'll either sample or duplicate
        if sample.shape[1] >= target_channels:
            # Downsample channels if we have too many
            channel_indices = np.linspace(0, sample.shape[1]-1, target_channels).astype(int)
            reshaped[i, :time_points, :] = sample[:time_points, channel_indices]
        else:
            # Duplicate channels if we have too few
            reshaped[i, :time_points, :sample.shape[1]] = sample[:time_points, :]
            # Duplicate last channel for remaining positions
            for c in range(sample.shape[1], target_channels):
                reshaped[i, :time_points, c] = sample[:time_points, -1]
        
        # Pad time dimension if needed
        if time_points < target_length:
            for t in range(time_points, target_length):
                reshaped[i, t, :] = reshaped[i, time_points-1, :]
    
    return reshaped

def load_real_eeg_data(batch_size, signal_length, num_channels):
    """Simplified, more robust data loading"""
    try:
        print("Starting to load EEG data batch...")
        file_path = "C:/Users/deeks/OneDrive/Desktop/Research/Preprocessed/segmented_eeg_data.npy"
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"EEG data file not found at {file_path}")
            
        # Memory-map the file to avoid loading everything
        eeg_data_mmap = np.load(file_path, mmap_mode='r')
        num_samples = eeg_data_mmap.shape[0]
        
        # Select random indices
        indices = np.random.choice(num_samples, batch_size, replace=False)
        
        # Load only the selected samples
        batch_data = eeg_data_mmap[indices].copy()
        
        # Reshape with our helper function
        batch_data = reshape_eeg_data(batch_data, signal_length, num_channels)
        
        # Normalize to [-1, 1]
        batch_data = np.clip(batch_data, -5, 5)
        max_val = np.max(np.abs(batch_data))
        if max_val > 0:
            batch_data = batch_data / max_val
            
        print(f"Successfully loaded batch with shape: {batch_data.shape}")
        return torch.tensor(batch_data, dtype=torch.float32)
        
    except Exception as e:
        print(f"Error loading EEG data: {e}")
        print("Creating dummy data instead")
        dummy_data = torch.randn(batch_size, signal_length, num_channels)
        return torch.tanh(dummy_data)

# Custom loss functions
def spectral_loss(real_data, fake_data, fs=256):
    """Calculate loss based on spectral characteristics"""
    # Calculate on CPU for numerical stability
    real_cpu = real_data.detach().cpu().numpy()
    fake_cpu = fake_data.detach().cpu().numpy()
    
    batch_size = real_cpu.shape[0]
    total_loss = 0
    
    # Calculate for a subset of samples to save computation
    samples_to_use = min(10, batch_size)
    
    for i in range(samples_to_use):
        # Take a few random channels for efficiency
        channels_to_use = np.random.choice(real_cpu.shape[2], size=min(16, real_cpu.shape[2]), replace=False)
        
        real_sample = real_cpu[i, :, channels_to_use]
        fake_sample = fake_cpu[i, :, channels_to_use]
        
        # For each channel, compare spectra
        for c in range(len(channels_to_use)):
            # Use scipy's welch method for PSD estimation
            # Use smaller nperseg for our short signals
            nperseg = min(64, real_sample.shape[0] // 2)
            
            try:
                f_real, psd_real = signal.welch(real_sample[:, c], fs=fs, nperseg=nperseg)
                f_fake, psd_fake = signal.welch(fake_sample[:, c], fs=fs, nperseg=nperseg)
                
                # Calculate mean squared error in log space (focus on relative differences)
                psd_real_log = np.log(psd_real + 1e-10)
                psd_fake_log = np.log(psd_fake + 1e-10)
                
                mse = np.mean((psd_real_log - psd_fake_log) ** 2)
                total_loss += mse
            except Exception as e:
                # Fallback if spectral calculation fails
                total_loss += 1.0
    
    # Return mean loss across all samples and channels
    avg_loss = total_loss / (samples_to_use * len(channels_to_use))
    return torch.tensor(avg_loss, device=real_data.device, requires_grad=True)

def temporal_coherence_loss(fake_data):
    """Encourage temporal smoothness and coherence"""
    # Calculate difference between consecutive time steps
    diff = fake_data[:, 1:, :] - fake_data[:, :-1, :]
    
    # Calculate squared differences
    squared_diff = diff.pow(2)
    
    # Mean across all dimensions
    mean_squared_diff = squared_diff.mean()
    
    # Additionally add autocorrelation term to encourage rhythmic patterns
    # Take a few random channels
    batch_size = fake_data.size(0)
    num_channels = fake_data.size(2)
    
    # Sample a few channels for efficiency
    channels_to_use = min(8, num_channels)
    auto_loss = 0.0
    
    for i in range(batch_size):
        for c in range(channels_to_use):
            signal = fake_data[i, :, c]
            
            # Calculate autocorrelation at typical EEG rhythm lags
            for lag in [4, 8, 16]:  # Approximate alpha, theta rhythms
                signal1 = signal[:-lag]
                signal2 = signal[lag:]
                
                # Correlation should be higher for EEG rhythms
                # We use negative correlation to minimize loss
                corr = F.cosine_similarity(signal1, signal2, dim=0)
                auto_loss += (1.0 - corr.abs())  # Want high absolute correlation
    
    auto_loss = auto_loss / (batch_size * channels_to_use * 3)  # Average over samples, channels and lags
    
    # Combine smoothness and autocorrelation terms
    return mean_squared_diff + auto_loss

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """WGAN-GP gradient penalty"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=real_samples.device)
    alpha = alpha.expand_as(real_samples)
    
    # Interpolated samples
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    
    # Get discriminator output for interpolated samples
    d_interpolated = discriminator(interpolated)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated).to(real_samples.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Reshape gradients to be contiguous
    gradients = gradients.reshape(batch_size, -1)  # Changed from view() to reshape()
    
    # Calculate gradient penalty
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

def train():
    """Train the GAN model with improved techniques"""
    # Initialize models
    generator = ImprovedEEGGenerator(NOISE_DIM, HIDDEN_DIM, SIGNAL_LENGTH, NUM_CHANNELS)
    discriminator = ImprovedEEGDiscriminator(SIGNAL_LENGTH, HIDDEN_DIM, NUM_CHANNELS)
    
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)
    
    print(f"Using device: {device}")
    
    # Optimizers with lower learning rate for stability
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, BETA2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, BETA2))
    
    # Add learning rate schedulers
    g_scheduler = optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, 'min', patience=10, factor=0.5)
    d_scheduler = optim.lr_scheduler.ReduceLROnPlateau(d_optimizer, 'min', patience=10, factor=0.5)
    
    # Fixed noise for visualization
    fixed_noise = torch.randn(8, SIGNAL_LENGTH, NOISE_DIM, device=device)
    
    # Track losses for monitoring
    g_losses = []
    d_losses = []
    spectral_losses = []
    temporal_losses = []
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        try:
            # Load real batch and move to device
            real_data = load_real_eeg_data(BATCH_SIZE, SIGNAL_LENGTH, NUM_CHANNELS).to(device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            for _ in range(CRITIC_ITERATIONS):
                d_optimizer.zero_grad()
                
                # Generate fake data
                noise = torch.randn(BATCH_SIZE, SIGNAL_LENGTH, NOISE_DIM, device=device)
                fake_data = generator(noise)
                
                # Real and fake losses
                real_validity = discriminator(real_data)
                fake_validity = discriminator(fake_data.detach())
                
                # Gradient penalty
                gp = compute_gradient_penalty(discriminator, real_data, fake_data.detach())
                
                # Discriminator loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + LAMBDA_GP * gp
                
                d_loss.backward()
                d_optimizer.step()
            
            # -----------------
            # Train Generator
            # -----------------
            g_optimizer.zero_grad()
            
            noise = torch.randn(BATCH_SIZE, SIGNAL_LENGTH, NOISE_DIM, device=device)
            fake_data = generator(noise)
            
            # Generator loss components
            adv_loss = -torch.mean(discriminator(fake_data))
            
            # Add spectral and temporal losses for better quality
            spec_loss = spectral_loss(real_data, fake_data)
            temp_loss = temporal_coherence_loss(fake_data)
            
            # Combined loss
            g_loss = adv_loss + LAMBDA_SPECTRAL * spec_loss + LAMBDA_TEMPORAL * temp_loss
            
            g_loss.backward()
            g_optimizer.step()
            
            # Update schedulers
            g_scheduler.step(g_loss)
            d_scheduler.step(d_loss)
            
            # Save losses for plotting
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            spectral_losses.append(spec_loss.item())
            temporal_losses.append(temp_loss.item())
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
                print(f"  Spec Loss: {spec_loss.item():.4f} | Temp Loss: {temp_loss.item():.4f}")
                # Save model checkpoints periodically
                if (epoch + 1) % 500 == 0:
                    torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
                    torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')
                    
                # Visualize samples periodically
                if (epoch + 1) % 100 == 0:
                    visualize_samples(generator, fixed_noise, epoch)
                    
                    # Plot training curves
                    plt.figure(figsize=(12, 6))
                    plt.subplot(1, 2, 1)
                    plt.plot(g_losses[-100:], label='Generator')
                    plt.plot(d_losses[-100:], label='Discriminator')
                    plt.legend()
                    plt.title('Adversarial Losses')
                    
                    plt.subplot(1, 2, 2)
                    plt.plot(spectral_losses[-100:], label='Spectral')
                    plt.plot(temporal_losses[-100:], label='Temporal')
                    plt.legend()
                    plt.title('Component Losses')
                    
                    plt.savefig(f'training_curves_epoch_{epoch+1}.png')
                    plt.close()
                
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return generator

def visualize_samples(generator, fixed_noise, epoch=None):
    """Visualize generated samples"""
    generator.eval()
    with torch.no_grad():
        fake_data = generator(fixed_noise).cpu().numpy()
    
    # Plot a few samples
    plt.figure(figsize=(15, 10))
    
    # Plot time domain
    for i in range(min(4, fake_data.shape[0])):
        plt.subplot(4, 2, i*2+1)
        # Plot first few channels
        for c in range(min(5, fake_data.shape[2])):
            plt.plot(fake_data[i, :, c], label=f'Ch {c+1}')
        plt.title(f'Sample {i+1} (Time Domain)')
        if i == 0:
            plt.legend()
    
    # Plot frequency domain (FFT)
    for i in range(min(4, fake_data.shape[0])):
        plt.subplot(4, 2, i*2+2)
        
        # Calculate FFT for first few channels
        for c in range(min(5, fake_data.shape[2])):
            signal = fake_data[i, :, c]
            fft_vals = np.abs(np.fft.rfft(signal))
            fft_freqs = np.fft.rfftfreq(len(signal), 1.0/256)
            
            # Plot only from 0 to 50 Hz (typical EEG range)
            mask = fft_freqs <= 50
            plt.plot(fft_freqs[mask], fft_vals[mask], label=f'Ch {c+1}')
            
        plt.title(f'Sample {i+1} (Frequency Domain)')
        plt.xlabel('Frequency (Hz)')
        
    plt.tight_layout()
    
    # Save with epoch number if provided
    if epoch is not None:
        plt.savefig(f'generated_samples_epoch_{epoch+1}.png')
    else:
        plt.savefig('generated_samples.png')
    
    plt.close()
    generator.train()

def generate_synthetic_data(generator, num_files=10, samples_per_file=5000):
    """Generate and save synthetic EEG data files"""
    print(f"Generating {num_files} synthetic EEG data files with {samples_per_file} samples each...")
    
    # Create output directory
    output_dir = "generated_eeg_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set generator to evaluation mode
    generator.eval()
    
    # Determine device
    device = next(generator.parameters()).device
    
    # Generate in batches to avoid memory issues
    batch_size_generation = 50  # Smaller batch for memory efficiency
    num_batches = math.ceil(samples_per_file / batch_size_generation)
    
    for file_idx in range(num_files):
        print(f"Generating file {file_idx+1}/{num_files}...")
        
        # Initialize array to hold all samples for this file
        all_samples = []
        
        # Generate in batches
        for batch_idx in range(num_batches):
            print(f"  Processing batch {batch_idx+1}/{num_batches}...")
            
            # Calculate actual batch size (last batch might be smaller)
            actual_batch_size = min(batch_size_generation, 
                                samples_per_file - batch_idx * batch_size_generation)
            
            # Generate noise and EEG data
            with torch.no_grad():
                noise = torch.randn(actual_batch_size, SIGNAL_LENGTH, NOISE_DIM, device=device)
                fake_eeg_signals = generator(noise)
            
            # Add to our collection
            fake_eeg_np = fake_eeg_signals.cpu().numpy()
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

if __name__ == "__main__":
    # Train the model
    trained_generator = train()
    
    # Generate synthetic data
    generate_synthetic_data(trained_generator)