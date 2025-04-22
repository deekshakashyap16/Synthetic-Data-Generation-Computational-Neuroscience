import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, welch
import matplotlib.pyplot as plt

# Define input and output directories
input_folder = "C:/Users/deeks/OneDrive/Desktop/Research/21679035"  # Contains s01.mat - s52.mat
output_folder = "C:/Users/deeks/OneDrive/Desktop/Research/Preprocessed"
os.makedirs(output_folder, exist_ok=True)

# Improved bandpass filter function (0.5-45 Hz for broader EEG coverage)
def bandpass_filter(data, lowcut=0.5, highcut=45, fs=256, order=5):
    """Apply a bandpass filter to the EEG data"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    print(f"Filtering data of shape: {data.shape}")
    return filtfilt(b, a, data, axis=1)

# Artifact removal function
def remove_artifacts(data, threshold=5):
    """Remove extreme outliers based on z-score threshold"""
    print(f"Removing artifacts from data of shape: {data.shape}")
    clean_data = data.copy()
    
    for i in range(data.shape[0]):
        signal = data[i]
        # Calculate z-scores on a rolling window to handle local artifacts
        window_size = min(2000, signal.shape[0] // 10)
        z_scores = np.zeros_like(signal)
        
        # Calculate local z-scores using sliding windows
        for j in range(0, len(signal) - window_size, window_size // 2):
            window = signal[j:j+window_size]
            window_mean = np.mean(window)
            window_std = np.std(window)
            if window_std > 0:
                z_scores[j:j+window_size] = np.abs((window - window_mean) / window_std)
        
        # Find artifacts using threshold
        artifact_indices = np.where(z_scores > threshold)[0]
        artifact_count = len(artifact_indices)
        
        # Replace artifacts with interpolated values
        for idx in artifact_indices:
            if idx > 0 and idx < len(signal) - 1:
                # Simple linear interpolation
                clean_data[i, idx] = (clean_data[i, idx-1] + clean_data[i, idx+1]) / 2
            else:
                # Edge case - replace with nearby value
                clean_data[i, idx] = clean_data[i, idx-1] if idx > 0 else clean_data[i, idx+1]
        
        print(f"  Channel {i+1}: Removed {artifact_count} artifacts")
    
    return clean_data

# Modified segmentation function with more lenient segment selection
def segment_data(data, segment_length=2048, overlap=0.5, max_segments=None):
    """Segment data into smaller chunks with overlap"""
    print(f"Segmenting data of shape: {data.shape}")
    step = int(segment_length * (1-overlap))
    
    # Calculate potential start positions for segments
    potential_positions = list(range(0, data.shape[1] - segment_length, step))
    
    # Modified approach: Calculate channel-specific thresholds
    # Instead of rejecting entire segments, score them
    channel_means = np.mean(data, axis=1)
    channel_stds = np.std(data, axis=1)
    
    segment_scores = []
    for pos in potential_positions:
        score = 0
        for i in range(data.shape[0]):  # Check each channel
            segment = data[i, pos:pos+segment_length]
            # Calculate a quality score based on deviation from channel statistics
            max_abs_value = np.max(np.abs(segment))
            if max_abs_value > 10 * channel_stds[i]:  # More lenient threshold
                score -= 1  # Penalize but don't immediately reject
            else:
                score += 1  # Reward good segments
        segment_scores.append(score)
    
    # Sort positions by score (best first)
    scored_positions = sorted(zip(potential_positions, segment_scores), 
                              key=lambda x: x[1], reverse=True)
    
    # Take at least 5 segments, more if available with positive scores
    valid_positions = [pos for pos, score in scored_positions[:max(5, len([s for _, s in scored_positions if s > 0]))]]
    
    # If we still don't have any valid segments, try even more lenient approach
    if len(valid_positions) < 5:
        print("Warning: Few valid segments found with standard criteria, using more lenient criteria")
        # Take at least some segments even if they're not ideal
        valid_positions = [pos for pos, _ in scored_positions[:max(5, len(scored_positions)//4)]]
    
    # If still nothing usable, print diagnostic info and return zeros
    if len(valid_positions) == 0:
        print("Warning: No valid segments found! Data may have serious quality issues.")
        print(f"Data statistics: mean={np.mean(data):.4f}, std={np.std(data):.4f}, min={np.min(data):.4f}, max={np.max(data):.4f}")
        return np.zeros((data.shape[0], 1, segment_length))
    
    # Initialize 3D array: [channels, segments, segment_length]
    num_channels = data.shape[0]
    num_segments = len(valid_positions)
    segmented_data = np.zeros((num_channels, num_segments, segment_length))
    
    # Extract segments for each channel
    for i in range(num_channels):  # For each channel
        for j, start_pos in enumerate(valid_positions):
            segment = data[i, start_pos:start_pos+segment_length]
            segmented_data[i, j] = segment
    
    print(f"  Created {num_segments} valid segments across all channels")
    print(f"Final segmented shape: {segmented_data.shape}")
    return segmented_data
           
# Normalization function
def normalize_segments(data):
    """Normalize each segment to have zero mean and unit variance using robust statistics"""
    print(f"Normalizing data of shape: {data.shape}")
    normalized = np.zeros_like(data)
    
    for i in range(data.shape[0]):  # For each channel
        for j in range(data.shape[1]):  # For each segment
            # Use robust statistics to avoid influence of any remaining artifacts
            q1, q3 = np.percentile(data[i, j], [25, 75])
            iqr = q3 - q1
            if iqr > 0:  # Avoid division by zero
                normalized[i, j] = (data[i, j] - np.median(data[i, j])) / iqr
            else:
                normalized[i, j] = (data[i, j] - np.mean(data[i, j])) / (np.std(data[i, j]) + 1e-10)
    
    return normalized

# Quality check function with visualization
def check_preprocessing_quality(file_path, raw_data, processed_data):
    """Check the quality of preprocessing by comparing statistics and plots"""
    base_filename = os.path.basename(file_path).replace('.mat', '')
    
    # Create quality check directory
    quality_dir = os.path.join(output_folder, "quality_checks")
    os.makedirs(quality_dir, exist_ok=True)
    
    # Calculate statistics
    raw_stats = {
        "min": np.min(raw_data),
        "max": np.max(raw_data),
        "mean": np.mean(raw_data),
        "std": np.std(raw_data)
    }
    
    proc_stats = {
        "min": np.min(processed_data),
        "max": np.max(processed_data),
        "mean": np.mean(processed_data),
        "std": np.std(processed_data)
    }
    
    # Print statistics
    print("\nPreprocessing Quality Check:")
    print(f"Raw data stats: min={raw_stats['min']:.2f}, max={raw_stats['max']:.2f}, mean={raw_stats['mean']:.2f}, std={raw_stats['std']:.2f}")
    print(f"Processed data stats: min={proc_stats['min']:.2f}, max={proc_stats['max']:.2f}, mean={proc_stats['mean']:.2f}, std={proc_stats['std']:.2f}")
    
    # Plot raw vs processed data
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot raw data - first channel, first 1024 samples
    raw_channel = 0
    raw_plot_data = raw_data[raw_channel, :1024]
    axes[0].plot(raw_plot_data)
    axes[0].set_title(f"{base_filename} - Raw Signal (Channel {raw_channel+1})")
    axes[0].grid(True)
    
    # Plot processed data - first channel, first segment, all samples
    if len(processed_data.shape) == 3:  # [channels, segments, samples]
        proc_plot_data = processed_data[raw_channel, 0, :]
    else:  # [channels, samples]
        proc_plot_data = processed_data[raw_channel, :1024]
        
    axes[1].plot(proc_plot_data)
    axes[1].set_title(f"{base_filename} - Processed Signal (Channel {raw_channel+1})")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(quality_dir, f"{base_filename}_signal_comparison.png"))
    plt.close()
    
    # Plot PSD
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fs = 256  # Sampling frequency
    
    # Raw data PSD
    f, Pxx = welch(raw_data[raw_channel, :], fs=fs, nperseg=fs*2)
    axes[0].semilogy(f, Pxx)
    axes[0].set_title(f"{base_filename} - Raw Power Spectral Density")
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Power/Frequency (dB/Hz)')
    axes[0].grid(True)
    axes[0].set_xlim([0, 50])
    
    # Mark EEG bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }
    colors = ['r', 'g', 'b', 'm', 'y']
    for (band, (low, high)), color in zip(bands.items(), colors):
        axes[0].axvspan(low, high, color=color, alpha=0.3, label=band)
    axes[0].legend()
    
    # Processed data PSD
    if len(processed_data.shape) == 3:
        f, Pxx = welch(processed_data[raw_channel, 0, :], fs=fs, nperseg=fs*2)
    else:
        f, Pxx = welch(processed_data[raw_channel, :], fs=fs, nperseg=fs*2)
        
    axes[1].semilogy(f, Pxx)
    axes[1].set_title(f"{base_filename} - Processed Power Spectral Density")
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power/Frequency (dB/Hz)')
    axes[1].grid(True)
    axes[1].set_xlim([0, 50])
    
    # Mark EEG bands
    for (band, (low, high)), color in zip(bands.items(), colors):
        axes[1].axvspan(low, high, color=color, alpha=0.3, label=band)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(quality_dir, f"{base_filename}_psd_comparison.png"))
    plt.close()
    
    return raw_stats, proc_stats

# Initialize lists to store all processed data and track excluded subjects
all_processed_data = []
excluded_subjects = []

# Process all .mat files
file_count = 0
for file_name in sorted(os.listdir(input_folder)):
    if file_name.endswith(".mat") and file_name.startswith("s") and file_count < 52:
        file_path = os.path.join(input_folder, file_name)
        print(f"\n🔄 Processing: {file_name} ({file_count+1}/52)")
        
        try:
            # Load .mat file
            mat_data = sio.loadmat(file_path)
            
            # Extract EEG data
            if 'eeg' not in mat_data:
                print(f"⚠️ Skipping {file_name}: 'eeg' key not found")
                excluded_subjects.append(file_name)
                continue
                
            eeg_data = mat_data['eeg']
            
            # Check for required fields
            if not (hasattr(eeg_data, 'dtype') and hasattr(eeg_data.dtype, 'names') and 
                   'imagery_left' in eeg_data.dtype.names and 'imagery_right' in eeg_data.dtype.names):
                print(f"⚠️ Skipping {file_name}: Required fields not found")
                if hasattr(eeg_data, 'dtype') and hasattr(eeg_data.dtype, 'names'):
                    print(f"Available fields: {eeg_data.dtype.names}")
                excluded_subjects.append(file_name)
                continue
                
            # Extract left and right imagery data
            imagery_left = eeg_data['imagery_left'][0,0]
            imagery_right = eeg_data['imagery_right'][0,0]
            
            # Ensure data is in the correct format
            if len(imagery_left.shape) != 2:
                print(f"⚠️ Reshaping left imagery data in {file_name}")
                imagery_left = imagery_left.reshape(-1, imagery_left.size)
                
            if len(imagery_right.shape) != 2:
                print(f"⚠️ Reshaping right imagery data in {file_name}")
                imagery_right = imagery_right.reshape(-1, imagery_right.size)
                
            # Ensure matching number of channels
            if imagery_left.shape[0] != imagery_right.shape[0]:
                print(f"⚠️ Channel count mismatch in {file_name}")
                min_channels = min(imagery_left.shape[0], imagery_right.shape[0])
                imagery_left = imagery_left[:min_channels, :]
                imagery_right = imagery_right[:min_channels, :]
                
            # Concatenate imagery data along time axis
            combined_imagery = np.concatenate([imagery_left, imagery_right], axis=1)
            print(f"✅ Combined imagery shape: {combined_imagery.shape}")
            
            # Skip files with too short data
            if combined_imagery.shape[1] <= 30:
                print(f"⚠️ Skipping {file_name}: Data length too short ({combined_imagery.shape[1]})")
                excluded_subjects.append(file_name)
                continue
                
            # Store raw data for quality check
            raw_data = combined_imagery.copy()
            
            # Apply artifact removal
            clean_data = remove_artifacts(combined_imagery)
            
            # Apply bandpass filter
            filtered_data = bandpass_filter(clean_data)
            
            # Segment data
            segmented_data = segment_data(filtered_data, segment_length=2048, overlap=0.5)
            
            # Check if segmentation failed (returned zeros)
            if np.all(np.abs(segmented_data) < 0.0001):
                print(f"⚠️ Skipping {file_name}: Segmentation produced flat data")
                excluded_subjects.append(file_name)
                continue
                
            # Check if we have too few segments (less than 2)
            if segmented_data.shape[1] < 2:
                print(f"⚠️ Skipping {file_name}: Too few segments ({segmented_data.shape[1]})")
                excluded_subjects.append(file_name)
                continue
            
            # Normalize segments
            normalized_data = normalize_segments(segmented_data)
            
            # Double check for flatlines after normalization
            signal_variance = np.var(normalized_data[:, 0, :])
            if signal_variance < 0.01:  # Very low variance indicates flatline
                print(f"⚠️ Skipping {file_name}: Data has too low variance ({signal_variance:.6f})")
                excluded_subjects.append(file_name)
                continue
            
            # Check quality of preprocessing
            raw_stats, proc_stats = check_preprocessing_quality(file_path, raw_data, normalized_data)
            
            # Save processed data
            output_path = os.path.join(output_folder, file_name.replace(".mat", "_processed.npy"))
            np.save(output_path, normalized_data)
            print(f"✅ Saved processed data to {output_path}")
            
            # Append to list for merging
            all_processed_data.append(normalized_data)
            file_count += 1
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {str(e)}")
            excluded_subjects.append(file_name)
            import traceback
            traceback.print_exc()

# After processing, print summary of excluded subjects
print(f"\n⚠️ Excluded {len(excluded_subjects)} subjects due to data quality issues:")
for subject in excluded_subjects:
    print(f"  - {subject}")

# Merge all processed data
if all_processed_data:
    try:
        print("\n🔄 Merging all processed data...")
        
        # Get shapes of all processed arrays
        shapes = [data.shape for data in all_processed_data]
        print(f"Shapes of processed data: {shapes}")
        
        # Find minimum number of segments across all files to ensure consistent merging
        min_segments = min(shape[1] for shape in shapes)
        print(f"Using {min_segments} segments from each file for consistent merging")
        
        # Trim all arrays to have the same number of segments
        uniform_data = [data[:, :min_segments, :] for data in all_processed_data]
        
        # Merge along a new axis (creating [subjects, channels, segments, samples])
        merged_data = np.stack(uniform_data, axis=0)
        print(f"Merged data shape: {merged_data.shape}")
        
        # Save merged data
        merged_path = os.path.join(output_folder, "merged_eeg_data.npy")
        np.save(merged_path, merged_data)
        print(f"✅ Successfully merged all data and saved to {merged_path}")
        
        # Save list of included subjects for reference
        included_subjects = [file_name for file_name in sorted(os.listdir(input_folder)) 
                            if file_name.endswith(".mat") and file_name not in excluded_subjects]
        with open(os.path.join(output_folder, "included_subjects.txt"), "w") as f:
            for subject in included_subjects:
                f.write(f"{subject}\n")
        print(f"✅ Saved list of included subjects ({len(included_subjects)}) to included_subjects.txt")
        
    except Exception as e:
        print(f"❌ Error during merging: {str(e)}")
        import traceback
        traceback.print_exc()
else:
    print("❌ No valid EEG data found to merge.")

print("\n✅ Preprocessing completed!")