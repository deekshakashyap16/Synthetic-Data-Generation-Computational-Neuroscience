""" import os
import numpy as np
import scipy.io as sio

# Define input and output directories
input_folder = "C:/Users/deeks/OneDrive/Desktop/Research/21679035"
output_folder = "C:/Users/deeks/OneDrive/Desktop/Research/Preprocessed"
os.makedirs(output_folder, exist_ok=True)

# Initialize list to store all subjects' imagery data
all_eeg_data = []

# Loop through all .mat files in the input folder
for file_name in sorted(os.listdir(input_folder)):
    if file_name.endswith(".mat"):
        file_path = os.path.join(input_folder, file_name)
        print(f"🔄 Processing: {file_name}")
        
        # Load .mat file
        mat_data = sio.loadmat(file_path)
        
        # Ensure 'eeg' key exists
        if 'eeg' not in mat_data:
            print(f"⚠️ Skipping {file_name}: 'eeg' key not found")
            continue
        
        eeg_data = mat_data['eeg']
        
        # Extract only motor imagery data
        if 'imagery_left' in eeg_data.dtype.names and 'imagery_right' in eeg_data.dtype.names:
            imagery_left = np.array(eeg_data['imagery_left'][()])
            imagery_right = np.array(eeg_data['imagery_right'][()])
            
            # Combine left and right imagery data
            combined_imagery = np.concatenate([imagery_left, imagery_right], axis=1)
            
            # Save processed data
            output_path = os.path.join(output_folder, file_name.replace(".mat", "_processed.npy"))
            np.save(output_path, combined_imagery)
            
            print(f"✅ Saved processed data to {output_path}")
            
            # Append to list for potential merging
            all_eeg_data.append(combined_imagery)
        else:
            print(f"⚠️ Skipping {file_name}: 'imagery_left' or 'imagery_right' not found")

# Merge all subjects' data into one file
if all_eeg_data:
    merged_data = np.concatenate(all_eeg_data, axis=1)
    np.save(os.path.join(output_folder, "merged_eeg_data.npy"), merged_data)
    print("✅ Merged all subjects' data and saved as merged_eeg_data.npy")
else:
    print("❌ No valid EEG data found to merge.")
 """
import os
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, spectrogram, welch

# Define input and output directories
input_folder = "C:/Users/deeks/OneDrive/Desktop/Research/21679035"
output_folder = "C:/Users/deeks/OneDrive/Desktop/Research/Preprocessed"
os.makedirs(output_folder, exist_ok=True)

# Bandpass filter function (8-30 Hz for motor imagery EEG)
def bandpass_filter(data, lowcut=8, highcut=30, fs=256, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

# Initialize list to store all subjects' imagery data
all_eeg_data = []

# Loop through all .mat files in the input folder
for file_name in sorted(os.listdir(input_folder)):
    if file_name.endswith(".mat"):
        file_path = os.path.join(input_folder, file_name)
        print(f"🔄 Processing: {file_name}")
        
        # Load .mat file
        mat_data = sio.loadmat(file_path)
        
        # Ensure 'eeg' key exists
        if 'eeg' not in mat_data:
            print(f"⚠️ Skipping {file_name}: 'eeg' key not found")
            continue
        
        eeg_data = mat_data['eeg']
        
        # Extract only motor imagery data
        if 'imagery_left' in eeg_data.dtype.names and 'imagery_right' in eeg_data.dtype.names:
            imagery_left = np.array(eeg_data['imagery_left'][()])
            imagery_right = np.array(eeg_data['imagery_right'][()])
            
            # Combine left and right imagery data
            combined_imagery = np.concatenate([imagery_left, imagery_right], axis=1)
            
            # Apply bandpass filter
            filtered_data = bandpass_filter(combined_imagery)
            
            # Compute spectrogram
            f, t, Sxx = spectrogram(filtered_data, fs=256, axis=1)
            
            # Compute Power Spectral Density (PSD)
            freqs, psd = welch(filtered_data, fs=256, axis=1)
            
            # Remove noise (mean subtraction)
            denoised_data = filtered_data - np.mean(filtered_data, axis=1, keepdims=True)
            
            # Save processed data
            output_path = os.path.join(output_folder, file_name.replace(".mat", "_processed.npy"))
            np.save(output_path, denoised_data)
            
            print(f"✅ Saved processed data to {output_path}")
            
            # Append to list for potential merging
            all_eeg_data.append(denoised_data)
        else:
            print(f"⚠️ Skipping {file_name}: 'imagery_left' or 'imagery_right' not found")

# Merge all subjects' data into one file
if all_eeg_data:
    merged_data = np.concatenate(all_eeg_data, axis=1)
    np.save(os.path.join(output_folder, "merged_eeg_data.npy"), merged_data)
    print("✅ Merged all subjects' data and saved as merged_eeg_data.npy")
else:
    print("❌ No valid EEG data found to merge.")
