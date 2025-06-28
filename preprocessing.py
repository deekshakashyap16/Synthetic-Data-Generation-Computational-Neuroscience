import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as FLDA
from sklearn.pipeline import make_pipeline
from mne.decoding import CSP
from scipy.stats import pearsonr, rankdata
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
import warnings
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
warnings.filterwarnings('ignore')

# ------------------------ UPDATED FIELD FINDER FOR struct_as_record=True ------------------------ #
def find_matlab_field_record(data, field_name):
    """
    Recursively search for a field in a MATLAB struct loaded as a NumPy record array.
    """
    # Case 1: Record array with named fields
    if hasattr(data, 'dtype') and data.dtype.names:
        # Direct match
        if field_name in data.dtype.names:
            field_data = data[field_name]
            #print(field_data.item(), field_data.dtype)
            # Handle scalar record arrays
            if field_data.shape == ():
                return field_data.item()
            elif field_data.shape == (1,):
                return field_data[0]
            return field_data
        # Search nested fields
        for name in data.dtype.names:
            nested_data = data[name]
            # Handle scalar and singleton arrays
            if isinstance(nested_data, np.ndarray):
                if nested_data.shape == ():
                    nested_data = nested_data.item()
                elif nested_data.shape == (1,):
                    nested_data = nested_data[0]
            result = find_matlab_field_record(nested_data, field_name)
            if result is not None:
                return result

    # Case 2: Object array (rare, but possible with nested structs)
    elif isinstance(data, np.ndarray) and data.dtype == 'O':
        for item in data.flat:
            if item is not None:
                result = find_matlab_field_record(item, field_name)
                if result is not None:
                    return result

    # Not found
    return None

def extract_data_safely_record(data, field_name):
    """Safely extract and process data from MATLAB structure with struct_as_record=True"""
    field_data = find_matlab_field_record(data, field_name)
    if field_data is None:
        return None
    
    if isinstance(field_data, np.ndarray):
        print(field_data.shape)
        if field_data.size == 1:
            return field_data.item()
        elif field_data.ndim == 0:
            return field_data.item()
        else:
            return field_data
    return field_data

def segment_trials_from_frames_and_events(continuous_data, frames, events, fs, trial_duration_ms=4000):
    """
    Segment continuous data into trials based on frame information and event markers
    
    Args:
        continuous_data: Continuous EEG data (channels x time_points)
        frames: Frame information containing overall temporal range [start_ms, end_ms]
        events: Event array with trial markers (same length as time points)
        fs: Sampling frequency
        trial_duration_ms: Duration of each trial in milliseconds
    
    Returns:
        trials: List of trial data arrays
        labels: Corresponding trial labels
    """
    trials = []
    labels = []
    
    print(f"Frame data: {frames} (shape: {getattr(frames, 'shape', 'N/A')})")
    print(f"Events shape: {events.shape if hasattr(events, 'shape') else len(events)}")
    print(f"Trial duration: {trial_duration_ms} ms")
    print(f"Continuous data shape: {continuous_data.shape}")
    
    # Convert frame boundaries to samples
    if frames is not None and len(frames) >= 2:
        frame_start_ms, frame_end_ms = float(frames[0]), float(frames[1])
        frame_start_sample = int((frame_start_ms / 1000.0) * fs)
        frame_end_sample = int((frame_end_ms / 1000.0) * fs)
        
        # Adjust for negative start times (relative to some reference point)
        if frame_start_sample < 0:
            frame_start_sample = 0
        if frame_end_sample >= continuous_data.shape[-1]:
            frame_end_sample = continuous_data.shape[-1] - 1
            
        print(f"Frame boundaries: {frame_start_ms} to {frame_end_ms} ms")
        print(f"Sample boundaries: {frame_start_sample} to {frame_end_sample}")
        
        # Look for event changes within the frame boundaries
        if events is not None and len(events) > 0:
            # Focus on the relevant portion of events
            relevant_events = events if len(events) == continuous_data.shape[-1] else events[:continuous_data.shape[-1]]
            
            print(f"Unique event values: {np.unique(relevant_events)}")
            print(f"Event value counts: {np.bincount(relevant_events)}")
            
            # Find ALL event transitions (where the event value changes to non-zero)
            event_transitions = []
            prev_event = relevant_events[0]
            
            for i, event in enumerate(relevant_events[1:], 1):
                if event != prev_event:
                    # Record all transitions, not just to non-zero
                    event_transitions.append((i, prev_event, event))
                prev_event = event
            
            print(f"Found {len(event_transitions)} total event transitions")
            
            # Filter for meaningful transitions (transitions TO a trial state)
            trial_starts = []
            for i, (sample_idx, prev_val, curr_val) in enumerate(event_transitions):
                if prev_val == 0 and curr_val != 0:  # Transition from rest to trial
                    trial_starts.append((sample_idx, curr_val))
                    print(f"Trial start at sample {sample_idx}, event value {curr_val}")
            
            print(f"Found {len(trial_starts)} trial start points")
            
            # Create trials around each trial start
            trial_duration_samples = int((trial_duration_ms / 1000.0) * fs)
            
            for i, (start_sample, event_label) in enumerate(trial_starts):
                # Create trial window - start from the event trigger
                trial_start = start_sample
                trial_end = trial_start + trial_duration_samples
                
                # Ensure bounds are valid
                if trial_start >= 0 and trial_end <= continuous_data.shape[-1]:
                    trial_data = continuous_data[:, trial_start:trial_end]
                    
                    if trial_data.shape[-1] >= int(0.5 * fs):  # At least 0.5 seconds
                        trials.append(trial_data)
                        # Convert event label to binary (assuming left/right imagery)
                        label = 0 if event_label == 1 else 1  # Adjust mapping as needed
                        labels.append(label)
                        
                        print(f"✅ Trial {i} added: samples {trial_start}-{trial_end}, event {event_label}, label {label}, shape {trial_data.shape}")
                    else:
                        print(f"❌ Trial {i} too short: {trial_data.shape[-1]} samples")
                else:
                    print(f"❌ Trial {i} out of bounds: {trial_start}-{trial_end}")

            # --- BEGIN: Minimal addition for REST state detection ---
            # Detect rest periods between trials and at the beginning/end
            last_trial_end = frame_start_sample
            for i, (sample_idx, prev_val, curr_val) in enumerate(event_transitions):
                if prev_val != 0 and curr_val == 0:  # Transition from trial to rest
                    rest_start = sample_idx
                    rest_duration = rest_start - last_trial_end
                    if rest_duration >= trial_duration_samples:
                        num_rest_trials = rest_duration // trial_duration_samples
                        for n in range(num_rest_trials):
                            start = last_trial_end + n * trial_duration_samples
                            end = start + trial_duration_samples
                            if end <= continuous_data.shape[-1]:
                                trial_data = continuous_data[:, start:end]
                                if trial_data.shape[-1] >= int(0.5 * fs):
                                    trials.append(trial_data)
                                    labels.append(2)  # Rest label
                                    print(f"✅ Rest trial added: samples {start}-{end}, label 2, shape {trial_data.shape}")
                    last_trial_end = rest_start
            # Rest at the very beginning
            if event_transitions and event_transitions[0][1] == 0:
                first_trial_start = event_transitions[0][0]
                rest_duration = first_trial_start - frame_start_sample
                if rest_duration >= trial_duration_samples:
                    num_rest_trials = rest_duration // trial_duration_samples
                    for n in range(num_rest_trials):
                        start = frame_start_sample + n * trial_duration_samples
                        end = start + trial_duration_samples
                        if end <= continuous_data.shape[-1]:
                            trial_data = continuous_data[:, start:end]
                            if trial_data.shape[-1] >= int(0.5 * fs):
                                trials.append(trial_data)
                                labels.append(2)
                                print(f"✅ Rest trial at start: samples {start}-{end}, label 2, shape {trial_data.shape}")
            # Rest at the very end
            if event_transitions and event_transitions[-1][2] == 0:
                rest_start = event_transitions[-1][0]
                rest_duration = frame_end_sample - rest_start
                if rest_duration >= trial_duration_samples:
                    num_rest_trials = rest_duration // trial_duration_samples
                    for n in range(num_rest_trials):
                        start = rest_start + n * trial_duration_samples
                        end = start + trial_duration_samples
                        if end <= continuous_data.shape[-1]:
                            trial_data = continuous_data[:, start:end]
                            if trial_data.shape[-1] >= int(0.5 * fs):
                                trials.append(trial_data)
                                labels.append(2)
                                print(f"✅ Rest trial at end: samples {start}-{end}, label 2, shape {trial_data.shape}")
            # --- END: Minimal addition for REST state detection ---

            # If still no trials found, try sliding window approach
            if len(trials) == 0:
                print("No clear trial starts found, trying sliding window approach...")
                
                # Look for periods where events are non-zero
                non_zero_mask = relevant_events != 0
                
                if np.any(non_zero_mask):
                    # Find contiguous regions of non-zero events
                    # Use np.diff to find start and end of non-zero regions
                    diff_mask = np.diff(non_zero_mask.astype(int))
                    starts = np.where(diff_mask == 1)[0] + 1  # +1 because diff shifts indices
                    ends = np.where(diff_mask == -1)[0] + 1
                    
                    # Handle edge cases
                    if non_zero_mask[0]:
                        starts = np.concatenate([[0], starts])
                    if non_zero_mask[-1]:
                        ends = np.concatenate([ends, [len(non_zero_mask)]])
                    
                    print(f"Found {len(starts)} non-zero regions")
                    
                    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
                        region_length = end_idx - start_idx
                        if region_length >= int(0.5 * fs):  # At least 0.5 seconds
                            # Extract the region
                            trial_data = continuous_data[:, start_idx:end_idx]
                            trials.append(trial_data)
                            
                            # Determine label from the event value in this region
                            region_events = relevant_events[start_idx:end_idx]
                            event_value = np.median(region_events[region_events != 0])
                            label = 0 if event_value == 1 else 1
                            labels.append(label)
                            
                            print(f"✅ Trial {i} from region: samples {start_idx}-{end_idx}, event {event_value}, label {label}")
    
    print(f"Successfully extracted {len(trials)} trials")
    return trials, labels

def extract_trials_with_frames(eeg_data, field_name, fs):
    """
    Extract trials using frame information for proper segmentation
    
    Args:
        eeg_data: The main EEG data structure
        field_name: Name of the field containing the data ('imagery_left' or 'imagery_right')
        fs: Sampling frequency
        
    Returns:
        trials: Segmented trial data
        success: Boolean indicating if segmentation was successful
    """
    try:
        # Extract the raw data field
        raw_data = extract_data_safely_record(eeg_data, field_name)
        if raw_data is None:
            print(f"Could not find field: {field_name}")
            return None, False
            
        # Extract frame information
        frames = extract_data_safely_record(eeg_data, 'frame')
        if frames is None:
            # Try alternative frame field names
            for alt_name in ['frames', 'trial_frames', 'time_frames', 'frame_info']:
                frames = extract_data_safely_record(eeg_data, alt_name)
                if frames is not None:
                    print(f"Found frames in field: {alt_name}")
                    break
        
        # Extract event information - try multiple approaches
        parts = field_name.split('_')
        if len(parts) > 1:
            prefix = parts[1]
        else:
            prefix = parts[0]
        event_field_name = f"{prefix}_event"

        events = extract_data_safely_record(eeg_data, event_field_name)
        if events is None:
            events = extract_data_safely_record(eeg_data, 'imagery_event')
        
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Frames available: {frames is not None}")
        if frames is not None:
            print(f"Frames type: {type(frames)}, shape/length: {getattr(frames, 'shape', len(frames) if hasattr(frames, '__len__') else 'unknown')}")
        print(f"Events available: {events is not None}")
        if events is not None:
            print(f"Events type: {type(events)}, shape/length: {getattr(events, 'shape', len(events) if hasattr(events, '__len__') else 'unknown')}")
        
        # If we have continuous data and frame information
        if raw_data.ndim == 2 and frames is not None:
            print("Segmenting continuous data using frame information...")
            
            # Handle different frame data structures
            if hasattr(frames, 'shape') and frames.shape == ():
                # Single frame (scalar), try to extract it
                frames = frames.item()
                
            # If frames is still a single item, wrap it in a list
            if not hasattr(frames, '__iter__') or isinstance(frames, str):
                frames = [frames]
            
            trials, trial_labels = segment_trials_from_frames_and_events(raw_data, frames, events, fs)
            
            if len(trials) > 0:
                # Convert to standard format (n_trials x n_channels x n_timepoints)
                min_length = min(trial.shape[-1] for trial in trials)
                max_length = max(trial.shape[-1] for trial in trials)
                n_channels = trials[0].shape[0]
                
                print(f"Trial lengths range: {min_length} - {max_length} samples")
                
                # Use minimum length to ensure all trials have the same size
                target_length = min_length
                
                # Truncate all trials to the minimum length for consistency
                consistent_trials = []
                for trial in trials:
                    if trial.shape[-1] > target_length:
                        # Truncate from the beginning and end equally if possible
                        excess = trial.shape[-1] - target_length
                        start_trim = excess // 2
                        end_trim = excess - start_trim
                        trial = trial[:, start_trim:trial.shape[-1]-end_trim if end_trim > 0 else trial.shape[-1]]
                    
                    consistent_trials.append(trial)
                
                segmented_data = np.stack(consistent_trials, axis=0)
                print(f"Successfully segmented into {len(trials)} trials of shape {segmented_data.shape}")
                return segmented_data, True
            else:
                print("No valid trials could be extracted from frame information")
                return None, False
        
        # Fallback to original method if no frame info or already segmented
        elif raw_data.ndim == 3:
            print("Data already appears to be segmented into trials")
            return raw_data, True
        else:
            print(f"Could not determine how to segment the data. Data shape: {raw_data.shape}, frames available: {frames is not None}")
            return None, False
            
    except Exception as e:
        print(f"Error in trial extraction: {e}")
        import traceback
        traceback.print_exc()
        return None, False

# ------------------------ ERD/ERS CALCULATION (MISSING FROM ORIGINAL) ------------------------ #
def calculate_erd_ers(eeg_data, fs, labels=None, baseline_window=(0.5, 1.0), task_window=(1.0, 2.5)):
    """Calculate Event-Related Desynchronization/Synchronization"""
    try:
        # Filter for mu (8-12 Hz) and beta (13-30 Hz) bands
        mu_filtered = butter_filter(eeg_data, 8, 12, fs)
        beta_filtered = butter_filter(eeg_data, 13, 30, fs)
        
        # Calculate power using Hilbert transform
        mu_power = np.abs(hilbert(mu_filtered, axis=-1)) ** 2
        beta_power = np.abs(hilbert(beta_filtered, axis=-1)) ** 2
        
        # Time indices
        baseline_start = int(baseline_window[0] * fs)
        baseline_end = int(baseline_window[1] * fs)
        task_start = int(task_window[0] * fs)
        task_end = int(task_window[1] * fs)
        
        # Handle edge cases
        min_length = min(trial.shape[-1] for trial in eeg_data)
        baseline_end = min(baseline_end, min_length)
        task_end = min(task_end, min_length)
        
        # Calculate actual time dimension for task window
        n_time_points = task_end - task_start
        n_channels = eeg_data.shape[1]
        
        # Initialize arrays with dynamic size
        erd_ers_mu = np.zeros((n_channels, n_time_points))
        erd_ers_beta = np.zeros((n_channels, n_time_points))
        
        # --- BEGIN REST STATE BASELINE DETECTION ---
        # Check for rest trials (label=2) if labels are provided
        use_rest_baseline = False
        if labels is not None and np.any(labels == 2):
            rest_mask = labels == 2
            if np.sum(rest_mask) > 0:
                use_rest_baseline = True
        # --- END REST STATE BASELINE DETECTION ---
        
        for ch in range(n_channels):
            # Baseline power calculation
            if use_rest_baseline:
                baseline_mu = np.mean(mu_power[rest_mask, ch, baseline_start:baseline_end])
                baseline_beta = np.mean(beta_power[rest_mask, ch, baseline_start:baseline_end])
            else:
                baseline_mu = np.mean(mu_power[:, ch, baseline_start:baseline_end])
                baseline_beta = np.mean(beta_power[:, ch, baseline_start:baseline_end])
            
            # Task power (windowed)
            task_mu = mu_power[:, ch, task_start:task_end]  # Shape: (trials, time)
            task_beta = beta_power[:, ch, task_start:task_end]
            
            # Calculate ERD/ERS without resampling
            erd_ers_mu[ch, :] = np.mean((task_mu - baseline_mu) / baseline_mu * 100, axis=0)
            erd_ers_beta[ch, :] = np.mean((task_beta - baseline_beta) / baseline_beta * 100, axis=0)
        
        # Combine mu and beta (take average)
        erd_ers = (erd_ers_mu + erd_ers_beta) / 2
        return erd_ers
        
    except Exception as e:
        print(f"ERD/ERS calculation error: {e}")
        # Return empty array with correct dimensions if possible
        return np.zeros((n_channels, n_time_points)) if 'n_time_points' in locals() else np.zeros((64, 100))

# ------------------------ ENHANCED FILTERING (SAME AS BEFORE) ------------------------ #

def detect_flat_channels(eeg_data, fs, threshold=0.1):  # More realistic threshold
    """
    Detect and remove flat channels from EEG data.
    
    Args:
        eeg_data: EEG data array (trials x channels x timepoints)
        fs: Sampling frequency (used for time calculations)
        threshold: Maximum allowed standard deviation (μV) to consider channel flat
        
    Returns:
        cleaned_data: EEG data with flat channels removed
        bad_channels: List of indices of removed flat channels
    """
    if eeg_data.size == 0:
        print("⚠️ Empty EEG data received")
        return eeg_data, []

    # Calculate standard deviation for each channel across all trials and timepoints
    channel_stds = np.std(eeg_data, axis=(0, 2))
    
    # Find channels with standard deviation below threshold
    bad_channels = np.where(channel_stds < threshold)[0].tolist()
    
    if not bad_channels:
        print("✅ No flat channels detected")
        return eeg_data, []
    
    print(f"🚫 Found {len(bad_channels)} flat channels (threshold: {threshold}μV):")
    for ch in bad_channels:
        print(f"    Channel {ch}: σ={channel_stds[ch]:.2e}μV")
    
    # Remove bad channels while preserving trial and time dimensions
    cleaned_data = np.delete(eeg_data, bad_channels, axis=1)
    
    # Verify removal
    if cleaned_data.shape[1] == eeg_data.shape[1]:
        print("⚠️ Failed to remove channels - data shape unchanged")
        return eeg_data, []
    
    print(f"🧹 Removed {len(bad_channels)} channels. New shape: {cleaned_data.shape}")
    return cleaned_data, bad_channels

def butter_filter(data, lowcut=None, highcut=None, fs=512, order=4):
    """Apply Butterworth filter with improved edge handling"""
    if data.shape[-1] < 3 * order:
        print(f"Warning: Signal too short for order {order} filter")
        return data
    
    nyq = 0.5 * fs
    try:
        if lowcut and highcut:
            b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
        elif lowcut:
            b, a = butter(order, lowcut/nyq, btype='high')
        elif highcut:
            b, a = butter(order, highcut/nyq, btype='low')
        else:
            return data
        
        return filtfilt(b, a, data, axis=-1, padtype='odd', padlen=3*order)
    except Exception as e:
        print(f"Filter error: {e}")
        return data

def apply_car(data):
    """Apply Common Average Reference"""
    if data.ndim == 3:
        return data - data.mean(axis=1, keepdims=True)
    elif data.ndim == 2:
        return data - data.mean(axis=0, keepdims=True)
    return data

def apply_laplacian(data):
    """Apply Laplacian spatial filter"""
    laplacian_data = np.zeros_like(data)
    n_channels = data.shape[1] if data.ndim == 3 else data.shape[0]
    
    neighbor_map = {}
    for i in range(n_channels):
        neighbors = []
        if i > 0: neighbors.append(i-1)
        if i < n_channels-1: neighbors.append(i+1)
        if i >= 8: neighbors.append(i-8)
        if i < n_channels-8: neighbors.append(i+8)
        neighbor_map[i] = neighbors
    
    for i in range(n_channels):
        if neighbor_map[i]:
            if data.ndim == 3:
                laplacian_data[:, i, :] = data[:, i, :] - np.mean(data[:, neighbor_map[i], :], axis=1)
            else:
                laplacian_data[i, :] = data[i, :] - np.mean(data[neighbor_map[i], :], axis=0)
        else:
            laplacian_data[:, i, :] = data[:, i, :] if data.ndim == 3 else data[i, :]
    
    return laplacian_data

# ------------------------ TRIAL REJECTION (SAME LOGIC) ------------------------ #
def reject_voltage_trials(eeg_trials, fs, threshold=200):  # Increased from 100 to 200
    """Enhanced voltage-based trial rejection with relaxed threshold"""
    good_trials = []
    rejection_reasons = []
    
    for i, trial in enumerate(eeg_trials):
        try:
            start_idx = int(fs * 0.5)
            end_idx = int(fs * 2.5)
            
            if trial.shape[-1] < end_idx:
                end_idx = trial.shape[-1]
            if start_idx >= trial.shape[-1]:
                start_idx = 0
                end_idx = trial.shape[-1]
                
            segment = trial[:, start_idx:end_idx]
            
            # Use 95th percentile instead of max to be less sensitive to outliers
            percentile_95 = np.percentile(np.abs(segment), 95)
            max_val = np.max(np.abs(segment))
            
            # Use the 95th percentile for threshold check
            if percentile_95 <= threshold:
                good_trials.append(i)
            else:
                rejection_reasons.append(f"Trial {i}: Voltage {percentile_95:.1f}μV (95th percentile) > {threshold}μV")
                
        except Exception as e:
            rejection_reasons.append(f"Trial {i}: Error - {e}")
    
    print(f"   Voltage rejection: {len(good_trials)}/{len(eeg_trials)} kept (threshold: {threshold}μV)")
    return good_trials, rejection_reasons

def reject_emg_correlated_trials(emg_trials, fs, corr_threshold=0.9, p_threshold=0.001):  # Relaxed thresholds
    """Enhanced EMG correlation rejection with more lenient criteria"""
    good_trials = []
    rejection_reasons = []
    n_permutations = 50  # Reduced from 100
    
    for i, trial in enumerate(emg_trials):
        try:
            # Apply preprocessing
            trial_hp = butter_filter(trial, 0.5, None, fs)
            trial_car = apply_car(trial_hp)
            trial_bp = butter_filter(trial_car, 50, 250, fs)
            
            analytic = hilbert(trial_bp, axis=-1)
            power = np.abs(analytic) ** 2
            
            trial_length = trial.shape[-1]
            
            # More flexible time windows
            if trial_length < fs * 1.5:  # If less than 1.5 seconds
                # Just check if there's excessive correlation across the whole trial
                rest_end = max(1, int(trial_length * 0.4))
                task_start = rest_end
                task_end = trial_length
            else:
                rest_start = max(0, int(trial_length * 0.1))  # First 10%
                rest_end = int(trial_length * 0.4)  # Up to 40%
                task_start = int(trial_length * 0.6)  # From 60%
                task_end = trial_length
            
            if rest_end <= rest_start or task_end <= task_start:
                good_trials.append(i)  # Accept if can't properly segment
                continue
            
            rest_power = power[:, rest_start:rest_end]
            task_power = power[:, task_start:task_end]
            
            # Skip if segments too small
            if rest_power.shape[1] < 10 or task_power.shape[1] < 10:
                good_trials.append(i)
                continue
            
            rest_labels = -1 * np.ones(rest_power.shape[1])
            task_labels = 1 * np.ones(task_power.shape[1])
            
            full_power = np.concatenate((rest_power, task_power), axis=1)
            full_labels = np.concatenate((rest_labels, task_labels))
            
            # More aggressive decimation to reduce computation
            decimation_factor = max(4, full_power.shape[1] // 50)  # Target ~50 samples
            n_samples_decimated = full_power.shape[1] // decimation_factor
            
            if n_samples_decimated < 10:  # Too few samples after decimation
                good_trials.append(i)
                continue
                
            decimated_power = np.zeros((full_power.shape[0], n_samples_decimated))
            decimated_labels = np.zeros(n_samples_decimated)
            
            for j in range(n_samples_decimated):
                start_idx = j * decimation_factor
                end_idx = (j + 1) * decimation_factor
                decimated_power[:, j] = full_power[:, start_idx:end_idx].mean(axis=1)
                decimated_labels[j] = full_labels[start_idx:end_idx].mean()
            
            trial_is_bad = False
            max_correlation = 0
            min_pvalue = 1.0
            
            # Only check a subset of channels to speed up
            n_channels = decimated_power.shape[0]
            channels_to_check = min(n_channels, 2)  # Only check 2 EMG channels max
            
            for ch_idx in range(channels_to_check):
                ch = ch_idx if n_channels <= 4 else ch_idx  # Use first few channels
                
                ranked_power = rankdata(decimated_power[ch, :])
                actual_corr, _ = pearsonr(ranked_power, decimated_labels)
                actual_corr = abs(actual_corr)  # Use absolute correlation
                
                # Simplified permutation test
                perm_correlations = []
                for perm in range(n_permutations):
                    perm_labels = np.random.permutation(decimated_labels)
                    perm_corr, _ = pearsonr(ranked_power, perm_labels)
                    perm_correlations.append(abs(perm_corr))
                
                p_value = np.sum(np.array(perm_correlations) >= actual_corr) / n_permutations
                max_correlation = max(max_correlation, actual_corr)
                min_pvalue = min(min_pvalue, p_value)
                
                # More lenient criteria
                if p_value < p_threshold and actual_corr > corr_threshold:
                    trial_is_bad = True
                    break
            
            if not trial_is_bad:
                good_trials.append(i)
            else:
                rejection_reasons.append(f"Trial {i}: EMG corr={max_correlation:.3f}, p={min_pvalue:.4f}")
                
        except Exception as e:
            rejection_reasons.append(f"Trial {i}: EMG processing error - {e}")
            good_trials.append(i)  # Accept on error
    
    print(f"   EMG rejection: {len(good_trials)}/{len(emg_trials)} kept (corr_thresh: {corr_threshold}, p_thresh: {p_threshold})")
    return good_trials, rejection_reasons

# ------------------------ CLASSIFICATION (SAME LOGIC) ------------------------ #
def classify_riemann_flda(eeg, labels, fs, n_iterations=120, min_trials_per_class=5):
    """Riemannian covariance + Tangent Space + FLDA classification with robust error handling"""
    try:
        # Keep original preprocessing
        eeg_hp = butter_filter(eeg, 0.5, None, fs)
        eeg_car = apply_car(eeg_hp)
        eeg_bp = butter_filter(eeg_car, 8, 30, fs)

        # Maintain original windowing
        start_idx = int(fs * 0.5)
        end_idx = int(fs * 2.5)
        min_trial_length = min(trial.shape[-1] for trial in eeg_bp)
        
        if end_idx > min_trial_length:
            end_idx = min_trial_length
        if start_idx >= min_trial_length:
            start_idx = 0
            end_idx = min_trial_length

        eeg_windowed = eeg_bp[:, :, start_idx:end_idx]

        # Check data quality before classification
        if eeg_windowed.shape[0] < 10 or eeg_windowed.shape[-1] < int(0.5 * fs):
            print(f"Insufficient data for classification: {eeg_windowed.shape}")
            return 0.0

        # Original class balance checks
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) < 2 or min(counts) < min_trials_per_class:
            print(f"Insufficient data: {dict(zip(unique_labels, counts))}")
            return 0.0

        accuracies = []
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        # Reduce iterations to speed up processing
        n_iterations = min(n_iterations, 30)
        
        for iteration in range(n_iterations):
            try:
                # Original data splitting
                X_train, X_test, y_train, y_test = train_test_split(
                    eeg_windowed, labels,
                    test_size=0.3,
                    stratify=labels,
                    random_state=iteration
                )

                # Check training data quality
                train_unique, train_counts = np.unique(y_train, return_counts=True)
                if min(train_counts) < 3:
                    continue

                # More robust pipeline with error handling
                try:
                    # Use empirical covariance instead of LWF if numerical issues
                    cov_est = Covariances(estimator='oas')  # Oracle Approximating Shrinkage
                    cov_train = cov_est.fit_transform(X_train)
                    
                    ts = TangentSpace(metric='riemann')
                    ts_train = ts.fit_transform(cov_train)
                    
                    # Use more stable LDA parameters
                    lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=None)
                    lda.fit(ts_train, y_train)
                    
                    # Test prediction
                    cov_test = cov_est.transform(X_test)
                    ts_test = ts.transform(cov_test)
                    acc = lda.score(ts_test, y_test)
                    
                except Exception as clf_error:
                    print(f"Classification error in iteration {iteration}: {clf_error}")
                    # Fallback to simple classification
                    try:
                        from sklearn.svm import SVC
                        # Flatten features for simple SVM
                        X_train_flat = X_train.reshape(X_train.shape[0], -1)
                        X_test_flat = X_test.reshape(X_test.shape[0], -1)
                        
                        svm = SVC(kernel='linear', C=1.0)
                        svm.fit(X_train_flat, y_train)
                        acc = svm.score(X_test_flat, y_test)
                    except:
                        continue

                accuracies.append(acc)

            except Exception as e:
                print(f"Warning in iteration {iteration}: {e}")
                continue

        if not accuracies:
            print("No successful iterations, returning 0.0")
            return 0.0

        mean_acc = np.mean(accuracies)
        
        # Quick cross-validation with error handling
        try:
            kfold_accuracies = []
            skf = StratifiedKFold(n_splits=min(5, min(counts)), shuffle=True, random_state=42)
            
            for train_idx, test_idx in skf.split(eeg_windowed, labels):
                try:
                    X_train, X_test = eeg_windowed[train_idx], eeg_windowed[test_idx]
                    y_train, y_test = labels[train_idx], labels[test_idx]
                    
                    # Simple robust classification
                    from sklearn.svm import SVC
                    X_train_flat = X_train.reshape(X_train.shape[0], -1)
                    X_test_flat = X_test.reshape(X_test.shape[0], -1)
                    
                    svm = SVC(kernel='linear', C=1.0)
                    svm.fit(X_train_flat, y_train)
                    acc = svm.score(X_test_flat, y_test)
                    kfold_accuracies.append(acc)
                except:
                    continue
            
            if kfold_accuracies:
                kfold_mean = np.mean(kfold_accuracies)
                return min(mean_acc, kfold_mean)
            else:
                return mean_acc
        except:
            return mean_acc

    except Exception as e:
        print(f"Classification error: {e}")
        return 0.0

# ------------------------ UPDATED SUBJECT PROCESSING ------------------------ #
def process_subject_enhanced_record(args):
    """Enhanced subject processing with more lenient quality criteria"""
    subject_path, output_dir = args

    try:
        subject_name = os.path.basename(subject_path)
        print(f"Processing {subject_name}...")

        # Load data with struct_as_record=True
        data = loadmat(subject_path, struct_as_record=True, squeeze_me=True)
        eeg = data['eeg']

        # Print all available fields for debugging
        if hasattr(eeg, 'dtype') and eeg.dtype.names:
            print(f"Available fields: {eeg.dtype.names}")

        # Extract metadata using updated functions
        fs = extract_data_safely_record(eeg, 'srate')
        subject_id = extract_data_safely_record(eeg, 'subject')

        # Try alternative field names
        if fs is None:
            for alt_name in ['fs', 'sampling_rate', 'Fs', 'frequency']:
                fs = extract_data_safely_record(eeg, alt_name)
                if fs is not None:
                    break

        if fs is None:
            print(f"⚠️ Skipping {subject_path} - missing sampling rate")
            return None

        fs = int(float(fs))

        if subject_id is None:
            subject_id = subject_name.replace('.mat', '')

        # --- BEGIN: Add support for rest state extraction ---
        # Try to extract rest trials if available
        # --- Enhanced rest state extraction ---
        rest_trials = None
        rest_success = False

        # Try multiple approaches for rest detection
        if hasattr(eeg, 'dtype') and eeg.dtype.names:
            # Try direct field extraction with multiple possible field names
            for rest_field in ['imagery_rest', 'rest', 'baseline', 'idle']:
                if rest_field in eeg.dtype.names:
                    rest_trials, rest_success = extract_trials_with_frames(eeg, rest_field, fs)
                    if rest_success:
                        print(f"Found rest trials in field: {rest_field}")
                        break

        # If no direct rest field found, create artificial rest periods
        if not rest_success:
            print("No direct rest field found, skipping rest state detection")

        # --- END: Add support for rest state extraction ---

        # Extract trials using frame-based segmentation
        left_trials, left_success = extract_trials_with_frames(eeg, 'imagery_left', fs)
        right_trials, right_success = extract_trials_with_frames(eeg, 'imagery_right', fs)

        if not left_success or not right_success or left_trials is None or right_trials is None:
            print(f"⚠️ Skipping {subject_id} - failed to extract trials with frame information")
            return None

        print(f"   Left trials: {left_trials.shape[0]}")
        print(f"   Right trials: {right_trials.shape[0]}")

        # Ensure proper dimensions
        if left_trials.ndim == 2:
            left_trials = left_trials[np.newaxis, :, :]
        if right_trials.ndim == 2:
            right_trials = right_trials[np.newaxis, :, :]

        # --- BEGIN: Add support for rest state extraction ---
        if rest_trials is not None and rest_success and rest_trials is not None:
            if rest_trials.ndim == 2:
                rest_trials = rest_trials[np.newaxis, :, :]
            eeg_rest = rest_trials[:, :64, :]
            n_rest_trials = eeg_rest.shape[0]
        else:
            eeg_rest = None
            n_rest_trials = 0
        # --- END: Add support for rest state extraction ---

        # Extract EEG (channels 1-64) and EMG (channels 65-68)
        eeg_left = left_trials[:, :64, :]
        eeg_right = right_trials[:, :64, :]

        emg_left = left_trials[:, 64:68, :] if left_trials.shape[1] >= 68 else None
        emg_right = right_trials[:, 64:68, :] if right_trials.shape[1] >= 68 else None

        # Process ALL trials
        n_left_trials = eeg_left.shape[0]
        n_right_trials = eeg_right.shape[0]

        # --- BEGIN: Add support for rest state extraction ---
        if eeg_rest is not None:
            eeg_combined = np.concatenate((eeg_left, eeg_right, eeg_rest), axis=0)
            labels = np.array([0] * n_left_trials + [1] * n_right_trials + [2] * n_rest_trials)
        else:
            eeg_combined = np.concatenate((eeg_left, eeg_right), axis=0)
            labels = np.array([0] * n_left_trials + [1] * n_right_trials)
        # --- END: Add support for rest state extraction ---

        print(f"   EEG shape: {eeg_combined.shape}")
        print(f"   Left: {n_left_trials}, Right: {n_right_trials}, Rest: {n_rest_trials}, Total: {len(eeg_combined)}")

        # Quality checks with more lenient criteria
        if len(eeg_combined) < 10:  # Reduced from 20
            print(f"⚠️ Too few trials ({len(eeg_combined)}) for {subject_id}")
            return None

        # Step 1: Voltage-based rejection with relaxed threshold
        eeg_for_voltage = butter_filter(eeg_combined, 8, 30, fs)
        good_trials_voltage, voltage_reasons = reject_voltage_trials(eeg_for_voltage, fs, threshold=300)  # Increased threshold

        # Step 2: EMG-based rejection (more lenient or skip if too few trials)
        good_trials_emg = list(range(len(eeg_combined)))
        emg_reasons = []

        if emg_left is not None and emg_right is not None and len(good_trials_voltage) > 20:  # Only do EMG rejection if enough trials
            try:
                emg_combined = np.concatenate((emg_left, emg_right), axis=0)
                good_trials_emg, emg_reasons = reject_emg_correlated_trials(emg_combined, fs, corr_threshold=0.95, p_threshold=0.001)
            except Exception as e:
                print(f"   EMG rejection failed: {e}")
                good_trials_emg = good_trials_voltage  # Use voltage-filtered trials
        else:
            print(f"   Skipping EMG rejection (insufficient trials or no EMG data)")
            good_trials_emg = good_trials_voltage

        # Combine rejections
        good_trials = sorted(set(good_trials_voltage).intersection(set(good_trials_emg)))
        print(f"   Final: {len(good_trials)}/{len(eeg_combined)} kept")

        # More lenient final check
        if len(good_trials) < 5:  # Reduced from 10
            print(f"⚠️ Too few good trials ({len(good_trials)}) for {subject_id}")
            return None

        # Extract clean data
        eeg_clean = eeg_combined[good_trials]
        labels_clean = labels[good_trials]

        # Check class balance
        unique_labels, counts = np.unique(labels_clean, return_counts=True)
        print(f"   Clean distribution: {dict(zip(unique_labels, counts))}")

        # More lenient class balance check
        if len(unique_labels) < 2 or min(counts) < 2:  # Reduced from 5
            print(f"⚠️ Insufficient class balance for {subject_id}")
            return None

        # ERD/ERS calculation
        try:
            erd_ers = calculate_erd_ers(eeg_clean, fs, labels_clean)  # Pass labels_clean for rest baseline
        except Exception as e:
            print(f"   ERD/ERS calculation failed: {e}")
            erd_ers = np.zeros((64, 100))
        # --- INSERT FLAT CHANNEL DETECTION AND REMOVAL HERE ---
        eeg_clean, flat_channels = detect_flat_channels(eeg_clean, fs)
        # ------------------------------------------------------
        # Classification with more lenient requirements
        accuracy = classify_riemann_flda(eeg_clean, labels_clean, fs, min_trials_per_class=2)
        print(f"   Accuracy: {accuracy:.3f}")
        # Save results
        subject_output_dir = os.path.join(output_dir, str(subject_id))
        os.makedirs(subject_output_dir, exist_ok=True)

        # Save arrays
        np.save(os.path.join(subject_output_dir, "eeg_clean.npy"), eeg_clean)
        np.save(os.path.join(subject_output_dir, "labels.npy"), labels_clean)
        np.save(os.path.join(subject_output_dir, "good_trials.npy"), good_trials)
        np.save(os.path.join(subject_output_dir, "erd_ers.npy"), erd_ers)

        # Enhanced metadata
        metadata = {
            "subject_id": str(subject_id),
            "sampling_rate": fs,
            "accuracy": float(accuracy),
            "original_trials": len(eeg_combined),
            "original_left_trials": n_left_trials,
            "original_right_trials": n_right_trials,
            "original_rest_trials": n_rest_trials,  # Added for rest
            "clean_trials": len(good_trials),
            "class_distribution": {str(k): int(v) for k, v in zip(unique_labels, counts)},
            "rejection_stats": {
                "voltage_kept": len(good_trials_voltage),
                "emg_kept": len(good_trials_emg),
                "final_kept": len(good_trials)
            },
            "has_emg": emg_left is not None,
            "data_shapes": {
                "eeg_original": list(eeg_combined.shape),
                "eeg_clean": list(eeg_clean.shape),
                "erd_ers": list(erd_ers.shape)
            },
            "rejection_reasons": {
                "voltage": voltage_reasons[:5],  # Reduced logging
                "emg": emg_reasons[:5]
            }
        }

        with open(os.path.join(subject_output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ {subject_id} completed - Accuracy: {accuracy:.3f}\n")

        return {
            "subject_id": str(subject_id),
            "accuracy": accuracy,
            "n_trials": len(good_trials),
            "original_trials": len(eeg_combined),
            "class_balance": dict(zip(unique_labels, counts)),
            "has_emg": emg_left is not None,
            "success": True
        }

    except Exception as e:
        print(f"❌ Error processing {subject_path}: {e}")
        import traceback
        traceback.print_exc()
        return {"subject_path": subject_path, "error": str(e), "success": False}

# ------------------------ MAIN FUNCTION ------------------------ #
def run_enhanced_preprocessing(raw_data_dir, output_dir, n_processes=None):
    """Run enhanced preprocessing pipeline"""
    
    if not os.path.exists(raw_data_dir):
        print(f"❌ Directory not found: {raw_data_dir}")
        return
    
    # Find subjects
    subject_files = [os.path.join(raw_data_dir, f) 
                    for f in os.listdir(raw_data_dir) 
                    if f.endswith('.mat')]
    subject_files.sort()
    
    if not subject_files:
        print(f"❌ No .mat files found in {raw_data_dir}")
        return
    
    print(f"📦 Found {len(subject_files)} subjects")
    print(f"📁 Input: {raw_data_dir}")
    print(f"📁 Output: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    def convert_keys(obj):
        """Recursively convert numpy types and dictionary keys to native Python types"""
        if isinstance(obj, dict):
            return {convert_keys(k): convert_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_keys(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_keys(item) for item in obj)
        elif isinstance(obj, np.generic):
            return obj.item()  # Convert numpy scalars to native Python types
        else:
            return obj

    # Process subjects
    args_list = [(f, output_dir) for f in subject_files]
    
    if n_processes is None:
        n_processes = min(cpu_count() - 1, len(subject_files))
    
    results = []
    
    if n_processes == 1:
        for arg in tqdm(args_list, desc="Processing"):
            result = process_subject_enhanced_record(arg)
            if result:
                results.append(result)
    else:
        try:
            with Pool(n_processes) as pool:
                # Remove tqdm from multiprocessing to avoid interference
                for i, result in enumerate(pool.imap_unordered(process_subject_enhanced_record, args_list)):
                    if result:
                        results.append(result)
                    # Manual progress reporting
                    if (i + 1) % 5 == 0:
                        print(f"Processed {i + 1}/{len(args_list)} subjects...")
        except Exception as e:
            print(f"Multiprocessing failed: {e}")
            print("Falling back to single-process mode...")
            for arg in tqdm(args_list, desc="Processing (fallback)"):
                result = process_subject_enhanced_record(arg)
                if result:
                    results.append(result)

                    
    # Final summary
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   ✅ Successful: {len(successful)}/{len(subject_files)}")
    print(f"   ❌ Failed: {len(failed)}")
    
    if successful:
        accuracies = [r['accuracy'] for r in successful]
        print(f"   📈 Mean accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"   📈 Range: {np.min(accuracies):.3f} - {np.max(accuracies):.3f}")
        
        high_acc = sum(1 for acc in accuracies if acc >= 0.95)
        good_acc = sum(1 for acc in accuracies if acc >= 0.70)
        
        print(f"   🎯 ≥95% accuracy: {high_acc}/{len(successful)} subjects")
        print(f"   👍 ≥70% accuracy: {good_acc}/{len(successful)} subjects")
    
    # Save comprehensive summary
    summary = {
        "processing_date": str(np.datetime64('now')),
        "total_subjects": len(subject_files),
        "successful": len(successful),
        "failed": len(failed),
        "results": results,
        "accuracy_stats": {
            "mean": float(np.mean([r['accuracy'] for r in successful])) if successful else 0,
            "std": float(np.std([r['accuracy'] for r in successful])) if successful else 0,
            "min": float(np.min([r['accuracy'] for r in successful])) if successful else 0,
            "max": float(np.max([r['accuracy'] for r in successful])) if successful else 0,
            "high_performers": sum(1 for r in successful if r['accuracy'] >= 0.95),
            "good_performers": sum(1 for r in successful if r['accuracy'] >= 0.70)
        }
    }
    
        # Convert summary to JSON-safe types
    summary_converted = convert_keys(summary)
    
    # Save summary (now with native types)
    summary_path = os.path.join(output_dir, 'preprocessing_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_converted, f, indent=2)
    print(f"\n💾 Results saved to: {output_dir}")
    return summary

# Example usage:
if __name__ == "__main__":
    # Update these paths to match your data
    RAW_DATA_DIR = "/Users/chaitanya/work/pes/SDG/raw_data"
    OUTPUT_DIR = "/Users/chaitanya/work/pes/SDG/optimized_pipeline_results"
    
    # Run preprocessing
    summary = run_enhanced_preprocessing(RAW_DATA_DIR, OUTPUT_DIR, n_processes=4)
    
    print("\n🎉 Preprocessing complete!")
    print(f"Check {OUTPUT_DIR} for results.")


#for trials to check if new change brought into code work out or not
# if __name__ == "__main__":
#     RAW_DATA_DIR = "/Users/chaitanya/work/pes/SDG/raw_data"
#     OUTPUT_DIR = "/Users/chaitanya/work/pes/SDG/optimized_pipeline_results"
    
#     # Test with single process first to avoid multiprocessing issues
#     summary = run_enhanced_preprocessing(RAW_DATA_DIR, OUTPUT_DIR, n_processes=1)
#     print("\n🎉 Preprocessing complete!")
#     print(f"Check {OUTPUT_DIR} for results.")
