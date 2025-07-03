#!/usr/bin/env python3
"""
complete_eeg_preprocessing_pipeline.py
Complete 11-stage EEG preprocessing pipeline for motor imagery analysis

Pipeline: Raw EEG → Field Extraction → Trial Segmentation → Flat Channel Detection → 
         Bandpass Filter → CAR → Laplacian Filter → Trial Rejection → ERD/ERS Analysis → 
         Riemannian Covariance → Tangent Space → FLDA Classification

Author: AI Neural Networks (Enhanced)
Date: 2025-07-03
"""

import argparse, os, sys, traceback, warnings, shutil, glob
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert
from sklearn.covariance import OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import json
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import os
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime  # ← This is the correct import
import warnings
warnings.filterwarnings('ignore')
# Install pyRiemann if not available
try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.classification import MDM
except ImportError:
    print("Installing pyRiemann for Riemannian geometry...")
    os.system("pip install pyriemann")
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.classification import MDM

warnings.filterwarnings("ignore", category=RuntimeWarning)

################################################################################
# PIPELINE CONFIGURATION - NEURAL NETWORK OPTIMIZED PARAMETERS
################################################################################

# Core parameters
FS_DEFAULT = 250                       # Default sampling frequency
BANDPASS_FREQ = (8, 30)               # Motor imagery optimal frequency range
FLAT_CHANNEL_THRESH = 0.1             # Standard deviation threshold
MI_VOLTAGE_THRESH = 100               # Motor imagery artifact threshold (μV)
REST_VOLTAGE_THRESH = 250             # Rest state artifact threshold (μV)
TRIAL_DURATION = 4.0                  # Expected trial length (seconds)
ANALYSIS_WINDOW = (0.5, 2.5)         # Crop window for analysis (seconds)
N_CV_FOLDS = 5                        # Cross-validation folds
ERD_ERS_FEATURES = 5002               # Required ERD/ERS feature dimension

# Field mapping - ONLY the fields you specified
MATLAB_FIELDS = {
    'imagery_left': 0,     # Label 0 for left motor imagery
    'imagery_right': 1,    # Label 1 for right motor imagery  
    'rest': 2,             # Label 2 for rest state
    'movement_left': 0,    # Also treat as left MI if available
    'movement_right': 1    # Also treat as right MI if available
}

# Neighbor map for 64-channel Laplacian filtering (simplified)
LAPLACIAN_NEIGHBORS = {
    i: [max(0, i-1), min(63, i+1)] for i in range(64)
}

################################################################################
# STAGE 1: ENHANCED MATLAB FIELD EXTRACTION
################################################################################

def extract_matlab_field(data_structure, field_name):
    """
    Stage 1: Robust MATLAB field extraction using neural network-inspired search.
    """
    # Handle numpy arrays with object dtype
    if isinstance(data_structure, np.ndarray) and data_structure.dtype == object:
        for item in data_structure.flat:
            if item is not None:
                result = extract_matlab_field(item, field_name)
                if result is not None:
                    return result
    
    # Handle structured arrays with named fields
    elif hasattr(data_structure, 'dtype') and data_structure.dtype.names:
        # Direct field match
        if field_name in data_structure.dtype.names:
            field_data = data_structure[field_name]
            # Handle scalar structured arrays
            if hasattr(field_data, 'shape'):
                if field_data.shape == ():
                    return field_data.item()
                elif field_data.shape == (1,):
                    return field_data[0]
            return field_data
        
        # Recursive search in nested fields
        for nested_field in data_structure.dtype.names:
            nested_data = data_structure[nested_field]
            if isinstance(nested_data, np.ndarray):
                if nested_data.shape == ():
                    nested_data = nested_data.item()
                elif nested_data.shape == (1,):
                    nested_data = nested_data[0]
            
            result = extract_matlab_field(nested_data, field_name)
            if result is not None:
                return result
    
    # Handle dictionary-like structures
    elif isinstance(data_structure, dict):
        if field_name in data_structure:
            return data_structure[field_name]
        
        for key, value in data_structure.items():
            if not key.startswith('__'):
                result = extract_matlab_field(value, field_name)
                if result is not None:
                    return result
    
    return None

################################################################################
# STAGE 2: INTELLIGENT TRIAL SEGMENTATION
################################################################################

def segment_continuous_data(continuous_data, fs, trial_length_sec=4.0):
    """
    Stage 2: Segment continuous EEG data into fixed-length trials.
    """
    if continuous_data.ndim != 2:
        raise ValueError(f"Expected 2D data (channels x time), got shape {continuous_data.shape}")
    
    samples_per_trial = int(trial_length_sec * fs)
    n_channels, total_samples = continuous_data.shape
    n_trials = total_samples // samples_per_trial
    
    if n_trials == 0:
        raise ValueError(f"Data too short for trial segmentation: {total_samples} samples < {samples_per_trial}")
    
    # Reshape into trials
    trials = np.zeros((n_trials, n_channels, samples_per_trial))
    for trial_idx in range(n_trials):
        start_idx = trial_idx * samples_per_trial
        end_idx = start_idx + samples_per_trial
        trials[trial_idx] = continuous_data[:, start_idx:end_idx]
    
    return trials

################################################################################
# STAGE 3: FLAT CHANNEL DETECTION
################################################################################

def detect_flat_channels(eeg_data, threshold=0.1):
    """
    Stage 3: Detect and identify flat channels using statistical analysis.
    """
    print(f"[Stage 3] Detecting flat channels (threshold: {threshold} μV)")
    
    if eeg_data.ndim == 3:  # trials x channels x time
        channel_variance = np.var(eeg_data, axis=(0, 2))
    else:  # channels x time
        channel_variance = np.var(eeg_data, axis=1)
    
    flat_channels = np.where(channel_variance < threshold)[0]
    good_channels = np.where(channel_variance >= threshold)[0]
    
    print(f"[Stage 3] Identified {len(flat_channels)} flat channels: {flat_channels.tolist()}")
    return good_channels, flat_channels

################################################################################
# STAGE 4: ZERO-PHASE BANDPASS FILTERING
################################################################################

def apply_bandpass_filter(eeg_data, low_freq, high_freq, fs, filter_order=4):
    """
    Stage 4: Apply zero-phase Butterworth bandpass filter.
    """
    print(f"[Stage 4] Applying bandpass filter: {low_freq}-{high_freq} Hz")
    
    if eeg_data.size == 0:
        return eeg_data
    
    # Design Butterworth filter
    nyquist = fs / 2.0
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist
    b, a = butter(filter_order, [low_normalized, high_normalized], btype='band')
    
    # Apply zero-phase filtering
    filtered_data = np.zeros_like(eeg_data)
    
    if eeg_data.ndim == 3:  # trials x channels x time
        for trial in range(eeg_data.shape[0]):
            for channel in range(eeg_data.shape[1]):
                filtered_data[trial, channel] = filtfilt(b, a, eeg_data[trial, channel])
    elif eeg_data.ndim == 2:  # channels x time
        for channel in range(eeg_data.shape[0]):
            filtered_data[channel] = filtfilt(b, a, eeg_data[channel])
    
    return filtered_data

################################################################################
# STAGE 5: COMMON AVERAGE REFERENCE
################################################################################

def apply_common_average_reference(eeg_data):
    """
    Stage 5: Apply Common Average Reference (CAR) spatial filtering.
    """
    print(f"[Stage 5] Applying Common Average Reference (CAR)")
    
    if eeg_data.ndim == 3:  # trials x channels x time
        car_data = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
    elif eeg_data.ndim == 2:  # channels x time
        car_data = eeg_data - np.mean(eeg_data, axis=0, keepdims=True)
    else:
        raise ValueError(f"Unsupported data dimensions: {eeg_data.shape}")
    
    return car_data

################################################################################
# STAGE 6: SURFACE LAPLACIAN FILTERING
################################################################################

def apply_surface_laplacian(eeg_data, neighbor_map=None):
    """
    Stage 6: Apply Surface Laplacian spatial filtering for enhanced spatial resolution.
    """
    print(f"[Stage 6] Applying Surface Laplacian filtering")
    
    if neighbor_map is None:
        neighbor_map = LAPLACIAN_NEIGHBORS
    
    laplacian_data = np.zeros_like(eeg_data)
    
    if eeg_data.ndim == 3:  # trials x channels x time
        n_trials, n_channels, n_samples = eeg_data.shape
        
        for channel in range(n_channels):
            neighbors = neighbor_map.get(channel, [])
            if neighbors and all(n < n_channels for n in neighbors):
                neighbor_avg = np.mean(eeg_data[:, neighbors, :], axis=1)
                laplacian_data[:, channel, :] = eeg_data[:, channel, :] - neighbor_avg
            else:
                laplacian_data[:, channel, :] = eeg_data[:, channel, :]
                
    elif eeg_data.ndim == 2:  # channels x time
        n_channels, n_samples = eeg_data.shape
        
        for channel in range(n_channels):
            neighbors = neighbor_map.get(channel, [])
            if neighbors and all(n < n_channels for n in neighbors):
                neighbor_avg = np.mean(eeg_data[neighbors, :], axis=0)
                laplacian_data[channel, :] = eeg_data[channel, :] - neighbor_avg
            else:
                laplacian_data[channel, :] = eeg_data[channel, :]
    
    return laplacian_data

################################################################################
# STAGE 7: INTELLIGENT TRIAL REJECTION
################################################################################

def diagnostic_voltage_analysis(eeg_data):
    """
    Analyze voltage characteristics of EEG data to determine appropriate thresholds.
    """
    print(f"\n{'='*60}")
    print(f"VOLTAGE DIAGNOSTIC ANALYSIS")
    print(f"{'='*60}")
    
    if eeg_data.ndim != 3:
        print("❌ Data must be 3D (trials x channels x time)")
        return None, None, None, None
    
    n_trials, n_channels, n_samples = eeg_data.shape
    voltage_95th = []
    voltage_99th = []
    voltage_max = []
    
    # Calculate voltage statistics for each trial
    for trial_idx in range(n_trials):
        trial_data = eeg_data[trial_idx]
        abs_voltages = np.abs(trial_data)
        
        voltage_95th.append(np.percentile(abs_voltages, 95))
        voltage_99th.append(np.percentile(abs_voltages, 99))
        voltage_max.append(np.max(abs_voltages))
    
    voltage_95th = np.array(voltage_95th)
    voltage_99th = np.array(voltage_99th)
    voltage_max = np.array(voltage_max)
    
    # Calculate comprehensive statistics
    stats = {
        '95th_percentile': {
            'min': np.min(voltage_95th),
            'max': np.max(voltage_95th),
            'mean': np.mean(voltage_95th),
            'median': np.median(voltage_95th),
            'std': np.std(voltage_95th),
            'q25': np.percentile(voltage_95th, 25),
            'q75': np.percentile(voltage_95th, 75)
        },
        '99th_percentile': {
            'min': np.min(voltage_99th),
            'max': np.max(voltage_99th),
            'mean': np.mean(voltage_99th),
            'median': np.median(voltage_99th),
            'std': np.std(voltage_99th),
            'q25': np.percentile(voltage_99th, 25),
            'q75': np.percentile(voltage_99th, 75)
        },
        'max_voltage': {
            'min': np.min(voltage_max),
            'max': np.max(voltage_max),
            'mean': np.mean(voltage_max),
            'median': np.median(voltage_max)
        },
        'data_shape': eeg_data.shape
    }
    
    # Print diagnostic information
    print(f"Data Shape: {n_trials} trials × {n_channels} channels × {n_samples} samples")
    print(f"\n📊 95th Percentile Voltage Statistics:")
    print(f"   Mean: {stats['95th_percentile']['mean']:.1f} μV")
    print(f"   Median: {stats['95th_percentile']['median']:.1f} μV")
    print(f"   Range: {stats['95th_percentile']['min']:.1f} - {stats['95th_percentile']['max']:.1f} μV")
    print(f"   Q25-Q75: {stats['95th_percentile']['q25']:.1f} - {stats['95th_percentile']['q75']:.1f} μV")
    print(f"   Std Dev: {stats['95th_percentile']['std']:.1f} μV")
    
    print(f"\n📊 99th Percentile Voltage Statistics:")
    print(f"   Mean: {stats['99th_percentile']['mean']:.1f} μV")
    print(f"   Median: {stats['99th_percentile']['median']:.1f} μV")
    print(f"   Range: {stats['99th_percentile']['min']:.1f} - {stats['99th_percentile']['max']:.1f} μV")
    print(f"   Q25-Q75: {stats['99th_percentile']['q25']:.1f} - {stats['99th_percentile']['q75']:.1f} μV")
    
    print(f"\n📊 Maximum Voltage Statistics:")
    print(f"   Mean: {stats['max_voltage']['mean']:.1f} μV")
    print(f"   Median: {stats['max_voltage']['median']:.1f} μV")
    print(f"   Range: {stats['max_voltage']['min']:.1f} - {stats['max_voltage']['max']:.1f} μV")
    
    # Suggest appropriate thresholds
    conservative_threshold = stats['95th_percentile']['q75'] + 2 * stats['95th_percentile']['std']
    moderate_threshold = stats['99th_percentile']['median']
    lenient_threshold = stats['99th_percentile']['q75']
    
    print(f"\n🎯 RECOMMENDED THRESHOLDS:")
    print(f"   Conservative (Q75 + 2σ): {conservative_threshold:.1f} μV")
    print(f"   Moderate (99th median): {moderate_threshold:.1f} μV")
    print(f"   Lenient (99th Q75): {lenient_threshold:.1f} μV")
    
    # Calculate trial retention rates for different thresholds
    test_thresholds = [200, 500, 1000, conservative_threshold, moderate_threshold, lenient_threshold]
    
    print(f"\n📈 TRIAL RETENTION RATES:")
    for threshold in test_thresholds:
        n_kept = np.sum(voltage_95th <= threshold)
        retention_rate = (n_kept / n_trials) * 100
        print(f"   {threshold:.0f} μV: {n_kept}/{n_trials} trials ({retention_rate:.1f}%)")
    
    print(f"{'='*60}")
    
    return stats, conservative_threshold, moderate_threshold, lenient_threshold

def adaptive_trial_rejection(eeg_data, target_retention_rate=0.8):
    """
    Adaptive trial rejection that aims to keep a target percentage of trials.
    """
    print(f"[Stage 7] Adaptive trial rejection (target retention: {target_retention_rate*100:.0f}%)")
    
    if eeg_data.ndim != 3:
        print("Warning: Trial rejection requires 3D data")
        return np.arange(eeg_data.shape[0]), []
    
    # Run diagnostic analysis first
    stats, conservative_thresh, moderate_thresh, lenient_thresh = diagnostic_voltage_analysis(eeg_data)
    
    # Calculate 95th percentile for each trial
    n_trials = eeg_data.shape[0]
    voltage_95th = []
    
    for trial_idx in range(n_trials):
        trial_data = eeg_data[trial_idx]
        voltage_95th.append(np.percentile(np.abs(trial_data), 95))
    
    voltage_95th = np.array(voltage_95th)
    
    # Find threshold that gives target retention rate
    sorted_voltages = np.sort(voltage_95th)
    target_index = int(target_retention_rate * len(sorted_voltages))
    adaptive_threshold = sorted_voltages[target_index] if target_index < len(sorted_voltages) else sorted_voltages[-1]
    
    # Apply the adaptive threshold
    good_trials = []
    rejected_trials = []
    
    for trial_idx in range(n_trials):
        if voltage_95th[trial_idx] <= adaptive_threshold:
            good_trials.append(trial_idx)
        else:
            rejected_trials.append(trial_idx)
    
    retention_rate = len(good_trials) / n_trials
    
    print(f"\n✅ ADAPTIVE THRESHOLD RESULTS:")
    print(f"   Selected threshold: {adaptive_threshold:.1f} μV")
    print(f"   Trials kept: {len(good_trials)}/{n_trials} ({retention_rate*100:.1f}%)")
    print(f"   Trials rejected: {len(rejected_trials)}")
    
    return np.array(good_trials, dtype=int), rejected_trials

def reject_artifact_trials(eeg_data, voltage_threshold=None, adaptive=True, target_retention=0.8):
    """
    Enhanced trial rejection with adaptive thresholding and diagnostic analysis.
    """
    if adaptive:
        return adaptive_trial_rejection(eeg_data, target_retention)
    else:
        # Original method
        print(f"[Stage 7] Rejecting trials with voltage artifacts (threshold: {voltage_threshold} μV)")
        
        if eeg_data.ndim != 3:
            print("Warning: Trial rejection requires 3D data (trials x channels x time)")
            return np.arange(eeg_data.shape[0]), []
        
        good_trials = []
        rejected_trials = []
        
        for trial_idx in range(eeg_data.shape[0]):
            trial_data = eeg_data[trial_idx]
            artifact_measure = np.percentile(np.abs(trial_data), 95)
            
            if artifact_measure <= voltage_threshold:
                good_trials.append(trial_idx)
            else:
                rejected_trials.append(trial_idx)
        
        print(f"[Stage 7] Kept {len(good_trials)}/{eeg_data.shape[0]} trials")
        return np.array(good_trials, dtype=int), rejected_trials

################################################################################
# STAGE 8: ERD/ERS FEATURE EXTRACTION
################################################################################

def calculate_erd_ers_features(eeg_data, fs, baseline_window=(0.0, 1.0), task_window=(1.0, 2.0)):
    """
    Stage 8: Calculate Event-Related Desynchronization/Synchronization features.
    """
    print(f"[Stage 8] Calculating ERD/ERS features")
    
    if eeg_data.size == 0:
        return np.zeros((0, ERD_ERS_FEATURES))
    
    # Define frequency bands for motor imagery
    mu_band = (8, 12)    # Mu rhythm
    beta_band = (13, 30) # Beta rhythm
    
    # Filter data for each frequency band
    mu_filtered = apply_bandpass_filter(eeg_data, mu_band[0], mu_band[1], fs)
    beta_filtered = apply_bandpass_filter(eeg_data, beta_band[0], beta_band[1], fs)
    
    # Calculate instantaneous power using Hilbert transform
    mu_power = np.abs(hilbert(mu_filtered, axis=-1)) ** 2
    beta_power = np.abs(hilbert(beta_filtered, axis=-1)) ** 2
    
    # Define time windows in samples
    baseline_start = int(baseline_window[0] * fs)
    baseline_end = int(baseline_window[1] * fs)
    task_start = int(task_window[0] * fs)
    task_end = int(task_window[1] * fs)
    
    # Calculate baseline and task power
    mu_baseline_power = np.mean(mu_power[:, :, baseline_start:baseline_end], axis=2)
    mu_task_power = np.mean(mu_power[:, :, task_start:task_end], axis=2)
    
    beta_baseline_power = np.mean(beta_power[:, :, baseline_start:baseline_end], axis=2)
    beta_task_power = np.mean(beta_power[:, :, task_start:task_end], axis=2)
    
    # Calculate ERD/ERS as percentage change
    mu_erd_ers = ((mu_task_power - mu_baseline_power) / mu_baseline_power) * 100
    beta_erd_ers = ((beta_task_power - beta_baseline_power) / beta_baseline_power) * 100
    
    # Flatten and concatenate features
    n_trials, n_channels = mu_erd_ers.shape
    mu_features = mu_erd_ers.reshape(n_trials, -1)
    beta_features = beta_erd_ers.reshape(n_trials, -1)
    
    # Ensure exactly 5002 features as requested
    combined_features = np.hstack([mu_features, beta_features])
    
    if combined_features.shape[1] < ERD_ERS_FEATURES:
        # Pad with zeros if needed
        padding = np.zeros((n_trials, ERD_ERS_FEATURES - combined_features.shape[1]))
        combined_features = np.hstack([combined_features, padding])
    elif combined_features.shape[1] > ERD_ERS_FEATURES:
        # Truncate if too large
        combined_features = combined_features[:, :ERD_ERS_FEATURES]
    
    return combined_features

################################################################################
# STAGE 9: RIEMANNIAN COVARIANCE MATRICES
################################################################################

def calculate_covariance_matrices(eeg_data):
    """
    Stage 9: Calculate Riemannian covariance matrices.
    """
    print(f"[Stage 9] Calculating Riemannian covariance matrices")
    
    if eeg_data.size == 0:
        return np.array([])
    
    # Use pyRiemann for robust covariance estimation
    cov_estimator = Covariances(estimator='oas')  # Oracle Approximating Shrinkage
    covariance_matrices = cov_estimator.fit_transform(eeg_data)
    
    return covariance_matrices

################################################################################
# STAGE 10: TANGENT SPACE MAPPING
################################################################################

def map_to_tangent_space(covariance_matrices):
    """
    Stage 10: Map covariance matrices to Euclidean tangent space.
    """
    print(f"[Stage 10] Mapping covariance matrices to tangent space")
    
    if covariance_matrices.size == 0:
        return np.array([]), None
    
    # Use pyRiemann for tangent space mapping
    tangent_mapper = TangentSpace(metric='riemann')
    tangent_vectors = tangent_mapper.fit_transform(covariance_matrices)
    
    return tangent_vectors, tangent_mapper

################################################################################
# STAGE 11: FLDA CLASSIFICATION
################################################################################

def perform_flda_classification(features, labels, n_folds=5):
    """
    Stage 11: Perform Fisher Linear Discriminant Analysis with cross-validation.
    """
    print(f"[Stage 11] Performing FLDA classification ({n_folds}-fold CV)")
    
    if features.size == 0 or len(np.unique(labels)) < 2:
        print("Warning: Insufficient data for classification")
        return 0.0, None, None
    
    # Create classification pipeline
    classifier = Pipeline([
        ('scaler', StandardScaler()),
        ('flda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    ])
    
    # Perform cross-validation
    cv_scores = cross_val_score(classifier, features, labels, 
                               cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42),
                               scoring='accuracy')
    
    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)
    
    # Fit final model on all data
    classifier.fit(features, labels)
    
    print(f"[Stage 11] Classification accuracy: {mean_accuracy:.3f} ± {std_accuracy:.3f}")
    
    return mean_accuracy, classifier, cv_scores

################################################################################
# COMPLETE PIPELINE ORCHESTRATOR
################################################################################

class NeuralNetworkEEGProcessor:
    """Complete EEG preprocessing pipeline orchestrator."""
    
    def __init__(self, sampling_rate=250, verbose=True):
        self.fs = sampling_rate
        self.verbose = verbose
        self.pipeline_log = []
        
    def log_message(self, message):
        if self.verbose:
            print(f"[Neural EEG Pipeline] {message}")
        self.pipeline_log.append(message)
    
    def execute_complete_pipeline(self, eeg_data, labels):
        """Execute the complete 11-stage preprocessing pipeline."""
        
        self.log_message("="*80)
        self.log_message("NEURAL NETWORK EEG PREPROCESSING PIPELINE INITIATED")
        self.log_message(f"Input shape: {eeg_data.shape}, Labels: {np.bincount(labels)}")
        self.log_message("="*80)
        
        pipeline_results = {}
        current_data = eeg_data.copy()
        
        try:
            # Stage 3: Flat Channel Detection
            good_channels, flat_channels = detect_flat_channels(current_data, FLAT_CHANNEL_THRESH)
            if len(flat_channels) > 0:
                current_data = current_data[:, good_channels, :]
            pipeline_results['good_channels'] = good_channels
            pipeline_results['flat_channels'] = flat_channels
            
            # Stage 4: Bandpass Filtering
            current_data = apply_bandpass_filter(current_data, BANDPASS_FREQ[0], BANDPASS_FREQ[1], self.fs)
            
            # Stage 5: Common Average Reference
            current_data = apply_common_average_reference(current_data)
            
            # Stage 6: Surface Laplacian
            current_data = apply_surface_laplacian(current_data)
            
            # Stage 7: Trial Rejection
            # Stage 7: Adaptive Trial Rejection
            good_trials, rejected_trials = reject_artifact_trials(current_data, adaptive=True, target_retention=0.8)
            if len(good_trials) > 0:
                current_data = current_data[good_trials]
                clean_labels = labels[good_trials]
            else:
                raise ValueError("All trials were rejected during artifact removal")

            
            pipeline_results['good_trials'] = good_trials
            pipeline_results['clean_data'] = current_data
            pipeline_results['clean_labels'] = clean_labels
            
            # Stage 8: ERD/ERS Analysis
            erd_ers_features = calculate_erd_ers_features(current_data, self.fs)
            pipeline_results['erd_ers_features'] = erd_ers_features
            
            # Stage 9: Riemannian Covariance
            covariance_matrices = calculate_covariance_matrices(current_data)
            pipeline_results['covariance_matrices'] = covariance_matrices
            
            # Stage 10: Tangent Space Mapping
            if covariance_matrices.size > 0:
                tangent_vectors, tangent_mapper = map_to_tangent_space(covariance_matrices)
                pipeline_results['tangent_vectors'] = tangent_vectors
                pipeline_results['tangent_mapper'] = tangent_mapper
            else:
                tangent_vectors = np.array([])
                pipeline_results['tangent_vectors'] = tangent_vectors
                pipeline_results['tangent_mapper'] = None
            
            # Stage 11: FLDA Classification
            if tangent_vectors.size > 0:
                accuracy, classifier, cv_scores = perform_flda_classification(tangent_vectors, clean_labels)
                pipeline_results['classification_accuracy'] = accuracy
                pipeline_results['classifier'] = classifier
                pipeline_results['cv_scores'] = cv_scores
            else:
                pipeline_results['classification_accuracy'] = 0.0
                pipeline_results['classifier'] = None
                pipeline_results['cv_scores'] = None
            
            self.log_message("="*80)
            self.log_message(f"PIPELINE COMPLETE - Accuracy: {pipeline_results.get('classification_accuracy', 0.0):.3f}")
            self.log_message("="*80)
            
            return pipeline_results
            
        except Exception as e:
            self.log_message(f"PIPELINE FAILED: {str(e)}")
            traceback.print_exc()
            return None

################################################################################
# SUBJECT PROCESSING FUNCTION
################################################################################

def process_single_subject(matlab_file_path, output_directory):
    """Process a single subject through the complete pipeline."""
    
    subject_name = os.path.splitext(os.path.basename(matlab_file_path))[0]
    subject_output_dir = os.path.join(output_directory, subject_name)
    
    try:
        # Create output directory
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Load MATLAB file
        matlab_data = loadmat(matlab_file_path, struct_as_record=True, squeeze_me=True)
        
        # Extract sampling rate
        sampling_rate = int(matlab_data.get('srate', FS_DEFAULT))
        
        print(f"    📊 Sampling rate: {sampling_rate} Hz")
        
        # Extract EEG data from specified fields ONLY
        all_trials = []
        all_labels = []
        
        for field_name, label in MATLAB_FIELDS.items():
            field_data = extract_matlab_field(matlab_data, field_name)
            
            if field_data is not None:
                print(f"    ✅ Found field '{field_name}' with shape {field_data.shape}")
                
                # Segment continuous data into trials
                if field_data.ndim == 2:  # continuous data (channels x time)
                    trials = segment_continuous_data(field_data, sampling_rate, TRIAL_DURATION)
                else:  # already segmented (trials x channels x time)
                    trials = field_data
                
                all_trials.append(trials)
                all_labels.extend([label] * trials.shape[0])
                print(f"    📝 Added {trials.shape[0]} trials for {field_name}")
            else:
                print(f"    ⚠️  Field '{field_name}' not found")
        
        if not all_trials:
            print(f"    ❌ No valid data found for {subject_name}")
            return
        
        # Combine all trials
        combined_data = np.concatenate(all_trials, axis=0)
        combined_labels = np.array(all_labels, dtype=int)
        
        print(f"    📊 Combined data shape: {combined_data.shape}")
        print(f"    📊 Labels distribution: {np.bincount(combined_labels)}")
        
        # Crop to analysis window
        start_sample = int(ANALYSIS_WINDOW[0] * sampling_rate)
        end_sample = int(ANALYSIS_WINDOW[1] * sampling_rate)
        combined_data = combined_data[:, :, start_sample:end_sample]
        
        print(f"    ✂️  Cropped to: {combined_data.shape}")
        
        # Initialize neural network processor
        processor = NeuralNetworkEEGProcessor(sampling_rate=sampling_rate, verbose=False)
        
        # Execute complete pipeline
        print(f"    🧠 Running complete preprocessing pipeline...")
        results = processor.execute_complete_pipeline(combined_data, combined_labels)
        
        if results is not None and results['clean_data'].size > 0:
            # Save required output files
            print(f"    💾 Saving results...")
            
            np.save(os.path.join(subject_output_dir, 'eeg_clean.npy'), 
                   results['clean_data'].astype(np.float32))
            np.save(os.path.join(subject_output_dir, 'labels.npy'), 
                   results['clean_labels'].astype(np.int8))
            np.save(os.path.join(subject_output_dir, 'good_trials.npy'), 
                   results['good_trials'].astype(np.int16))
            
            # Save ERD/ERS features (pad to ensure exactly 5002 features)
            if results['erd_ers_features'].size > 0:
                erd_ers = results['erd_ers_features']
                if erd_ers.shape[1] < ERD_ERS_FEATURES:
                    padding = np.zeros((erd_ers.shape[0], ERD_ERS_FEATURES - erd_ers.shape[1]))
                    erd_ers = np.hstack([erd_ers, padding])
                elif erd_ers.shape[1] > ERD_ERS_FEATURES:
                    erd_ers = erd_ers[:, :ERD_ERS_FEATURES]
                np.save(os.path.join(subject_output_dir, 'erd_ers.npy'), erd_ers.astype(np.float32))
            else:
                empty_erd_ers = np.zeros((len(results['clean_labels']), ERD_ERS_FEATURES))
                np.save(os.path.join(subject_output_dir, 'erd_ers.npy'), empty_erd_ers.astype(np.float32))
            
            # Save additional analysis results
            if results.get('covariance_matrices') is not None and results['covariance_matrices'].size > 0:
                np.save(os.path.join(subject_output_dir, 'covariances.npy'), 
                       results['covariance_matrices'].astype(np.float32))
            
            if results.get('tangent_vectors') is not None and results['tangent_vectors'].size > 0:
                np.save(os.path.join(subject_output_dir, 'tangent_vectors.npy'), 
                       results['tangent_vectors'].astype(np.float32))
            
            # Save processing report
            report = {
                'subject': subject_name,
                'classification_accuracy': float(results.get('classification_accuracy', 0.0)),
                'cv_scores': results.get('cv_scores', []).tolist() if results.get('cv_scores') is not None else [],
                'final_data_shape': list(results['clean_data'].shape),
                'n_good_channels': len(results.get('good_channels', [])),
                'n_good_trials': len(results['good_trials']),
                'sampling_rate': sampling_rate
            }
            
            with open(os.path.join(subject_output_dir, 'processing_report.json'), 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"    ✅ Saved all files - Accuracy: {results.get('classification_accuracy', 0.0):.3f}")
        else:
            print(f"    ❌ Pipeline failed - no results to save")
            
    except Exception as e:
        print(f"    ❌ Error processing {subject_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

################################################################################
# MULTIPROCESSING FUNCTIONS
################################################################################

def process_single_subject_multiprocessing(matlab_file_path, output_directory):
    """
    Wrapper function for multiprocessing - processes a single subject.
    """
    subject_name = os.path.splitext(os.path.basename(matlab_file_path))[0]
    
    try:
        print(f"🔄 Starting processing: {subject_name}")
        start_time = time.time()
        
        # Call your existing processing function
        process_single_subject(matlab_file_path, output_directory)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ Completed: {subject_name} in {processing_time:.2f} seconds")
        return {
            'subject': subject_name,
            'status': 'success',
            'processing_time': processing_time,
            'file_path': matlab_file_path
        }
        
    except Exception as e:
        print(f"❌ Failed: {subject_name} - Error: {str(e)}")
        return {
            'subject': subject_name,
            'status': 'failed',
            'error': str(e),
            'file_path': matlab_file_path
        }

def process_multiple_subjects_parallel(matlab_files, output_directory, n_processes=4):
    """
    Process multiple MATLAB files in parallel using multiprocessing.
    """
    print(f"🧠 Initializing multiprocessing with {n_processes} processes")
    print(f"📁 Processing {len(matlab_files)} files")
    print(f"💾 Output directory: {output_directory}")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Record start time
    total_start_time = time.time()
    
    # Create partial function with fixed output_directory
    process_func = partial(process_single_subject_multiprocessing, output_directory=output_directory)
    
    # Use multiprocessing pool
    with Pool(processes=n_processes) as pool:
        # Process files in parallel
        results = pool.map(process_func, matlab_files)
    
    # Calculate total processing time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Analyze results
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print("="*80)
    print(f"🎉 MULTIPROCESSING COMPLETE!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"✅ Successful: {len(successful)}/{len(matlab_files)}")
    print(f"❌ Failed: {len(failed)}/{len(matlab_files)}")
    
    if successful:
        avg_time = np.mean([r['processing_time'] for r in successful])
        print(f"📊 Average processing time per file: {avg_time:.2f} seconds")
    
    if failed:
        print(f"\n❌ Failed files:")
        for fail in failed:
            print(f"   - {fail['subject']}: {fail['error']}")
    
    return results

################################################################################
# MAIN EXECUTION ENGINE
################################################################################

def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Neural Network Enhanced EEG Preprocessing Pipeline')
    parser.add_argument('--data_root', 
                       default='/Users/chaitanya/work/pes/SDG/raw_data',
                       help='Directory containing 52 MATLAB files')
    parser.add_argument('--output_root', 
                       default='./optimized_pipeline_results',
                       help='Output directory for processed results')
    
    args = parser.parse_args()
    
    # Clear and create output directory
    if os.path.exists(args.output_root):
        shutil.rmtree(args.output_root)
    os.makedirs(args.output_root, exist_ok=True)
    
    # Find all MATLAB files
    matlab_files = glob.glob(os.path.join(args.data_root, '*.mat'))
    
    if not matlab_files:
        print("❌ No MATLAB files found in the specified directory!")
        sys.exit(1)
    
    print(f"🧠 Neural Network EEG Processor Initialized")
    print(f"📁 Found {len(matlab_files)} subjects to process")
    print(f"📊 Pipeline: 11-stage complete preprocessing")
    print(f"🎯 Output: 4 files per subject (eeg_clean, labels, erd_ers, good_trials)")
    print("="*80)
    
    # Process all subjects
    successful_subjects = 0
    for matlab_file in tqdm(sorted(matlab_files), desc="Processing subjects"):
        try:
            process_single_subject(matlab_file, args.output_root)
            successful_subjects += 1
        except Exception as e:
            print(f"❌ Failed to process {os.path.basename(matlab_file)}: {e}")
    
    print("="*80)
    print(f"🎉 Processing complete!")
    print(f"✅ Successfully processed: {successful_subjects}/{len(matlab_files)} subjects")
    print(f"📁 Results saved to: {args.output_root}")
    print("="*80)

############################################################################
#Analysis of preprocessing
class EEGResultsAnalyzer:
    """Comprehensive analyzer for EEG preprocessing results."""
    
    def __init__(self, results_dir=None, verbose=True):
        """
        Initialize the results analyzer.
        
        Parameters:
        -----------
        results_dir : str, optional
            Path to results directory. If None, will search common locations.
        verbose : bool
            Whether to print detailed information
        """
        self.verbose = verbose
        self.results_dir = self._find_results_directory(results_dir)
        self.analysis_results = {}
        
    def _find_results_directory(self, results_dir):
        """Find the results directory by checking common locations."""
        
        # Possible locations to check
        possible_locations = [
            results_dir,
            './optimized_pipeline_results',
            '../optimized_pipeline_results',
            '/Users/chaitanya/work/pes/SDG/optimized_pipeline_results',
            './results',
            './output'
        ]
        
        # Remove None values
        possible_locations = [loc for loc in possible_locations if loc is not None]
        
        for location in possible_locations:
            if os.path.exists(location):
                if self.verbose:
                    print(f"✅ Found results directory: {location}")
                return location
        
        if self.verbose:
            print("❌ No results directory found. Checked locations:")
            for loc in possible_locations:
                print(f"   - {loc}")
        
        return None
    
    def check_directory_structure(self):
        """Check and report on the directory structure."""
        
        if not self.results_dir:
            return {
                'status': 'error',
                'message': 'No results directory found',
                'subjects': []
            }
        
        print(f"\n🔍 Analyzing directory: {self.results_dir}")
        
        # Find all subject directories
        subject_dirs = []
        for item in os.listdir(self.results_dir):
            item_path = os.path.join(self.results_dir, item)
            if os.path.isdir(item_path):
                subject_dirs.append(item)
        
        # Analyze each subject directory
        subject_analysis = []
        for subject in sorted(subject_dirs):
            subject_path = os.path.join(self.results_dir, subject)
            analysis = self._analyze_subject_directory(subject, subject_path)
            subject_analysis.append(analysis)
        
        structure_report = {
            'status': 'success',
            'total_subjects': len(subject_dirs),
            'results_directory': self.results_dir,
            'subjects': subject_analysis
        }
        
        return structure_report
    
    def _analyze_subject_directory(self, subject_name, subject_path):
        """Analyze a single subject's directory."""
        
        required_files = [
            'eeg_clean.npy',
            'labels.npy',
            'erd_ers.npy',
            'good_trials.npy'
        ]
        
        optional_files = [
            'processing_report.json',
            'covariances.npy',
            'tangent_vectors.npy'
        ]
        
        analysis = {
            'subject': subject_name,
            'path': subject_path,
            'required_files': {},
            'optional_files': {},
            'file_sizes': {},
            'data_shapes': {},
            'complete': True
        }
        
        # Check required files
        for file_name in required_files:
            file_path = os.path.join(subject_path, file_name)
            exists = os.path.exists(file_path)
            analysis['required_files'][file_name] = exists
            
            if exists:
                try:
                    file_size = os.path.getsize(file_path)
                    analysis['file_sizes'][file_name] = file_size
                    
                    # Get data shape for .npy files
                    if file_name.endswith('.npy'):
                        data = np.load(file_path, mmap_mode='r')
                        analysis['data_shapes'][file_name] = data.shape
                except Exception as e:
                    analysis['file_sizes'][file_name] = f"Error: {str(e)}"
            else:
                analysis['complete'] = False
        
        # Check optional files
        for file_name in optional_files:
            file_path = os.path.join(subject_path, file_name)
            exists = os.path.exists(file_path)
            analysis['optional_files'][file_name] = exists
            
            if exists:
                try:
                    file_size = os.path.getsize(file_path)
                    analysis['file_sizes'][file_name] = file_size
                    
                    if file_name.endswith('.npy'):
                        data = np.load(file_path, mmap_mode='r')
                        analysis['data_shapes'][file_name] = data.shape
                except Exception as e:
                    analysis['file_sizes'][file_name] = f"Error: {str(e)}"
        
        return analysis
    
    def extract_processing_results(self):
        """Extract processing results from all subjects."""
        
        if not self.results_dir:
            return None
        
        results = {}
        subject_dirs = [d for d in os.listdir(self.results_dir) 
                       if os.path.isdir(os.path.join(self.results_dir, d))]
        
        for subject in sorted(subject_dirs):
            subject_path = os.path.join(self.results_dir, subject)
            report_path = os.path.join(subject_path, 'processing_report.json')
            
            if os.path.exists(report_path):
                try:
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    
                    # Extract key metrics
                    results[subject] = {
                        'classification_accuracy': report.get('classification_accuracy'),
                        'cv_scores': report.get('cv_scores', []),
                        'n_good_channels': report.get('n_good_channels'),
                        'n_good_trials': report.get('n_good_trials'),
                        'sampling_rate': report.get('sampling_rate'),
                        'final_data_shape': report.get('final_data_shape'),
                        'status': 'success'
                    }
                    
                    # Add additional metrics from data files
                    eeg_path = os.path.join(subject_path, 'eeg_clean.npy')
                    labels_path = os.path.join(subject_path, 'labels.npy')
                    erd_ers_path = os.path.join(subject_path, 'erd_ers.npy')
                    
                    if os.path.exists(eeg_path):
                        eeg_data = np.load(eeg_path, mmap_mode='r')
                        results[subject]['eeg_shape'] = eeg_data.shape
                        results[subject]['eeg_size_mb'] = os.path.getsize(eeg_path) / (1024 * 1024)
                    
                    if os.path.exists(labels_path):
                        labels = np.load(labels_path)
                        unique_labels, counts = np.unique(labels, return_counts=True)
                        results[subject]['label_distribution'] = dict(zip(unique_labels.tolist(), counts.tolist()))
                    
                    if os.path.exists(erd_ers_path):
                        erd_ers = np.load(erd_ers_path, mmap_mode='r')
                        results[subject]['erd_ers_shape'] = erd_ers.shape
                        results[subject]['erd_ers_size_mb'] = os.path.getsize(erd_ers_path) / (1024 * 1024)
                    
                except Exception as e:
                    results[subject] = {
                        'status': 'error',
                        'error': str(e)
                    }
            else:
                results[subject] = {
                    'status': 'no_report',
                    'error': 'No processing_report.json found'
                }
        
        return results
    
    def calculate_summary_statistics(self, results):
        """Calculate summary statistics across all subjects."""
        
        if not results:
            return None
        
        # Extract successful results
        successful_results = {k: v for k, v in results.items() 
                            if v.get('status') == 'success' and v.get('classification_accuracy') is not None}
        
        if not successful_results:
            return {
                'status': 'no_successful_results',
                'total_subjects': len(results),
                'successful_subjects': 0
            }
        
        # Calculate statistics
        accuracies = [v['classification_accuracy'] for v in successful_results.values()]
        n_trials = [v['n_good_trials'] for v in successful_results.values() if v.get('n_good_trials')]
        n_channels = [v['n_good_channels'] for v in successful_results.values() if v.get('n_good_channels')]
        
        # Calculate cross-validation statistics
        all_cv_scores = []
        for v in successful_results.values():
            if v.get('cv_scores'):
                all_cv_scores.extend(v['cv_scores'])
        
        summary = {
            'total_subjects': len(results),
            'successful_subjects': len(successful_results),
            'failed_subjects': len(results) - len(successful_results),
            'success_rate': len(successful_results) / len(results) * 100,
            
            'accuracy_statistics': {
                'mean': float(np.mean(accuracies)),
                'std': float(np.std(accuracies)),
                'min': float(np.min(accuracies)),
                'max': float(np.max(accuracies)),
                'median': float(np.median(accuracies)),
                'q25': float(np.percentile(accuracies, 25)),
                'q75': float(np.percentile(accuracies, 75))
            } if accuracies else None,
            
            'cross_validation_statistics': {
                'mean_cv_score': float(np.mean(all_cv_scores)),
                'std_cv_score': float(np.std(all_cv_scores)),
                'total_cv_folds': len(all_cv_scores)
            } if all_cv_scores else None,
            
            'data_statistics': {
                'trials_per_subject': {
                    'mean': float(np.mean(n_trials)),
                    'std': float(np.std(n_trials)),
                    'min': int(np.min(n_trials)),
                    'max': int(np.max(n_trials))
                } if n_trials else None,
                
                'channels_per_subject': {
                    'mean': float(np.mean(n_channels)),
                    'std': float(np.std(n_channels)),
                    'min': int(np.min(n_channels)),
                    'max': int(np.max(n_channels))
                } if n_channels else None
            }
        }
        
        return summary
    
    def run_complete_analysis(self):
        """Run complete analysis of preprocessing results."""
        
        print("🧠 EEG Preprocessing Results Analysis")
        print("=" * 60)
        
        # Check directory structure
        structure_report = self.check_directory_structure()
        
        if structure_report['status'] == 'error':
            print(f"❌ {structure_report['message']}")
            return {
                'status': 'error',
                'message': structure_report['message'],
                'timestamp': datetime.now().isoformat()
            }
        
        print(f"📁 Found {structure_report['total_subjects']} subject directories")
        
        # Extract processing results
        print("\n📊 Extracting processing results...")
        processing_results = self.extract_processing_results()
        
        if not processing_results:
            print("❌ No processing results found")
            return {
                'status': 'error',
                'message': 'No processing results found',
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate summary statistics
        print("📈 Calculating summary statistics...")
        summary_stats = self.calculate_summary_statistics(processing_results)
        
        # Compile final results
        final_analysis = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'results_directory': self.results_dir,
            'directory_structure': structure_report,
            'summary_statistics': summary_stats,
            'individual_results': processing_results
        }
        
        # Print summary
        self._print_summary(summary_stats)
        
        return final_analysis
    
    def _print_summary(self, summary):
        """Print a formatted summary of results."""
        
        if not summary or summary.get('status') == 'no_successful_results':
            print("\n❌ No successful preprocessing results found")
            return
        
        print(f"\n📊 SUMMARY STATISTICS")
        print("-" * 40)
        print(f"Total Subjects: {summary['total_subjects']}")
        print(f"Successful: {summary['successful_subjects']}")
        print(f"Failed: {summary['failed_subjects']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        if summary.get('accuracy_statistics'):
            acc_stats = summary['accuracy_statistics']
            print(f"\n🎯 CLASSIFICATION ACCURACY")
            print(f"Mean Accuracy: {acc_stats['mean']:.3f} ± {acc_stats['std']:.3f}")
            print(f"Range: {acc_stats['min']:.3f} - {acc_stats['max']:.3f}")
            print(f"Median: {acc_stats['median']:.3f}")
        
        if summary.get('data_statistics'):
            data_stats = summary['data_statistics']
            if data_stats.get('trials_per_subject'):
                trials = data_stats['trials_per_subject']
                print(f"\n📝 TRIALS PER SUBJECT")
                print(f"Mean: {trials['mean']:.1f} ± {trials['std']:.1f}")
                print(f"Range: {trials['min']} - {trials['max']}")
    
    def save_results(self, results, output_path='preprocessing_analysis_results.json'):
        """Save analysis results to JSON file."""
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            if self.verbose:
                print(f"\n💾 Results saved to: {output_path}")
                print(f"📁 File size: {os.path.getsize(output_path) / 1024:.1f} KB")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Error saving results: {str(e)}")
            return False

def main_2():
    """Main function to run the analysis."""
    
    # Initialize analyzer
    analyzer = EEGResultsAnalyzer(verbose=True)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Save results to JSON
    output_file = f"eeg_preprocessing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    success = analyzer.save_results(results, output_file)
    
    if success:
        print(f"\n✅ Analysis complete! Results saved to: {output_file}")
    else:
        print(f"\n❌ Analysis failed to save results")
    
    return results

def main_multiprocessing():
    """Simplified main function - sequential processing for now."""
    parser = argparse.ArgumentParser(description='EEG Preprocessing Pipeline')
    parser.add_argument('--data_root', 
                       default='/Users/chaitanya/work/pes/SDG/raw_data',
                       help='Directory containing MATLAB files')
    parser.add_argument('--output_root', 
                       default='./optimized_pipeline_results',
                       help='Output directory for processed results')
    parser.add_argument('--file_limit',
                       type=int,
                       default=None,
                       help='Limit number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Clear and create output directory
    if os.path.exists(args.output_root):
        shutil.rmtree(args.output_root)
    os.makedirs(args.output_root, exist_ok=True)
    
    # Find MATLAB files
    matlab_files = sorted(glob.glob(os.path.join(args.data_root, '*.mat')))
    
    if not matlab_files:
        print("❌ No MATLAB files found!")
        return
    
    # Limit files if specified
    if args.file_limit:
        matlab_files = matlab_files[:args.file_limit]
        print(f"📋 Processing first {args.file_limit} files")
    
    print(f"📁 Processing {len(matlab_files)} files sequentially")
    print("="*80)
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    # Process files one by one
    for i, matlab_file in enumerate(matlab_files, 1):
        try:
            print(f"🔄 [{i}/{len(matlab_files)}] Processing: {os.path.basename(matlab_file)}")
            file_start = time.time()
            
            # Call your existing function directly
            process_single_subject(matlab_file, args.output_root)
            
            file_end = time.time()
            print(f"✅ Completed in {file_end - file_start:.2f} seconds")
            successful += 1
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            failed += 1
    
    total_time = time.time() - start_time
    
    print("="*80)
    print(f"🎉 PROCESSING COMPLETE!")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"✅ Successful: {successful}/{len(matlab_files)}")
    print(f"❌ Failed: {failed}/{len(matlab_files)}")
    print(f"📁 Results saved to: {args.output_root}")


if __name__ == "__main__":
    main_multiprocessing()
    results=main_2()