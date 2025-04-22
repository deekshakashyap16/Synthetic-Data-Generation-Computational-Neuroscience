import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import loadmat
from scipy.stats import wasserstein_distance, ks_2samp
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data(real_data_path, synthetic_data_path):
    """Load both real and synthetic data with proper reshaping"""
    print(f"Loading real data from {real_data_path}")
    mat_data = loadmat(real_data_path)
    eeg_struct = mat_data['eeg'][0, 0]
    
    # Extract imagery data
    imagery_left = eeg_struct['imagery_left']
    imagery_right = eeg_struct['imagery_right']
    
    # Convert object arrays to usable numpy arrays if needed
    if isinstance(imagery_left, np.ndarray) and imagery_left.dtype == np.dtype('O'):
        imagery_left = np.array(imagery_left[0], dtype=float)
    
    if isinstance(imagery_right, np.ndarray) and imagery_right.dtype == np.dtype('O'):
        imagery_right = np.array(imagery_right[0], dtype=float)
    
    # Combine imagery data
    real_data = np.vstack((imagery_left, imagery_right))
    
    print(f"Loading synthetic data from {synthetic_data_path}")
    synthetic_data = np.load(synthetic_data_path)
    
    print(f"Original real data shape: {real_data.shape}")
    print(f"Original synthetic data shape: {synthetic_data.shape}")
    
    # Reshape synthetic data to match real data structure
    # The synthetic data shape is (5000, 128, 128)
    if len(synthetic_data.shape) == 3:
        n_samples, seq_len, n_channels = synthetic_data.shape
        # Reshape to 2D: (n_samples, seq_len * n_channels)
        synthetic_data = synthetic_data.reshape(n_samples, seq_len * n_channels)
        print(f"Reshaped synthetic data: {synthetic_data.shape}")
    
    # Take a subset of samples and features for comparison
    max_samples = min(real_data.shape[0], synthetic_data.shape[0])
    
    # For real data, we need to downsample the feature dimension significantly
    # Let's take evenly spaced points to reduce the dimensionality
    target_features = min(16384, synthetic_data.shape[1])  # Choose a reasonable size
    
    # Create evenly spaced indices for downsampling
    indices = np.linspace(0, real_data.shape[1]-1, target_features, dtype=int)
    real_data_downsampled = real_data[:max_samples, indices]
    
    # Take a subset of synthetic data to match
    synthetic_data_subset = synthetic_data[:max_samples, :target_features]
    
    print(f"Downsampled real data shape: {real_data_downsampled.shape}")
    print(f"Subset synthetic data shape: {synthetic_data_subset.shape}")
    
    return real_data_downsampled, synthetic_data_subset

def basic_statistics(real_data, synthetic_data):
    """Compare basic statistical properties"""
    print("\n=============== BASIC STATISTICS COMPARISON ===============")
    
    real_mean = np.mean(real_data)
    real_std = np.std(real_data)
    real_min = np.min(real_data)
    real_max = np.max(real_data)
    
    synth_mean = np.mean(synthetic_data)
    synth_std = np.std(synthetic_data)
    synth_min = np.min(synthetic_data)
    synth_max = np.max(synthetic_data)
    
    # Calculate percent differences
    mean_diff_pct = abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10) * 100
    std_diff_pct = abs(real_std - synth_std) / (real_std + 1e-10) * 100
    range_real = real_max - real_min
    range_synth = synth_max - synth_min
    range_diff_pct = abs(range_real - range_synth) / (range_real + 1e-10) * 100
    
    stats = {
        "Metric": ["Mean", "Standard Deviation", "Minimum", "Maximum", "Range"],
        "Real Data": [real_mean, real_std, real_min, real_max, range_real],
        "Synthetic Data": [synth_mean, synth_std, synth_min, synth_max, range_synth],
        "% Difference": [mean_diff_pct, std_diff_pct, "-", "-", range_diff_pct],
    }
    
    # Create a DataFrame for better formatting
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    
    # Calculate overall similarity score based on basic stats (lower is better)
    similarity_score = (mean_diff_pct + std_diff_pct + range_diff_pct) / 3
    
    # Cap the similarity score
    similarity_score = min(similarity_score, 100)
    
    print(f"\nBasic Statistics Similarity Score: {similarity_score:.2f}% difference")
    print(f"Similarity Rating: {get_similarity_rating(similarity_score)}")
    
    return similarity_score

def plot_distributions(real_data, synthetic_data):
    """Plot amplitude distributions of real vs synthetic data"""
    plt.figure(figsize=(12, 6))
    
    # Flatten data for distribution plots
    real_flat = real_data.flatten()
    synth_flat = synthetic_data.flatten()
    
    # Scale the data if there's a large difference in magnitude
    real_scale = np.std(real_flat)
    synth_scale = np.std(synth_flat)
    
    # If scales are very different, normalize both datasets
    scale_diff = abs(real_scale / (synth_scale + 1e-10) - 1)
    if scale_diff > 0.5:  # 50% difference in scale
        print("Normalizing data due to large scale difference")
        real_flat = (real_flat - np.mean(real_flat)) / (np.std(real_flat) + 1e-10)
        synth_flat = (synth_flat - np.mean(synth_flat)) / (np.std(synth_flat) + 1e-10)
    
    # Calculate histogram bins
    bins = np.linspace(
        min(np.min(real_flat), np.min(synth_flat)),
        max(np.max(real_flat), np.max(synth_flat)),
        100
    )
    
    # Plot histograms
    plt.hist(real_flat, bins=bins, alpha=0.5, label='Real Data', density=True)
    plt.hist(synth_flat, bins=bins, alpha=0.5, label='Synthetic Data', density=True)
    
    plt.title('Amplitude Distribution: Real vs Synthetic EEG Data')
    plt.xlabel('Amplitude')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    
    # Calculate distance between distributions
    # Use scaled data for consistency
    real_scaled = (real_flat - np.mean(real_flat)) / (np.std(real_flat) + 1e-10)
    synth_scaled = (synth_flat - np.mean(synth_flat)) / (np.std(synth_flat) + 1e-10)
    
    wd = wasserstein_distance(real_scaled, synth_scaled)
    ks_stat, ks_pval = ks_2samp(real_scaled, synth_scaled)
    
    print("\n=============== DISTRIBUTION COMPARISON ===============")
    print(f"Wasserstein Distance: {wd:.4f} (lower is better)")
    print(f"Kolmogorov-Smirnov Test: statistic={ks_stat:.4f}, p-value={ks_pval:.4e}")
    if ks_pval < 0.05:
        print("The distributions are significantly different (p < 0.05)")
    else:
        print("The distributions are not significantly different (p >= 0.05)")
    
    # Normalize Wasserstein distance to a 0-100 scale for easier interpretation
    # Empirically determined that good EEG matches typically have WD < 0.5
    wd_score = min(100, (1 - min(wd / 0.5, 1)) * 100)
    print(f"Distribution Similarity Score: {wd_score:.2f}% (higher is better)")
    print(f"Similarity Rating: {get_similarity_rating(100 - wd_score)}")
    
    plt.savefig('eeg_distribution_comparison.png')
    plt.close()
    
    return wd_score

def spectral_analysis_modified(real_data, synthetic_data, fs=256):
    """Modified spectral analysis for differently structured data"""
    print("\n=============== FREQUENCY ANALYSIS ===============")
    
    try:
        # Take a smaller subset for spectral analysis
        max_samples = min(100, real_data.shape[0], synthetic_data.shape[0])
        feature_dim = min(real_data.shape[1], synthetic_data.shape[1])
        
        # Extract data for spectral analysis
        real_subset = real_data[:max_samples, :feature_dim]
        synth_subset = synthetic_data[:max_samples, :feature_dim]
        
        # Calculate average PSD for each feature (treating columns as time series)
        avg_psd_real = np.zeros((feature_dim // 2 + 1,))
        avg_psd_synth = np.zeros((feature_dim // 2 + 1,))
        
        # Use appropriate nperseg value
        nperseg = min(256, feature_dim)
        if nperseg < 4:
            nperseg = 4
            
        for i in range(max_samples):
            f_real, psd_real = signal.welch(real_subset[i], fs=fs, nperseg=nperseg)
            f_synth, psd_synth = signal.welch(synth_subset[i], fs=fs, nperseg=nperseg)
            
            avg_psd_real += psd_real
            avg_psd_synth += psd_synth
            
        avg_psd_real /= max_samples
        avg_psd_synth /= max_samples
        
        # Plot PSD comparison
        plt.figure(figsize=(12, 6))
        plt.semilogy(f_real, avg_psd_real, label='Real Data')
        plt.semilogy(f_synth, avg_psd_synth, label='Synthetic Data')
        plt.title('Power Spectral Density: Real vs Synthetic EEG')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('eeg_psd_comparison.png')
        plt.close()
        
        # Calculate spectral difference score
        log_psd_real = np.log10(avg_psd_real + 1e-10)
        log_psd_synth = np.log10(avg_psd_synth + 1e-10)
        
        mse = np.mean((log_psd_real - log_psd_synth) ** 2)
        spectral_score = min(100, (1 - min(mse / 0.5, 1)) * 100)
        
        print(f"Spectral MSE (log scale): {mse:.4f}")
        print(f"Spectral Similarity Score: {spectral_score:.2f}% (higher is better)")
        print(f"Similarity Rating: {get_similarity_rating(100 - spectral_score)}")
        
        return spectral_score
    
    except Exception as e:
        print(f"Error in spectral analysis: {e}")
        return 0


def dimensionality_reduction(real_data, synthetic_data):
    """Use PCA to compare dimensionality and structure"""
    print("\n=============== DIMENSIONALITY ANALYSIS ===============")
    
    try:
        # Sample data for efficiency
        n_samples = min(1000, real_data.shape[0], synthetic_data.shape[0])
        real_sample = real_data[:n_samples]
        synth_sample = synthetic_data[:n_samples]
        
        # Standardize
        scaler = StandardScaler()
        real_scaled = scaler.fit_transform(real_sample)
        synth_scaled = scaler.transform(synth_sample)  # Use same scaler as real data
        
        # Apply PCA
        pca = PCA(n_components=min(10, min(real_scaled.shape[1], synth_scaled.shape[1])))
        pca.fit(real_scaled)
        real_transformed = pca.transform(real_scaled)
        synth_transformed = pca.transform(synth_scaled)
        
        # Calculate explained variance
        real_var_ratio = pca.explained_variance_ratio_
        print(f"Explained variance by first 3 components: {sum(real_var_ratio[:3]):.2%}")
        
        # Plot first two components
        plt.figure(figsize=(10, 8))
        plt.scatter(real_transformed[:, 0], real_transformed[:, 1], alpha=0.5, label='Real')
        plt.scatter(synth_transformed[:, 0], synth_transformed[:, 1], alpha=0.5, label='Synthetic')
        plt.title('PCA: First Two Principal Components')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.tight_layout()
        plt.savefig('eeg_pca_comparison.png')
        plt.close()
        
        # Calculate distribution distance in PCA space
        pca_distances = []
        for i in range(min(3, real_transformed.shape[1])):
            wd = wasserstein_distance(real_transformed[:, i], synth_transformed[:, i])
            pca_distances.append(wd)
        
        avg_pca_distance = np.mean(pca_distances)
        print(f"Average Wasserstein distance in PCA space: {avg_pca_distance:.4f}")
        
        # Normalize to a 0-100 score (empirically good matches have avg distance < 2)
        pca_score = min(100, (1 - min(avg_pca_distance / 2, 1)) * 100)
        print(f"PCA Similarity Score: {pca_score:.2f}% (higher is better)")
        print(f"Similarity Rating: {get_similarity_rating(100 - pca_score)}")
        
        return pca_score
    
    except Exception as e:
        print(f"Error in dimensionality reduction: {e}")
        return 0  # Return 0 if analysis fails

def discriminative_test(real_data, synthetic_data):
    """Train a classifier to distinguish real from synthetic data"""
    print("\n=============== DISCRIMINATIVE TEST ===============")
    
    try:
        # Sample balanced dataset
        n_samples = min(1000, real_data.shape[0], synthetic_data.shape[0])
        real_sample = real_data[:n_samples]
        synth_sample = synthetic_data[:n_samples]
        
        # Create labels (0 for real, 1 for synthetic)
        real_labels = np.zeros(real_sample.shape[0])
        synth_labels = np.ones(synth_sample.shape[0])
        
        # Combine data
        X = np.vstack((real_sample, synth_sample))
        y = np.hstack((real_labels, synth_labels))
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train a classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Classifier accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Synthetic']))
        
        # Perfect GAN would give 0.5 accuracy (can't distinguish)
        # Terrible GAN would give 1.0 accuracy (easy to distinguish)
        
        # Convert to a 0-100 score where higher means more similar
        # 0.5 accuracy (can't tell apart) = 100% similarity
        # 1.0 accuracy (perfectly distinguishable) = 0% similarity
        indistinguishability_score = (1 - abs(accuracy - 0.5) * 2) * 100
        print(f"Indistinguishability Score: {indistinguishability_score:.2f}% (higher is better)")
        print(f"Similarity Rating: {get_similarity_rating(100 - indistinguishability_score)}")
        
        return indistinguishability_score
    
    except Exception as e:
        print(f"Error in discriminative test: {e}")
        return 0  # Return 0 if analysis fails


def temporal_correlation_modified(real_data, synthetic_data):
    """Modified temporal correlation analysis for differently structured data"""
    print("\n=============== TEMPORAL CORRELATION ANALYSIS ===============")
    
    try:
        # Calculate temporal features
        def calc_temporal_features(data):
            features = []
            max_samples = min(100, data.shape[0])
            
            for i in range(max_samples):
                signal = data[i]
                
                # Calculate autocorrelation at different lags
                acfs = []
                for lag in [1, 5, 10]:
                    if lag < len(signal) - 1:
                        acf = np.corrcoef(signal[:-lag], signal[lag:])[0, 1]
                        acfs.append(acf)
                    else:
                        acfs.append(0)
                
                # Calculate standard statistics
                mean = np.mean(signal)
                std = np.std(signal)
                skew = np.mean(((signal - mean) / std) ** 3) if std > 0 else 0
                
                features.append([*acfs, skew])
            
            return np.array(features)
        
        real_features = calc_temporal_features(real_data)
        synth_features = calc_temporal_features(synthetic_data)
        
        # Plot temporal features (just first two for visualization)
        plt.figure(figsize=(10, 8))
        plt.scatter(real_features[:, 0], real_features[:, 1], alpha=0.5, label='Real')
        plt.scatter(synth_features[:, 0], synth_features[:, 1], alpha=0.5, label='Synthetic')
        plt.title('Temporal Features: Lag-1 Autocorrelation vs Lag-5 Autocorrelation')
        plt.xlabel('Lag-1 Autocorrelation')
        plt.ylabel('Lag-5 Autocorrelation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('eeg_temporal_features.png')
        plt.close()
        
        # Calculate distance between feature distributions
        feature_dists = []
        for i in range(real_features.shape[1]):
            real_col = real_features[:, i]
            synth_col = synth_features[:, i]
            
            real_col = real_col[~np.isnan(real_col)]
            synth_col = synth_col[~np.isnan(synth_col)]
            
            if len(real_col) > 0 and len(synth_col) > 0:
                wd = wasserstein_distance(real_col, synth_col)
                feature_dists.append(wd)
        
        avg_feature_dist = np.mean(feature_dists) if feature_dists else float('inf')
        print(f"Average Wasserstein distance in temporal feature space: {avg_feature_dist:.4f}")
        
        # Normalize to a 0-100 scale (empirically good matches have avg distance < 0.5)
        temporal_score = min(100, (1 - min(avg_feature_dist / 0.5, 1)) * 100)
        
        print(f"Temporal Feature Similarity Score: {temporal_score:.2f}% (higher is better)")
        print(f"Similarity Rating: {get_similarity_rating(100 - temporal_score)}")
        
        return temporal_score
    
    except Exception as e:
        print(f"Error in temporal correlation analysis: {e}")
        return 0

def get_similarity_rating(diff_percentage):
    """Convert a difference percentage to a verbal rating"""
    if diff_percentage < 5:
        return "Excellent Match"
    elif diff_percentage < 10:
        return "Very Good Match"
    elif diff_percentage < 20:
        return "Good Match"
    elif diff_percentage < 30:
        return "Fair Match"
    elif diff_percentage < 50:
        return "Poor Match"
    else:
        return "Very Poor Match"

def overall_evaluation(scores):
    """Combine all scores into an overall evaluation"""
    print("\n=============== OVERALL EVALUATION ===============")
    
    # Define weights for different metrics
    weights = {
        "Basic Statistics": 0.1,
        "Distribution Similarity": 0.2,
        "Spectral Similarity": 0.3,
        "PCA Similarity": 0.1,
        "Indistinguishability": 0.2,
        "Temporal Similarity": 0.1
    }
    
    # Calculate weighted average
    weighted_sum = sum(scores[key] * weights[key] for key in scores)
    weighted_avg = weighted_sum / sum(weights.values())
    
    print(f"\nOverall Similarity Score: {weighted_avg:.2f}% (higher is better)")
    print(f"Overall Rating: {get_similarity_rating(100 - weighted_avg)}")
    
    # Detailed breakdown
    print("\nDetailed Score Breakdown:")
    for key in scores:
        print(f"  {key}: {scores[key]:.2f}% (weight: {weights[key]:.1f})")
    
    # Conclusions and recommendations
    print("\nConclusions:")
    if weighted_avg >= 80:
        print("- The synthetic data is an excellent representation of the real EEG data")
        print("- Suitable for most EEG analysis tasks and research purposes")
    elif weighted_avg >= 70:
        print("- The synthetic data captures most characteristics of the real EEG data")
        print("- Suitable for many EEG analysis tasks with minor limitations")
    elif weighted_avg >= 60:
        print("- The synthetic data captures the general patterns but misses some details")
        print("- May be suitable for preliminary analysis but should be used with caution")
    elif weighted_avg >= 50:
        print("- The synthetic data shows moderate similarity to real EEG data")
        print("- Limited usefulness, consider improving the generation model")
    else:
        print("- The synthetic data does not adequately represent real EEG characteristics")
        print("- Not recommended for EEG analysis tasks")
        print("- Consider retraining the GAN with different parameters or architecture")
        
        # Provide specific recommendations based on scores
        print("\nRecommendations for improvement:")
        if scores["Basic Statistics"] < 50:
            print("- Adjust GAN to match the amplitude range and variance of real EEG data")
        if scores["Distribution Similarity"] < 50:
            print("- Incorporate additional loss terms to better match amplitude distributions")
        if scores["Spectral Similarity"] < 50:
            print("- Add spectral constraints or frequency-domain penalties to better match EEG rhythms")
        if scores["Temporal Similarity"] < 50:
            print("- Improve temporal dynamics with recurrent connections or attention mechanisms")
    
    return weighted_avg

def main():
    """Main function to run all comparisons"""
    # Paths to real and synthetic data
    real_data_path = "C:/Users/deeks/OneDrive/Desktop/Research/21679035/s01.mat"
    synthetic_data_path = "generated_eeg_data/synthetic_eeg_data_1.npy"
    
    # Load and preprocess data
    real_data, synthetic_data = load_data(real_data_path, synthetic_data_path)
    
    # Scale the data to similar ranges
    print("Normalizing data to comparable scales...")
    scaler = StandardScaler()
    real_data_scaled = scaler.fit_transform(real_data)
    synthetic_data_scaled = StandardScaler().fit_transform(synthetic_data)
    
    # Run all analyses on the scaled data
    scores = {}
    
    # Basic statistics
    basic_stats_score = basic_statistics(real_data_scaled, synthetic_data_scaled)
    scores["Basic Statistics"] = 100 - basic_stats_score  # Convert difference to similarity
    
    # Distribution analysis
    distribution_score = plot_distributions(real_data_scaled, synthetic_data_scaled)
    scores["Distribution Similarity"] = distribution_score
    
    # Spectral analysis - use a smaller sample and proper segment size
    spectral_score = spectral_analysis_modified(real_data_scaled, synthetic_data_scaled)
    scores["Spectral Similarity"] = spectral_score
    
    # Dimensionality analysis - already working with 2D data now
    pca_score = dimensionality_reduction(real_data_scaled, synthetic_data_scaled)
    scores["PCA Similarity"] = pca_score
    
    # Discriminative test - should work with 2D data
    indistinguish_score = discriminative_test(real_data_scaled, synthetic_data_scaled)
    scores["Indistinguishability"] = indistinguish_score
    
    # Temporal correlation - may need modification for the new data structure
    temporal_score = temporal_correlation_modified(real_data_scaled, synthetic_data_scaled)
    scores["Temporal Similarity"] = temporal_score
    
    # Overall evaluation
    overall_score = overall_evaluation(scores)
    
    print("\nAnalysis complete. Results saved as images in the current directory.")

if __name__ == "__main__":
    main()