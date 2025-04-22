import os
import numpy as np
import gc
import time

def merge_eeg_files(input_folder, output_file=None, segment_length=256, batch_size=1000):
    """
    Merge multiple preprocessed EEG files for BCI training and synthetic data generation,
    producing only the segmented output format ideal for BCI and GAN training.
    
    Parameters:
    - input_folder: Folder containing preprocessed .npy files
    - output_file: Path for the merged output file (default: segmented_eeg_data.npy in input folder)
    - segment_length: Length of each data segment (default: 256 timepoints)
    - batch_size: Number of segments to process at once for memory efficiency (default: 1000)
    
    Returns:
    - Path to the segmented data file
    """
    start_time = time.time()
    
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(input_folder, "segmented_eeg_data.npy")
    
    # Get list of processed files
    processed_files = [f for f in sorted(os.listdir(input_folder)) if f.endswith("_processed.npy")]
    print(f"Found {len(processed_files)} preprocessed EEG files to merge")
    
    if not processed_files:
        print("❌ No processed EEG files found in the specified folder")
        return None
    
    # Track files with issues
    excluded_files = []
    merged_files = []
    file_shapes = []
    
    try:
        # First pass: Check shapes and consistency
        print("\n🔍 Analyzing EEG files...")
        n_channels = None
        input_is_3d = False
        segment_timepoints = None
        
        for i, file_name in enumerate(processed_files):
            file_path = os.path.join(input_folder, file_name)
            print(f"  Checking file {i+1}/{len(processed_files)}: {file_name}")
            
            try:
                # Load with memory mapping to handle large files efficiently
                data = np.load(file_path, mmap_mode='r')
                
                # Check if data is already 3D (pre-segmented)
                if len(data.shape) == 3:
                    input_is_3d = True
                    print(f"    📊 Detected 3D data with shape {data.shape}")
                    
                    # Store shape for analysis
                    file_shapes.append(data.shape)
                    
                    # Set expected dimensions
                    if n_channels is None:
                        n_channels = data.shape[0]
                        segment_timepoints = data.shape[2]
                    elif data.shape[0] != n_channels or data.shape[2] != segment_timepoints:
                        print(f"    ⚠️ Warning: Dimension mismatch - expected ({n_channels}, any, {segment_timepoints}), got {data.shape}")
                        excluded_files.append(file_name)
                        continue
                else:
                    # Original 2D data handling
                    # Store shape for analysis
                    file_shapes.append(data.shape)
                    
                    # Check channel count consistency
                    if n_channels is None:
                        n_channels = data.shape[0]
                    elif data.shape[0] != n_channels:
                        print(f"    ⚠️ Warning: Channel count mismatch - expected {n_channels}, got {data.shape[0]}")
                        excluded_files.append(file_name)
                        continue
                
                # Check for NaN or Inf values (sample check on first few rows)
                sample_size = min(100, data.shape[0]) if len(data.shape) == 2 else data.shape[0]
                sample_data = data[:sample_size] if len(data.shape) == 2 else data[:sample_size, :10]
                
                if np.isnan(sample_data).any() or np.isinf(sample_data).any():
                    print(f"    ⚠️ Warning: Sample check found NaN or Inf values - additional checks will be performed")
                
                # File seems valid for merging
                merged_files.append(file_name)
                
                del data, sample_data
                gc.collect()
                
            except Exception as e:
                print(f"    ❌ Error analyzing file: {str(e)}")
                excluded_files.append(file_name)
        
        if not merged_files:
            print("❌ No valid files to merge after quality checks")
            return None
        
        print(f"\n✅ Will process {len(merged_files)} files ({len(excluded_files)} excluded)")
        
        # Calculate total segments for pre-allocation
        if input_is_3d:
            # Count total segments across all files
            total_segments = sum(shape[1] for shape, file in zip(file_shapes, processed_files) 
                                if file in merged_files)
            print(f"  Total segments across all files: {total_segments}")
            # Use the segment length from the input files
            segment_length = segment_timepoints
        else:
            # Original calculation for 2D data
            total_timepoints = sum(shape[1] for shape, file in zip(file_shapes, processed_files) 
                                  if file in merged_files)
            print(f"  Total data points across all files: {total_timepoints}")
            total_segments = total_timepoints // segment_length
            print(f"  Will create {total_segments} segments of length {segment_length}")
        
        if total_segments == 0:
            print("❌ Not enough data points to create even one segment")
            return None
        
        # Estimate memory required for full dataset
        element_size = 4  # bytes for float32
        total_size_gb = (total_segments * n_channels * segment_length * element_size) / (1024**3)
        print(f"  Estimated dataset size: {total_size_gb:.2f} GB")
        
        # Create output file with memmap to avoid loading all data into RAM
        print("\n🔄 Creating memory-mapped output file...")
        segmented_data = np.lib.format.open_memmap(
            output_file, 
            mode='w+', 
            dtype=np.float32,
            shape=(total_segments, n_channels, segment_length)
        )
        
        # Process files based on their dimensionality
        segment_index = 0
        temp_buffer = np.zeros((n_channels, segment_length * 2), dtype=np.float32)  # Buffer for overflow data
        buffer_filled = 0  # How many timepoints in the buffer are filled
        
        for i, file_name in enumerate(merged_files):
            file_path = os.path.join(input_folder, file_name)
            print(f"  Processing file {i+1}/{len(merged_files)}: {file_name}")
            
            data = np.load(file_path)
            
            if input_is_3d:
                # For 3D data, transpose dimensions from (channels, segments, timepoints)
                # to (segments, channels, timepoints) format
                n_file_segments = data.shape[1]
                if segment_index + n_file_segments <= total_segments:
                    # Process in batches to reduce memory usage
                    batch_count = (n_file_segments + batch_size - 1) // batch_size
                    for b in range(batch_count):
                        start_idx = b * batch_size
                        end_idx = min((b + 1) * batch_size, n_file_segments)
                        segmented_data[segment_index + start_idx:segment_index + end_idx] = np.transpose(
                            data[:, start_idx:end_idx], (1, 0, 2)
                        )
                        # Force flush to disk
                        segmented_data.flush()
                    segment_index += n_file_segments
                else:
                    # Handle case where we have more segments than expected
                    remaining = total_segments - segment_index
                    # Process in batches
                    batch_count = (remaining + batch_size - 1) // batch_size
                    for b in range(batch_count):
                        start_idx = b * batch_size
                        end_idx = min((b + 1) * batch_size, remaining)
                        segmented_data[segment_index + start_idx:segment_index + end_idx] = np.transpose(
                            data[:, start_idx:end_idx], (1, 0, 2)
                        )
                        # Force flush to disk
                        segmented_data.flush()
                    segment_index = total_segments
            else:
                # Original 2D data processing
                timepoints = data.shape[1]
                
                # First fill the buffer with any remaining data
                if buffer_filled > 0:
                    space_needed = segment_length - buffer_filled
                    if timepoints >= space_needed:
                        # We have enough data to complete a segment
                        temp_buffer[:, buffer_filled:segment_length] = data[:, :space_needed]
                        
                        # Store the completed segment
                        segmented_data[segment_index] = temp_buffer[:, :segment_length].reshape(n_channels, segment_length)
                        segment_index += 1
                        
                        # Process remaining data in this file
                        remaining_points = timepoints - space_needed
                        full_segments = remaining_points // segment_length
                        
                        # Extract full segments
                        for j in range(full_segments):
                            if segment_index >= total_segments:
                                break
                            
                            start_idx = space_needed + (j * segment_length)
                            end_idx = start_idx + segment_length
                            segmented_data[segment_index] = data[:, start_idx:end_idx].reshape(n_channels, segment_length)
                            segment_index += 1
                            
                            # Periodically flush to disk
                            if j % 100 == 0:
                                segmented_data.flush()
                        
                        # Store any leftover data in the buffer
                        leftover_start = space_needed + (full_segments * segment_length)
                        leftover_size = timepoints - leftover_start
                        
                        if leftover_size > 0:
                            temp_buffer[:, :leftover_size] = data[:, leftover_start:]
                            buffer_filled = leftover_size
                        else:
                            buffer_filled = 0
                    else:
                        # Not enough data to complete a segment, just add to buffer
                        temp_buffer[:, buffer_filled:buffer_filled+timepoints] = data
                        buffer_filled += timepoints
                else:
                    # No data in buffer, process this file directly
                    full_segments = timepoints // segment_length
                    
                    # Extract full segments
                    for j in range(full_segments):
                        if segment_index >= total_segments:
                            break
                            
                        start_idx = j * segment_length
                        end_idx = start_idx + segment_length
                        segmented_data[segment_index] = data[:, start_idx:end_idx].reshape(n_channels, segment_length)
                        segment_index += 1
                        
                        # Periodically flush to disk
                        if j % 100 == 0:
                            segmented_data.flush()
                    
                    # Store any leftover data in the buffer
                    leftover_start = full_segments * segment_length
                    leftover_size = timepoints - leftover_start
                    
                    if leftover_size > 0:
                        temp_buffer[:, :leftover_size] = data[:, leftover_start:]
                        buffer_filled = leftover_size
                    else:
                        buffer_filled = 0
            
            # Progress reporting
            if segment_index > 0 and segment_index % 1000 == 0:
                print(f"    Created {segment_index}/{total_segments} segments...")
            
            del data
            gc.collect()
        
        # Ensure data is flushed to disk
        segmented_data.flush()
        
        # Check min/max values and normalize in batches
        print("\n🔄 Finding global min/max values...")
        # Calculate min/max values in batches to save memory
        data_min = float('inf')
        data_max = float('-inf')
        
        for batch_start in range(0, total_segments, batch_size):
            batch_end = min(batch_start + batch_size, total_segments)
            batch = segmented_data[batch_start:batch_end]
            
            batch_min = np.min(batch)
            batch_max = np.max(batch)
            
            data_min = min(data_min, batch_min)
            data_max = max(data_max, batch_max)
            
            del batch
            gc.collect()
        
        print(f"  Global data range: [{data_min:.3f}, {data_max:.3f}]")
        
        # Normalize in batches if needed
        if data_min != -1 or data_max != 1:
            print("\n🔄 Normalizing data to range [-1, 1] in batches...")
            
            # Scale factor and offset for normalization
            scale = 2.0 / (data_max - data_min)
            offset = -1.0 - (data_min * scale)
            
            for batch_start in range(0, total_segments, batch_size):
                batch_end = min(batch_start + batch_size, total_segments)
                print(f"  Normalizing batch {batch_start//batch_size + 1}/{(total_segments+batch_size-1)//batch_size}...")
                
                # Load batch
                batch = segmented_data[batch_start:batch_end]
                
                # Normalize using pre-computed scale/offset (memory efficient)
                batch = batch * scale + offset
                
                # Write back
                segmented_data[batch_start:batch_end] = batch
                segmented_data.flush()
                
                del batch
                gc.collect()
            
            # Verify normalization by checking a sample
            sample_batch = segmented_data[:min(100, total_segments)]
            new_min = np.min(sample_batch)
            new_max = np.max(sample_batch)
            print(f"  Sample normalized data range: [{new_min:.3f}, {new_max:.3f}]")
            del sample_batch
        else:
            print("  Data already in optimal range [-1, 1]")
        
        # Write back the segmented_data
        print("\n✅ Dataset created and saved to disk")
        
        # Save list of included and excluded files for reference
        included_file = os.path.join(os.path.dirname(output_file), "included_eeg_files.txt")
        with open(included_file, "w") as f:
            for file_name in merged_files:
                f.write(f"{file_name}\n")
        
        if excluded_files:
            excluded_file = os.path.join(os.path.dirname(output_file), "excluded_eeg_files.txt")
            with open(excluded_file, "w") as f:
                for file_name in excluded_files:
                    f.write(f"{file_name}\n")
        
        # Close the memmap file
        del segmented_data
        gc.collect()
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Successfully created segmented dataset from {len(merged_files)} EEG files")
        print(f"✅ Created {segment_index} segments of length {segment_length}")
        print(f"✅ Final data shape: [{segment_index}, {n_channels}, {segment_length}]")
        print(f"✅ Processing completed in {elapsed_time:.1f} seconds")
        print(f"✅ Output saved to: {output_file}")
        
        if excluded_files:
            print(f"⚠️ {len(excluded_files)} files were excluded - see excluded_eeg_files.txt for details")
        
        return output_file
    
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set your input folder containing the preprocessed EEG files
    input_folder = "C:/Users/deeks/OneDrive/Desktop/Research/Preprocessed"
    
    # Set batch size for memory-efficient processing
    # Decrease this value if you still encounter memory issues
    batch_size = 500
    
    # Merge the data - focusing only on creating segmented output
    merged_file = merge_eeg_files(input_folder, batch_size=batch_size)
    
    if merged_file:
        print("\n🧠 Next steps for your BCI project:")
        print("1. Use the segmented data to train your synthetic EEG generator")
        print("2. Generate synthetic samples to augment your training data")
        print("3. Train your right-hand movement classifier")
        print("4. Connect the classifier to your Arduino setup")
    else:
        print("❌ Data merging failed - please check the error messages above")