import numpy as np
import pandas as pd
import os 
from simulation_core import get_hhg_spectrum
import time
from scipy.signal import find_peaks

def get_harmonic_intensity(harmonic_order_array, spectrum, target_order, width=0.5):
    """
    Finds the peak intensity of a specific harmonic.
    """
    try:
        mask = (harmonic_order_array > target_order - width) & (harmonic_order_array < target_order + width)
        if np.any(mask):
            return np.max(spectrum[mask])
        else:
            # Fallback: find closest point
            idx = np.argmin(np.abs(harmonic_order_array - target_order))
            return spectrum[idx]
    except Exception:
        return 0

def extract_spectrum_features(frequencies, spectrum, omega0, threshold_factor=1e-3):
    """
    Extracts key features from an HHG spectrum to use as ML inputs.
    
    Returns:
        A dictionary of features.
    """
    features = {
        'cutoff_order': 0,
        'plateau_intensity_mean': 0,
        'H5_intensity': 0,
        'H11_intensity': 0,
        'H21_intensity': 0
    }
    
    try:
        harmonic_order = frequencies / omega0
        
        # --- 1. Find Plateau and Cutoff ---
        plateau_mask = (harmonic_order > 5) & (harmonic_order < 15)
        if not np.any(plateau_mask):
             plateau_mask = (harmonic_order > 1) # Fallback
        
        if not np.any(plateau_mask):
            return features # Not enough data

        # Find peaks in the plateau for a stable intensity reference
        peaks, _ = find_peaks(spectrum[plateau_mask], height=np.mean(spectrum[plateau_mask])/10)
        if len(peaks) == 0:
            plateau_intensity_ref = np.max(spectrum[plateau_mask])
        else:
            plateau_intensity_ref = np.mean(spectrum[plateau_mask][peaks])
            
        cutoff_threshold = plateau_intensity_ref * threshold_factor
        
        # Find highest harmonic above the threshold
        above_threshold_mask = spectrum > cutoff_threshold
        if not np.any(above_threshold_mask):
            features['cutoff_order'] = 0
        else:
            features['cutoff_order'] = int(np.max(harmonic_order[above_threshold_mask]))
            
        # --- 2. Calculate Mean Plateau Intensity ---
        # Use a wider mask for the mean intensity calculation
        full_plateau_mask = (harmonic_order > 5) & (harmonic_order < features['cutoff_order'] * 0.8)
        if np.any(full_plateau_mask):
            features['plateau_intensity_mean'] = np.mean(spectrum[full_plateau_mask])
        else:
            features['plateau_intensity_mean'] = np.mean(spectrum[plateau_mask]) # fallback

        # --- 3. Get Specific Harmonic Intensities ---
        features['H5_intensity'] = get_harmonic_intensity(harmonic_order, spectrum, 5)
        features['H11_intensity'] = get_harmonic_intensity(harmonic_order, spectrum, 11)
        features['H21_intensity'] = get_harmonic_intensity(harmonic_order, spectrum, 21)
        
        # Log-transform intensities, as they span many orders of magnitude
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-30
        for key in ['plateau_intensity_mean', 'H5_intensity', 'H11_intensity', 'H21_intensity']:
            features[key] = np.log10(features[key] + epsilon)
            
        return features
        
    except Exception as e:
        print(f"Error in extract_spectrum_features: {e}")
        return features # Return zeros

# --- Main Data Generation Loop ---

def generate_dataset(num_samples, output_file='hhg_inverse_data.csv'):
    """
    Runs the simulation `num_samples` times with randomized
    parameters and saves the [features, parameters] to a CSV file.
    """
    
    # Define parameter ranges to sample from (Why these ranges ?)
    E0_range = (0.05, 0.1)
    omega0_range = (0.04, 0.07)
    
    results = []
    
    print(f"Starting data generation for {num_samples} samples...")
    start_time = time.time()
    
    for i in range(num_samples):
        print(f"\n--- Running Sample {i+1}/{num_samples} ---")
        
        # 1. Sample random parameters (These are our TARGETS now)
        E0_sample = np.random.uniform(*E0_range)
        omega0_sample = np.random.uniform(*omega0_range)
        
        # 2. Run simulation
        try:
            frequencies, spectrum, omega0 = get_hhg_spectrum(
                E0=E0_sample, 
                omega0=omega0_sample, 
                n_cycles=1.5,
                N=1000,
                L=100,
                dt=0.1
            )
            
            # 3. Analyze output: Extract features (These are our INPUTS now)
            features_dict = extract_spectrum_features(frequencies, spectrum, omega0)
            
            if features_dict['cutoff_order'] > 0:
                # 4. Store the [inputs, output] pair
                sample_data = {
                    # Outputs (Targets)
                    'E0': E0_sample,
                    'omega0': omega0_sample,
                    # Inputs (Features)
                    **features_dict
                }
                results.append(sample_data)
                print(f"Sample {i+1} complete. Cutoff: {features_dict['cutoff_order']:.0f}")
            else:
                print(f"Sample {i+1} failed (features=0). Skipping.")

        except np.linalg.LinAlgError:
            print(f"Sample {i+1} failed (Singular matrix). Skipping.")
        except Exception as e:
            print(f"Sample {i+1} failed with unexpected error: {e}. Skipping.")
            

    # 5. Save all results to a CSV
    df = pd.DataFrame(results)

    # Check if the output file already exists
    file_exists = os.path.exists(output_file)

    if file_exists:
        # If it exists, append the data without writing the header
        print(f"File exists. Appending {len(results)} new samples...")
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        # If it doesn't exist, create it and write the header
        print(f"Creating new file. Saving {len(results)} valid samples...")
        df.to_csv(output_file, mode='w', header=True, index=False)

    end_time = time.time()

    print(f"\n--- Data generation complete ---")
    if file_exists:
        print(f"Appended {len(results)} new samples to {output_file}")
    else:
        print(f"Saved {len(results)} valid samples to new file {output_file}")

    print(f"Total time: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    # Start with a small number like `num_samples=20` to test.
    # For a good model, you will need 100s or 1000s of samples.
    generate_dataset(num_samples=1000)