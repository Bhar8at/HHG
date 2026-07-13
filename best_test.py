import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.fft import fft

# --- Configuration ---
DATASET_FILENAME = "hhg_dataset.pkl"
MODEL_FILENAME = "models/hhg_modelv2.keras"
CYCLES_IN_WINDOW = 10 

# Define the ranks you want 
TARGET_RANKS = [1, 500, 600, 700]

def find_specific_rank_samples():
    # 1. Load Data
    print(f"Loading dataset from {DATASET_FILENAME}...")
    try:
        with open(DATASET_FILENAME, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {DATASET_FILENAME} not found.")
        return
    
    X_raw, Y_raw = data['X'], data['Y']
    
    # 2. Re-fit Scalers
    input_scaler = StandardScaler().fit(X_raw)
    output_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(Y_raw)
    
    # 3. Load Model
    print(f"Loading model from {MODEL_FILENAME}...")
    try:
        model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
    except OSError:
        print(f"Error: {MODEL_FILENAME} not found.")
        return

    # 4. Inference
    print("Running inference...")
    X_scaled = input_scaler.transform(X_raw)
    Y_pred_scaled = model.predict(X_scaled, batch_size=64, verbose=1)
    Y_pred_phys = output_scaler.inverse_transform(Y_pred_scaled)

    # 5. Sort by MAE
    mae_per_sample = np.mean(np.abs(Y_raw - Y_pred_phys), axis=1)
    sorted_indices = np.argsort(mae_per_sample)

    print("\n" + "="*40)
    print(f"RETRIEVING SPECIFIC RANKS")
    print("="*40)

    # 6. Extract and Plot Specific Ranks
    for rank in TARGET_RANKS:
        # Check if rank is within dataset bounds
        if rank > len(sorted_indices):
            print(f"Rank {rank} is out of bounds for dataset size {len(sorted_indices)}")
            continue
            
        # Convert rank to 0-based index (e.g., 1st rank is index 0)
        idx = sorted_indices[rank - 1]
        mae = mae_per_sample[idx]
        
        print(f"Rank: {rank:3d} | Index: {idx:5d} | MAE: {mae:.6f} a.u.")
        
        plot_comparison(
            Y_raw[idx], 
            Y_pred_phys[idx], 
            idx, 
            mae, 
            f"PERFORMANCE RANK: {rank}"
        )

def plot_comparison(sig_true, sig_pred, sample_idx, mae_score, title_prefix):
    plt.figure(figsize=(12, 10))
    
    # --- Time Domain ---
    plt.subplot(2, 1, 1)
    plt.plot(sig_true, label='Ground Truth (TDSE)', color='black', alpha=0.6)
    plt.plot(sig_pred, label='AI Prediction', color='darkorange', linestyle='--')
    plt.title(f"{title_prefix}\n(Sample {sample_idx}) | MAE: {mae_score:.6f}")
    plt.xlabel("Time Index")
    plt.ylabel("Acceleration (a.u.)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- Frequency Domain ---
    def get_spectrum_data(signal):
        N = len(signal)
        window = np.hanning(N)
        spec = np.abs(fft(signal * window))**2 + 1e-15
        spec = spec[:N//2]
        orders = np.arange(len(spec)) / CYCLES_IN_WINDOW
        return orders, spec

    orders, spec_true = get_spectrum_data(sig_true)
    _, spec_pred = get_spectrum_data(sig_pred)
    
    plt.subplot(2, 1, 2)
    plt.semilogy(orders, spec_true, label='Ground Truth', color='black', alpha=0.6)
    plt.semilogy(orders, spec_pred, label='AI Prediction', color='darkorange', linestyle='--')
    plt.xlim(0, 250) 
    plt.title("Frequency Domain: HHG Spectrum")
    plt.xlabel("Harmonic Order")
    plt.ylabel("Intensity (Log Scale)")
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    find_specific_rank_samples()