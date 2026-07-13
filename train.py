import tensorflow as tf
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from model import build_physics_informed_hhg_model

# --- Configuration ---
DATASET_FILENAME = "hhg_dataset.pkl"
MODEL_FILENAME = "models/hhg_modelv2.keras" # Save as new model
BATCH_SIZE = 32
EPOCHS = 500

# --- 1. THE PHYSICS-INFORMED LOSS FUNCTION ---
def spectral_loss(y_true, y_pred):
    """
    Combines standard Time-Domain MSE with Frequency-Domain MSE.
    The frequency component is Log-Scaled to force the model to 
    learn orders of magnitude (cutoff), not just linear amplitude.
    """
    
    # A. Time Domain Loss (Standard MSE)
    # Good for phase and general shape
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # B. Frequency Domain Loss (Physics)
    # 1. Ensure inputs are float32 for RFFT (Real Fast Fourier Transform)
    # ERROR FIX: tf.signal.rfft expects floats, NOT complex numbers.
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    
    # 2. Compute FFT (Real-valued input -> Complex spectrum)
    fft_true = tf.signal.rfft(y_true_f)
    fft_pred = tf.signal.rfft(y_pred_f)
    
    # 3. Compute Magnitude (Spectrum)
    # Add epsilon 1e-10 to prevent log(0)
    spec_true = tf.abs(fft_true) + 1e-10
    spec_pred = tf.abs(fft_pred) + 1e-10
    
    # 4. Log-Scale the Spectrum
    # This is the KEY step. It makes 10^-10 visible to the loss function.
    log_spec_true = tf.math.log(spec_true)
    log_spec_pred = tf.math.log(spec_pred)
    
    # 5. Calculate MSE on the Log-Spectrum
    freq_loss = tf.reduce_mean(tf.square(log_spec_true - log_spec_pred))
    
    # C. Combine
    # alpha weights the importance of spectral accuracy.
    # Start with 0.1. If cutoff is still bad, increase to 0.5 or 1.0
    alpha = 1
    return mse_loss + (alpha * freq_loss)

def train_model():
    # --- 2. Load and Prepare Data (Same as before) ---
    print(f"Loading {DATASET_FILENAME}...")
    try:
        with open(DATASET_FILENAME, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Dataset not found. Please ensure hhg_dataset.pkl exists.")
        return

    X = data['X']
    Y = data['Y']

    # Scale Data
    input_scaler = StandardScaler()
    X_scaled = input_scaler.fit_transform(X)
    
    output_scaler = MinMaxScaler(feature_range=(-1, 1))
    Y_scaled = output_scaler.fit_transform(Y)

    # Split
    X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

    # --- 3. Build Model ---
    # Updated to use Input layer explicitly to avoid UserWarning
    model = build_physics_informed_hhg_model(X_train.shape[1], Y_train.shape[1])

    # --- 4. Compile with Custom Loss ---
    print("Compiling with Spectral Loss...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=spectral_loss,  # <--- WE USE OUR CUSTOM FUNCTION HERE
        metrics=['mae', 'mse']
    )

    # --- 5. Train ---
    print("Starting training...")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    # --- 6. Save ---
    # Note: When loading this model later, you must pass custom_objects={'spectral_loss': spectral_loss}
    model.save(MODEL_FILENAME)
    print(f"Model saved to {MODEL_FILENAME}")

if __name__ == "__main__":
    train_model()