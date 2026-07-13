import tensorflow as tf
from tensorflow.keras import layers, models


def build_physics_informed_hhg_model(input_dim=5, output_length=4096):
    """
    input_dim=5: Intensity, Wavelength, Ip (Ionization Potential), 
                 Up (Ponderomotive Energy), Theoretical_Cutoff
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # Initial expansion
    x = layers.Dense(512, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Residual Blocks to maintain gradient flow and sharpness
    for _ in range(3):
        # 1. Store the input to this block as the shortcut
        # If the dimension of x is already 512, you don't strictly need the Dense layer, 
        # but keeping it for a "projection" shortcut is fine.
        shortcut = layers.Dense(512)(x) 
        
        # 2. Main branch
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(512)(x)
        
        # 3. Add shortcut and main branch
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        # 4. Apply Dropout LAST and re-assign to x
        x = layers.Dropout(0.1)(x) 

    # Final mapping to spectrum
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dense(2048, activation='relu')(x)
    outputs = layers.Dense(output_length, activation='linear')(x) 
    
    return models.Model(inputs, outputs)