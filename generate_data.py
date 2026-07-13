import numpy as np
import pickle
import multiprocessing
from tqdm import tqdm
from simulation_core import HHGSimulation

# --- Configuration ---
NUM_SAMPLES = 20000       # Adjust as needed (start small for testing)
OUTPUT_FILENAME = "hhg_dataset.pkl"
FIXED_OUTPUT_SIZE = 4096 # The Neural Network needs a fixed input size


PARAM_RANGES = {
    # CHANGE THIS: 0.05 is too weak. Start from 0.08 up to 0.16
    'E0': (0.08, 0.16),       
    
    'omega0': (0.04, 0.07),   
    
    # CHANGE THIS: Restrict 'a' so the well isn't too deep
    'a': (0.8, 1.5)           
}

def run_single_simulation(params):
    """
    Worker function to run one simulation.
    """
    E0, omega0, a = params
    
    # --- 1. Unit Conversion (a.u. -> Physics) ---
    # Intensity I = E^2 * 3.509e16 W/cm^2
    intensity = (E0**2) * 3.509e16
    
    # Wavelength lambda = 45.56 / omega [nm]
    wavelength = 45.56 / omega0
    
    # --- 2. Setup Simulation ---
    # We use the modular core we just wrote
    try:
        sim = HHGSimulation(
            intensity_w_cm2=intensity,
            wavelength_nm=wavelength,
            soft_core_a=a,
            pulse_duration_cycles=10, # Keep pulse length consistent in cycles
            grid_points=4096,
            grid_size=200.0,
            dt=0.05
        )
        
        # --- 3. Run Dynamics ---
        # The class handles Ground State + Propagation automatically
        times, dipole_acc = sim.run()
        
        # --- 4. Interpolation (CRITICAL) ---
        # Since omega varies, the total time (10 cycles) varies.
        # This means len(dipole_acc) will be different for every sample.
        # Neural Networks REQUIRE fixed input size. We must interpolate.
        
        # Create a normalized time axis (0 to 1) for the simulation result
        t_normalized_sim = np.linspace(0, 1, len(dipole_acc))
        
        # Create the target fixed-size grid (0 to 1)
        t_fixed = np.linspace(0, 1, FIXED_OUTPUT_SIZE)
        
        # Interpolate
        dipole_interpolated = np.interp(t_fixed, t_normalized_sim, dipole_acc)
        
        return ([E0, omega0, a], dipole_interpolated)
        
    except Exception as e:
        print(f"Simulation failed for {params}: {e}")
        return None

def generate_dataset():
    # 1. Generate Random Parameters
    print(f"Generating {NUM_SAMPLES} random parameter sets...")
    params_list = []
    for _ in range(NUM_SAMPLES):
        p = [
            np.random.uniform(*PARAM_RANGES['E0']),
            np.random.uniform(*PARAM_RANGES['omega0']),
            np.random.uniform(*PARAM_RANGES['a'])
        ]
        params_list.append(p)

    # 2. Run in Parallel
    # Use roughly 80% of available cores to avoid freezing your PC
    num_workers = max(1, int(multiprocessing.cpu_count() * 0.8))
    print(f"Starting simulation pool with {num_workers} workers...")
    
    valid_results = []
    
    # Using 'spawn' context is safer for heavy numeric libraries on some OSs
    # but standard Pool is usually fine for NumPy
    with multiprocessing.Pool(processes=num_workers) as pool:
        # tqdm shows a progress bar
        results = list(tqdm(pool.imap(run_single_simulation, params_list), total=NUM_SAMPLES))
        
    # Filter out any failed runs (None)
    for r in results:
        if r is not None:
            valid_results.append(r)
            
    print(f"Successfully generated {len(valid_results)} samples.")

    # 3. Save Data
    if len(valid_results) > 0:
        X = np.array([r[0] for r in valid_results])
        Y = np.array([r[1] for r in valid_results])
        
        data_dict = {
            'X': X, 
            'Y': Y, 
            'ranges': PARAM_RANGES,
            'output_size': FIXED_OUTPUT_SIZE
        }
        
        with open(OUTPUT_FILENAME, 'wb') as f:
            pickle.dump(data_dict, f)
        
        print(f"Dataset saved to {OUTPUT_FILENAME}")
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
    else:
        print("No valid data generated.")

if __name__ == "__main__":
    # Windows/Mac requires this protection for multiprocessing
    multiprocessing.freeze_support() 
    generate_dataset()