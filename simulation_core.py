import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.fft import fft, ifft, fftfreq

class HHGSimulation:
    """
    A modular 1D Time-Dependent Schrödinger Equation (TDSE) solver 
    specifically designed for High Harmonic Generation (HHG).
    
    Units: Atomic Units (a.u.) are used internally.
    """
    def __init__(self, 
                 intensity_w_cm2=1e14, 
                 wavelength_nm=800, 
                 soft_core_a=1.0, 
                 pulse_duration_cycles=10,
                 grid_points=2048, 
                 grid_size=200.0,
                 dt=0.05):
        
        # --- 1. Physics Parameters (Conversion to a.u.) ---
        # 1 a.u. intensity = 3.509e16 W/cm^2
        self.I0 = intensity_w_cm2 / 3.509e16 
        self.E0 = np.sqrt(self.I0) 
        
        # 1 a.u. energy = 27.211 eV, 1 a.u. length = 0.0529 nm
        # omega (a.u.) = 45.56 / wavelength (nm)
        self.omega = 45.56 / wavelength_nm
        self.period = 2 * np.pi / self.omega
        
        self.a = soft_core_a
        
        # --- 2. Grid Setup ---
        self.N = int(grid_points)
        self.L = float(grid_size)
        self.dx = 2 * self.L / self.N
        self.x = np.linspace(-self.L, self.L, self.N, endpoint=False)
        
        # Momentum grid (for Kinetic Operator)
        self.k = fftfreq(self.N, d=self.dx) * 2 * np.pi
        
        # --- 3. Time Setup ---
        self.dt = dt
        # Total duration: ramp up + plateau + ramp down (approx)
        # We ensure the grid is long enough for the pulse
        total_time = pulse_duration_cycles * self.period
        self.n_steps = int(total_time / self.dt)
        self.time_points = np.linspace(0, total_time, self.n_steps)
        
        # Store pulse info for laser function
        self.pulse_duration = total_time
        
        # --- 4. Potential & Operators ---
        self.V_x = self._soft_coulomb(self.x)
        self.dV_dx = self._soft_coulomb_derivative(self.x)
        
        # Absorbing Boundary (CAP) - Copied logic from TDSE_classes.py
        # Smoothly absorbs wavefunction at edges to prevent reflections
        self.mask = np.ones(self.N)
        boundary_start = self.L * 0.85
        mask_idx = np.abs(self.x) > boundary_start
        self.mask[mask_idx] = np.cos(np.pi * (np.abs(self.x[mask_idx]) - boundary_start) / 
                                     (2 * (self.L - boundary_start)))**0.25
        
        # Precompute Split-Step Operators
        # Kinetic Operator: T = p^2/2 = k^2/2
        self.T_op = np.exp(-1j * (self.k**2 / 2.0) * self.dt)
        
        # Initial Wavefunction placeholder
        self.psi = None
        self.ground_state_energy = None

    def _soft_coulomb(self, x):
        """Standard Soft-Coulomb Potential"""
        return -1.0 / np.sqrt(x**2 + self.a**2)

    def _soft_coulomb_derivative(self, x):
        """Analytical derivative dV/dx for Ehrenfest theorem"""
        return x / (x**2 + self.a**2)**1.5

    def compute_ground_state(self):
        """
        Solves the Time-Independent Schrödinger Equation (TISE)
        using Finite Difference method to find the exact ground state.
        This is more accurate than guessing a Gaussian.
        """
        # Construct Hamiltonian Matrix (Finite Difference)
        # H = -1/2 d^2/dx^2 + V(x)
        
        # Laplacian (2nd order central difference)
        ones = np.ones(self.N)
        # Diagonals: [1, -2, 1] / dx^2
        D2 = sp.diags([ones, -2*ones, ones], [-1, 0, 1], shape=(self.N, self.N))
        D2 = D2 / (self.dx**2)
        
        # Kinetic Energy Matrix
        T_mat = -0.5 * D2
        
        # Potential Energy Matrix
        V_mat = sp.diags(self.V_x, 0, shape=(self.N, self.N))
        
        # Total Hamiltonian
        H_mat = T_mat + V_mat
        
        # Solve for smallest eigenvalue (Algebraic, Real)
        # k=1 gives ground state
        eigvals, eigvecs = spla.eigsh(H_mat, k=1, which='SA')
        
        self.ground_state_energy = eigvals[0]
        self.psi = eigvecs[:, 0].astype(np.complex128)
        
        # Normalize
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        self.psi /= norm
        
        # Ensure continuity (sometimes eigen-solvers flip sign)
        # We want the peak to be positive real
        center_idx = self.N // 2
        phase_corr = np.angle(self.psi[center_idx])
        self.psi *= np.exp(-1j * phase_corr)
        
        return self.ground_state_energy

    def laser_field(self, t):
        """
        Returns Electric field E(t) with a sin^2 envelope.
        """
        # Center the pulse
        t0 = self.pulse_duration / 2
        
        # Width parameters
        sigma = self.pulse_duration / 3.0 # Roughly fits 3-sigma in window
        
        # Gaussian Envelope (cleaner than sin^2 for spectral purity)
        # envelope = np.exp(-(t - t0)**2 / (2 * (sigma)**2))
        
        # Sin^2 Envelope (Matches TDSE_params.py 'sinus' style)
        if 0 <= t <= self.pulse_duration:
            envelope = np.sin(np.pi * t / self.pulse_duration)**2
        else:
            envelope = 0.0

        return self.E0 * envelope * np.sin(self.omega * (t - t0))

    def run(self):
        """
        Runs the simulation.
        Returns: 
            times (array): Time points
            dipole_acc (array): Dipole acceleration <a(t)>
        """
        if self.psi is None:
            # print("Computing Ground State first...")
            self.compute_ground_state()
            
        psi = self.psi.copy()
        dipole_acc = np.zeros(self.n_steps)
        
        # print(f"Starting Propagation ({self.n_steps} steps)...")
        
        for n in range(self.n_steps):
            t = self.time_points[n]
            E_t = self.laser_field(t)
            
            # --- Split-Step Fourier Method ---
            
            # 1. Half-step Potential (Atomic V + Interaction x*E)
            # Length Gauge Hamiltonian: H = p^2/2 + V(x) + x*E(t)
            V_total = self.V_x + (self.x * E_t)
            V_half_op = np.exp(-1j * V_total * self.dt / 2.0)
            
            psi *= V_half_op
            
            # 2. Full-step Kinetic (in Momentum Space)
            psi = fft(psi)
            psi *= self.T_op
            psi = ifft(psi)
            
            # 3. Half-step Potential
            psi *= V_half_op
            
            # 4. Absorbing Boundary
            psi *= self.mask
            
            # --- Ehrenfest Acceleration Calculation ---
            # a(t) = -<dV/dx> - E(t)
            # <dV/dx> = Integral( psi* . dV/dx . psi ) dx
            
            grad_V_expect = np.sum(np.conj(psi) * self.dV_dx * psi) * self.dx
            acc = -np.real(grad_V_expect)
            
            dipole_acc[n] = acc
            
        return self.time_points, dipole_acc

# ==========================================
# Example Usage (for testing the module)
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 1. Configure
    sim = HHGSimulation(
        intensity_w_cm2=2e14,
        wavelength_nm=800,
        pulse_duration_cycles=20,
        grid_points=4096,
        grid_size=200
    )
    
    print(f"Ground State Energy: {sim.compute_ground_state():.4f} a.u. (Should be approx -0.67 for a=0.48, or -0.5 for a=1.0)")
    
    # 2. Run
    t, acc = sim.run()
    
    # 3. Spectrum
    def get_spectrum(signal, dt):
        window = np.hanning(len(signal))
        spec = np.abs(fft(signal * window))**2
        freqs = fftfreq(len(signal), d=dt) * 2 * np.pi
        return freqs, spec

    freqs, spectrum = get_spectrum(acc, sim.dt)
    
    # Harmonic Order = Frequency / Laser Frequency
    harmonic_order = freqs / sim.omega
    
    # 4. Plot
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2,1,1)
    plt.plot(t, acc)
    plt.title("Dipole Acceleration")
    plt.xlabel("Time (a.u.)")
    
    plt.subplot(2,1,2)
    plt.semilogy(harmonic_order, spectrum)
    plt.xlim(0, 200)
    plt.ylim(1e-15, 1e5)
    plt.title("HHG Spectrum")
    plt.xlabel("Harmonic Order")
    plt.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    plt.show()