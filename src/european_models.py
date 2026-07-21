import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- 1. GBM Path Simulation ---
def simulate_gbm_paths(S0, T, r, sigma, simulations=10, N=1000):
    dt = T / N
    t = np.linspace(0, T, N + 1)
    # Generate random paths
    Z = np.random.standard_normal((N, simulations))
    # Calculate geometric progression
    paths = np.zeros((N + 1, simulations))
    paths[0] = S0
    for i in range(1, N + 1):
        paths[i] = paths[i-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i-1])
    return t, paths

# --- 2. Monte Carlo Simulation ---
def monte_carlo(S0, K, T, r, sigma, simulations=100000):
    Z = np.random.standard_normal(simulations)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return np.exp(-r * T) * np.mean(np.maximum(ST - K, 0))

# --- 3. Black-Scholes PDE ---
def black_scholes_pde(S_max, K, T, r, sigma, M=100, N=1000):
    dt, dS = T / N, S_max / M
    S = np.linspace(0, S_max, M + 1)
    t = np.linspace(0, T, N + 1)
    V = np.zeros((M + 1, N + 1))
    
    # Boundary/Terminal Conditions
    V[:, -1] = np.maximum(S - K, 0)
    V[-1, :] = S_max - K * np.exp(-r * (T - t))
    
    # Explicit Finite Difference Solver
    a = 0.5 * dt * (sigma**2 * (np.arange(0, M + 1)**2) - r * np.arange(0, M + 1))
    b = 1.0 - dt * (sigma**2 * (np.arange(0, M + 1)**2) + r)
    c = 0.5 * dt * (sigma**2 * (np.arange(0, M + 1)**2) + r * np.arange(0, M + 1))
    
    for j in range(N - 1, -1, -1):
        for i in range(1, M):
            V[i, j] = a[i] * V[i-1, j+1] + b[i] * V[i, j+1] + c[i] * V[i+1, j+1]
            
    return S, t, V

# --- 4. Analytical Model ---
def black_scholes_model(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

if __name__ == "__main__":
    # Parameters
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    
    # Results
    print(f"MC Price: {monte_carlo(S0, K, T, r, sigma):.4f}")
    print(f"BS Price: {black_scholes_model(S0, K, T, r, sigma):.4f}")
    
    # Figure 1: GBM Paths
    t_gbm, paths = simulate_gbm_paths(S0, T, r, sigma, simulations=10)
    plt.figure(figsize=(10, 5))
    plt.plot(t_gbm, paths)
    plt.axhline(K, color='red', linestyle='--', label=f'Strike Price (K={K})')
    plt.title("Geometric Brownian Motion (GBM) Simulated Paths")
    plt.xlabel("Time (t)")
    plt.ylabel("Stock Price (S)")
    plt.legend()
    plt.grid(True)
    
    # Figure 2: FDM Visualization
    S_arr, t_arr, V_grid = black_scholes_pde(200, K, T, r, sigma)
    S_m, t_m = np.meshgrid(t_arr, S_arr)
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(t_m, S_m, V_grid, cmap='viridis')
    ax.set_title("Black-Scholes Surface")
    ax.set_xlabel("Time (T-t)")
    ax.set_ylabel("Stock Price (S)")
    ax.set_zlabel("Option Value (V)")
    
    plt.show()
