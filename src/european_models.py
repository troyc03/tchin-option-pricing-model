import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class EuropeanOptionModel:
    """Class to price European options using numerical and analytical methods."""
    
    def __init__(self, s0: float, k: float, t: float, r: float, sigma: float):
        self.s0 = s0
        self.k = k
        self.t = t
        self.r = r
        self.sigma = sigma

    def black_scholes_analytical(self) -> float:
        """Calculates the exact closed-form Black-Scholes price for a Call option."""
        d1 = (np.log(self.s0 / self.k) + (self.r + 0.5 * self.sigma**2) * self.t) / (self.sigma * np.sqrt(self.t))
        d2 = d1 - self.sigma * np.sqrt(self.t)
        return self.s0 * norm.cdf(d1) - self.k * np.exp(-self.r * self.t) * norm.cdf(d2)

    def black_scholes_pde(self, s_max: float, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solves Black-Scholes PDE using an explicitly stabilized finite difference scheme."""
        dt_max = 1.0 / (self.sigma**2 * nx**2 + self.r)
        dt = 0.9 * dt_max  
        nt = int(self.t / dt) + 1
        dt = self.t / nt   
        
        s_values = np.linspace(0, s_max, nx + 1)
        t_values = np.linspace(0, self.t, nt + 1)
        V = np.zeros((nt + 1, nx + 1))
        
        V[0, :] = np.maximum(s_values - self.k, 0)
        
        i = np.arange(1, nx)
        a = 0.5 * dt * (self.sigma**2 * i**2 - self.r * i)
        b = 1 - dt * (self.sigma**2 * i**2 + self.r)
        c = 0.5 * dt * (self.sigma**2 * i**2 + self.r * i)

        for n in range(nt):
            V[n + 1, 1:nx] = a * V[n, 0:nx-1] + b * V[n, 1:nx] + c * V[n, 2:nx+1]
            V[n + 1, 0] = 0.0  
            V[n + 1, nx] = s_max - self.k * np.exp(-self.r * t_values[n + 1]) 

        return t_values, s_values, V

    def monte_carlo(self, m: int, n_paths: int) -> tuple[np.ndarray, float, float]:
        """Simulates asset paths using Geometric Brownian Motion."""
        dt = self.t / m
        Z = np.random.standard_normal((m, n_paths))
        S = np.zeros((m + 1, n_paths))
        
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z
        S[0, :] = self.s0
        S[1:] = self.s0 * np.exp(np.cumsum(drift + diffusion, axis=0))
        
        st = S[-1, :]
        h = np.maximum(st - self.k, 0)
        
        option_value = np.exp(-self.r * self.t) * np.mean(h)
        std_dev_payoff = np.std(h, ddof=1)
        standard_error = (np.exp(-self.r * self.t) * std_dev_payoff) / np.sqrt(n_paths)
        
        return S, option_value, standard_error

def main():
    s0, k, t, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.2
    model = EuropeanOptionModel(s0=s0, k=k, t=t, r=r, sigma=sigma)
    
    # 1. Analytical Evaluation
    bs_analytical_price = model.black_scholes_analytical()

    # 2. PDE Numerical Evaluation
    s_max, nx = 200.0, 200
    t_vals, s_vals, V = model.black_scholes_pde(s_max=s_max, nx=nx)
    T, S_mesh = np.meshgrid(t_vals, s_vals)

    # Find the index closest to S0 in our numerical pricing array
    s0_idx = np.argmin(np.abs(s_vals - s0))
    bs_numerical_price = V[-1, s0_idx]

    # 3. Monte Carlo Simulation
    m, n_paths = 252, 1000
    S_paths, mc_val, std_err = model.monte_carlo(m=m, n_paths=n_paths)
    time_grid = np.linspace(0, t, m + 1)

    # Terminal Outputs
    print(f"Black-Scholes Analytical Option Price:  {bs_analytical_price:.4f}")
    print(f"Black-Scholes Numerical Option Price:   {bs_numerical_price:.4f}")
    print(f"Monte Carlo Option Price:        {mc_val:.4f} ± {std_err:.4f}")

    # Plotting code remains identical
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(T, S_mesh, V.T, cmap='viridis', edgecolor='none')
    ax1.set_title('Black-Scholes Option Price Surface')
    ax1.set_xlabel('Time to Maturity ($T$)')
    ax1.set_ylabel('Stock Price ($S$)')
    ax1.set_zlabel('Option Price ($V$)')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

    ax2 = fig.add_subplot(122)
    ax2.plot(time_grid, S_paths, alpha=0.4)
    ax2.set_title(f'Geometric Brownian Motion ({n_paths} Paths)')
    ax2.set_xlabel('Time ($t$)')
    ax2.set_ylabel('Asset Price ($S$)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
