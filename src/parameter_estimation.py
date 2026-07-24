import numpy as np

def param_estimation(prices: np.ndarray, dt: float):
    np.random.seed(42)  # Seed for reproducibility
    # 1. Calculate historical log-returns
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # 2. Estimate annualized volatility (sigma) using sample standard deviation
    # ddof=1 provides an unbiased estimator for sample variance
    sample_std = np.std(log_returns, ddof=1)
    sigma_est = sample_std / np.sqrt(dt)
    
    # 3. Estimate annualized drift (r) using the sample mean
    sample_mean = np.mean(log_returns)
    r_est = (sample_mean / dt) + 0.5 * (sigma_est ** 2)
    
    return r_est, sigma_est

if __name__ == '__main__':
    # True underlying parameters
    true_r = 0.05
    true_sigma = 0.20
    t_max = 2.0         # 2 years of data
    n_steps = 504       # 252 trading days per year
    dt_step = t_max / n_steps # dt = 1/252
    
    # Simulate a GBM path to test our estimator
    t_grid = np.linspace(0, t_max, n_steps + 1)
    Z = np.random.standard_normal(n_steps)
    drift_part = (true_r - 0.5 * true_sigma**2) * dt_step
    diffusion_part = true_sigma * np.sqrt(dt_step) * Z
    
    simulated_prices = np.zeros(n_steps + 1)
    simulated_prices[0] = 100.0  # S0
    simulated_prices[1:] = simulated_prices[0] * np.exp(np.cumsum(drift_part + diffusion_part))
    
    # Run the parameter estimation function
    estimated_r, estimated_sigma = param_estimation(simulated_prices, dt=dt_step)
    
    # Display performance mapping
    print(f"--- Parameter Estimation Results ({n_steps} data points) ---")
    print(f"Drift (r):      True = {true_r:.4f} | Estimated = {estimated_r:.4f}")
    print(f"Vol (sigma):    True = {true_sigma:.4f} | Estimated = {estimated_sigma:.4f}")