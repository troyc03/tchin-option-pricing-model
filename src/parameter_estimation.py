import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def param_estimation(prices: np.ndarray, dt: float):
    # Your original estimation logic preserved exactly
    log_returns = np.log(prices[1:] / prices[:-1])
    sample_std = np.std(log_returns, ddof=1)
    sigma_est = sample_std / np.sqrt(dt)
    sample_mean = np.mean(log_returns)
    r_est = (sample_mean / dt) + 0.5 * (sigma_est ** 2)
    return r_est, sigma_est

def run_mcmc_inverse(log_returns: np.ndarray, dt: float, iterations=20000, burn_in=4000):
    """Solves the Black-Scholes inverse problem via Metropolis MCMC."""
    n = len(log_returns)
    chain_r = np.zeros(iterations)
    chain_sigma = np.zeros(iterations)
    
    # Initialize away from truth to track convergence path
    current_r, current_sigma = 0.25, 0.45
    
    # Tuned proposal step sizes
    prop_std_r, prop_std_sigma = 0.08, 0.012
    
    def log_prior(r, sigma):
        return 0.0 if (-1.0 < r < 1.0 and 0.001 < sigma < 1.5) else -np.inf
        
    def log_likelihood(r, sigma):
        step_mu = (r - 0.5 * (sigma**2)) * dt
        step_var = (sigma**2) * dt
        sq_errors = np.sum((log_returns - step_mu)**2)
        return -0.5 * n * np.log(2 * np.pi * step_var) - (sq_errors / (2 * step_var))

    for i in range(iterations):
        prop_r = current_r + np.random.normal(0, prop_std_r)
        prop_sigma = current_sigma + np.random.normal(0, prop_std_sigma)
        
        if log_prior(prop_r, prop_sigma) == -np.inf:
            chain_r[i], chain_sigma[i] = current_r, current_sigma
            continue
            
        log_acc = (log_prior(prop_r, prop_sigma) + log_likelihood(prop_r, prop_sigma)) - \
                  (log_prior(current_r, current_sigma) + log_likelihood(current_r, current_sigma))
        
        if np.log(np.random.uniform(0, 1)) < log_acc:
            current_r, current_sigma = prop_r, prop_sigma
            
        chain_r[i], chain_sigma[i] = current_r, current_sigma
        
    return chain_r, chain_sigma

if __name__ == '__main__':
    np.random.seed(42) 
    true_r, true_sigma, t_max, n_steps = 0.05, 0.20, 2.0, 504
    dt_step = t_max / n_steps
    
    t_grid = np.linspace(0, t_max, n_steps + 1)
    Z = np.random.standard_normal(n_steps)
    drift_part = (true_r - 0.5 * true_sigma**2) * dt_step
    diffusion_part = true_sigma * np.sqrt(dt_step) * Z
    simulated_prices = np.zeros(n_steps + 1)
    simulated_prices[0] = 100.0 
    simulated_prices[1:] = simulated_prices[0] * np.exp(np.cumsum(drift_part + diffusion_part))
    
    # Calculate returns and point estimates
    log_returns = np.log(simulated_prices[1:] / simulated_prices[:-1])
    estimated_r, estimated_sigma = param_estimation(simulated_prices, dt=dt_step)
    
    # 2. GENERATE MCMC POSTERIOR HISTORY
    iterations, burn_in = 25000, 5000
    chain_r, chain_sigma = run_mcmc_inverse(log_returns, dt_step, iterations, burn_in)
    post_r, post_sigma = chain_r[burn_in:], chain_sigma[burn_in:]

    # 3. COMPREHENSIVE COMPONENT VISUALIZATION
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Drift (r) Parameter Convergence History
    ax = axes[0, 0]
    ax.plot(chain_r, color='royalblue', alpha=0.6, linewidth=0.5, label='MCMC Path ($r$)')
    ax.axhline(true_r, color='red', linestyle='--', linewidth=2, label=f'True $r$ = {true_r:.4f}')
    ax.axhline(estimated_r, color='purple', linestyle='-.', linewidth=2, label=f'MLE $\hat{{r}}$ = {estimated_r:.4f}')
    ax.axvline(burn_in, color='black', linestyle=':', linewidth=2, label='Burn-in Cutoff')
    ax.set_xlabel('Iteration (Epoch)')
    ax.set_ylabel('Drift Parameter $r$')
    ax.set_title('Drift ($r$) Chain Convergence History')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Drift (r) Marginal Posterior Distribution
    ax = axes[0, 1]
    ax.hist(post_r, bins=50, density=True, alpha=0.6, color='royalblue', edgecolor='navy')
    x_r = np.linspace(np.mean(post_r) - 3*np.std(post_r), np.mean(post_r) + 3*np.std(post_r), 200)
    ax.plot(x_r, stats.norm.pdf(x_r, np.mean(post_r), np.std(post_r)), 'r-', linewidth=2, label='Posterior Fit')
    ax.axvline(true_r, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Drift Value $r$')
    ax.set_ylabel('Probability Density')
    ax.set_title('Drift ($r$) Posterior Uncertainty Bounds')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Volatility (σ) Parameter Convergence History
    ax = axes[1, 0]
    ax.plot(chain_sigma, color='orange', alpha=0.6, linewidth=0.5, label='MCMC Path ($\sigma$)')
    ax.axhline(true_sigma, color='red', linestyle='--', linewidth=2, label=f'True $\sigma$ = {true_sigma:.4f}')
    ax.axhline(estimated_sigma, color='purple', linestyle='-.', linewidth=2, label=f'MLE $\hat{{\sigma}}$ = {estimated_sigma:.4f}')
    ax.axvline(burn_in, color='black', linestyle=':', linewidth=2, label='Burn-in Cutoff')
    ax.set_xlabel('Iteration (Epoch)')
    ax.set_ylabel('Volatility Parameter $\sigma$')
    ax.set_title('Volatility ($\sigma$) Chain Convergence History')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 4: Volatility (σ) Marginal Posterior Distribution
    ax = axes[1, 1]
    ax.hist(post_sigma, bins=50, density=True, alpha=0.6, color='darkorange', edgecolor='saddlebrown')
    x_s = np.linspace(np.mean(post_sigma) - 3*np.std(post_sigma), np.mean(post_sigma) + 3*np.std(post_sigma), 200)
    ax.plot(x_s, stats.norm.pdf(x_s, np.mean(post_sigma), np.std(post_sigma)), 'r-', linewidth=2, label='Posterior Fit')
    ax.axvline(true_sigma, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Volatility Value $\sigma$')
    ax.set_ylabel('Probability Density')
    ax.set_title('Volatility ($\sigma$) Posterior Uncertainty Bounds')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    # Generate a 2D hexbin plot to see density intersections
    plt.hexbin(post_r, post_sigma, gridsize=40, cmap='YlOrRd', mincnt=1)
    cb = plt.colorbar(label='Sample Density Count')
    
    # Overlay true parameters vs MLE estimates
    plt.plot(true_r, true_sigma, 'ro', markersize=10, label=f'True Parameters ({true_r}, {true_sigma})')
    plt.plot(estimated_r, estimated_sigma, 'm*', markersize=12, label=f'MLE Estimates ({estimated_r:.4f}, {estimated_sigma:.4f})')
    
    plt.xlabel('Drift Parameter ($r$)')
    plt.ylabel('Volatility Parameter ($\sigma$)')
    plt.title('2D Joint Posterior Distribution Surface')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.2)
    plt.show()

    # 4. COMPUTE STATISTICAL ACCURACY METRICS
    print(f"\nBAYESIAN INVERSE PROBLEM SOLUTION PERFORMANCE")
    print(f"="*50)
    print(f"Drift (r) Post Mean  : {np.mean(post_r):.6f} | 95% CI: [{np.percentile(post_r, 2.5):.4f}, {np.percentile(post_r, 97.5):.4f}]")
    print(f"Vol (sigma) Post Mean: {np.mean(post_sigma):.6f} | 95% CI: [{np.percentile(post_sigma, 2.5):.4f}, {np.percentile(post_sigma, 97.5):.4f}]")
