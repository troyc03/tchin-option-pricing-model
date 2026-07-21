import numpy as np

def estimate_black_scholes_mle(prices, dt=1/252):
    """
    Computes the Maximum Likelihood Estimators for the Black-Scholes model parameters.
    
    Parameters:
    prices (array-like): Time series of historical asset prices.
    dt (float): Time step increment as a fraction of a year (e.g., 1/252 for daily trading data).
                
    Returns:
    tuple: (mu_hat, sigma_hat) annualized drift and volatility estimators.
    """
    prices = np.array(prices)
    
    # 1. Compute continuously compounded log-returns: R_t = ln(S_t) - ln(S_{t-1})
    log_returns = np.diff(np.log(prices))
    N = len(log_returns)
    
    # 2. Calculate the sample mean of log-returns (\bar{R})
    bar_R = np.mean(log_returns)
    
    # 3. Calculate Volatility Estimator (\hat{\sigma}^2) 
    # Note: MLE divides by N, not N-1 (unbiased variance would use ddof=1)
    sigma_sq_hat = np.sum((log_returns - bar_R) ** 2) / (N * dt)
    sigma_hat = np.sqrt(sigma_sq_hat)
    
    # 4. Calculate Drift Estimator (\hat{\mu})
    mu_hat = (bar_R / dt) + (0.5 * sigma_sq_hat)
    
    return mu_hat, sigma_hat

if __name__ == "__main__":
    # Define true parameters
    true_mu = 0.10       # 10% annual drift
    true_sigma = 0.20    # 20% annual volatility
    dt = 1 / 252         # Daily steps
    n_days = 1000        # Roughly 4 years of trading data
    
    # Simulate a Black-Scholes price path
    np.random.seed(42)
    prices = [100.0]     # Initial asset price
    
    for _ in range(n_days):
        random_shock = np.random.normal()
        # GBM analytical step
        next_price = prices[-1] * np.exp(
            (true_mu - 0.5 * true_sigma**2) * dt + 
            true_sigma * np.sqrt(dt) * random_shock
        )
        prices.append(next_price)
        
    # Execute MLE estimation
    mu_est, sigma_est = estimate_black_scholes_mle(prices, dt)
    
    # Print comparison results
    print("--- Black-Scholes MLE Results ---")
    print(f"Drift (Mu)      | True: {true_mu:.4%}, Estimated: {mu_est:.4%}")
    print(f"Vol (Sigma)     | True: {true_sigma:.4%}, Estimated: {sigma_est:.4%}")
