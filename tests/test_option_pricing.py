import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from scipy.stats import norm
from src.call_option_models import CallOptionModel

class TestCallOptionModel(unittest.TestCase):
    def setUp(self):
        """Set up standard ATM parameters for testing."""
        self.s0 = 100.0
        self.k = 100.0
        self.t = 0.5
        self.r = 0.05
        self.sigma = 0.2
        self.model = CallOptionModel(
            s0=self.s0, k=self.k, t=self.t, r=self.r, sigma=self.sigma
        )

    def test_analytical_price(self):
        """Verify analytical pricing matches known benchmark value."""
        price = self.model.black_scholes_analytical()
        # Benchmark value for S=100, K=100, T=0.5, r=0.05, sigma=0.2
        expected_price = 6.8887
        self.assertAlmostEqual(price, expected_price, places=4)

    def test_greeks_boundaries(self):
        """Verify Greeks fall within valid theoretical limits."""
        greeks = self.model.calculate_greeks()
        
        # Delta for a European call must fall between 0 and 1
        self.assertGreaterEqual(greeks['delta'], 0.0)
        self.assertLessEqual(greeks['delta'], 1.0)
        
        # Gamma and Vega must be positive for standard options
        self.assertGreater(greeks['gamma'], 0.0)
        self.assertGreater(greeks['vega'], 0.0)
        
        # Rho must be positive for a standard long call option
        self.assertGreater(greeks['rho'], 0.0)

    def test_pde_convergence(self):
        """Verify explicit finite difference PDE converges to analytical value."""
        s_max, nx = 300.0, 300  # Expand grid limits for accurate convergence
        _, s_vals, V = self.model.black_scholes_pde(s_max=s_max, nx=nx)
        
        s0_idx = np.argmin(np.abs(s_vals - self.s0))
        pde_price = V[-1, s0_idx]
        bs_price = self.model.black_scholes_analytical()
        
        # Explicit scheme tolerance limit check (within 1.5% accuracy)
        self.assertNear(pde_price, bs_price, pct_tol=0.015)

    def test_monte_carlo_statistical_bounds(self):
        """Verify Monte Carlo price falls within statistical error bands."""
        np.random.seed(42)  # Seed state to guarantee test determinism
        m, n_paths = 100, 50000
        _, mc_price, std_err = self.model.monte_carlo(m=m, n_paths=n_paths)
        bs_price = self.model.black_scholes_analytical()
        
        # Check that true price falls within 3 standard errors of MC mean
        lower_bound = mc_price - 3 * std_err
        upper_bound = mc_price + 3 * std_err
        self.assertTrue(lower_bound <= bs_price <= upper_bound)

    def test_intrinsic_value_at_deep_itm(self):
        """Verify option price approaches intrinsic value deep ITM."""
        deep_itm_model = CallOptionModel(s0=150.0, k=100.0, t=0.5, r=0.05, sigma=0.2)
        price = deep_itm_model.black_scholes_analytical()
        intrinsic_val = 150.0 - 100.0 * np.exp(-0.05 * 0.5)
        self.assertGreaterEqual(price, intrinsic_val)

    def assertNear(self, actual, expected, pct_tol):
        """Custom assertion helper to evaluate percentage differences."""
        diff = abs(actual - expected) / expected
        if diff > pct_tol:
            self.fail(f"Calculated value {actual} not within {pct_tol*100}% of {expected}")

if __name__ == '__main__':
    unittest.main()

