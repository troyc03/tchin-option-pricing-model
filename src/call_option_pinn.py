import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class BlackScholesPINN(nn.Module):
    def __init__(self, r, sigma, K, T):
        super().__init__()
        self.r = r                  # Risk-free interest rate
        self.sigma = sigma          # Volatility of the underlying asset
        self.K = K                  # Strike price
        self.T = T                  # Time to maturity
        
        # Neural network architecture: 2 inputs (S, t) -> 3 hidden layers -> 1 output (V)
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, S, t):
        # Combine stock price S and time t into a single input tensor
        inputs = torch.cat([S, t], dim=1)
        return self.net(inputs)

    def pde_residual(self, S, t):
        """Calculates the Black-Scholes PDE residual."""
        # Enable gradient tracking for inputs
        S.requires_grad_(True)
        t.requires_grad_(True)
        
        V = self.forward(S, t)
        
        # First-order derivatives
        V_S = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        
        # Second-order derivative
        V_SS = torch.autograd.grad(V_S, S, grad_outputs=torch.ones_like(V_S), create_graph=True)[0]
        
        # Black-Scholes PDE: dV/dt + 0.5 * sigma^2 * S^2 * d^2V/dS^2 + r * S * dV/dS - r * V = 0
        residual = V_t + 0.5 * (self.sigma ** 2) * (S ** 2) * V_SS + self.r * S * V_S - self.r * V
        return residual

def train_pinn(model, epochs=5000, lr=1e-3, num_points=2000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Dictionary to track training metrics
    history = {
        'epoch': [],
        'total': [],
        'pde': [],
        'terminal': [],
        'boundary_zero': []
    }
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 1. Interior Domain Points (Collocation points for PDE loss)
        S_domain = torch.rand(num_points, 1) * (2 * model.K)  # S ranges from 0 to 2*K
        t_domain = torch.rand(num_points, 1) * model.T        # t ranges from 0 to T
        loss_pde = torch.mean(model.pde_residual(S_domain, t_domain) ** 2)
        
        # 2. Terminal Boundary Conditions (At maturity t = T, V = max(S - K, 0))
        S_terminal = torch.rand(num_points, 1) * (2 * model.K)
        t_terminal = torch.full((num_points, 1), model.T)
        V_terminal_pred = model(S_terminal, t_terminal)
        V_terminal_true = torch.clamp(S_terminal - model.K, min=0.0)
        loss_terminal = loss_fn(V_terminal_pred, V_terminal_true)
        
        # 3. Boundary Conditions at S = 0 (V = 0)
        S_zero = torch.zeros(num_points, 1)
        t_zero = torch.rand(num_points, 1) * model.T
        V_zero_pred = model(S_zero, t_zero)
        loss_boundary_zero = loss_fn(V_zero_pred, torch.zeros_like(V_zero_pred))
        
        # Total loss aggregation
        total_loss = loss_pde + loss_terminal + loss_boundary_zero
        total_loss.backward()
        optimizer.step()
        
        # Record metrics
        history['epoch'].append(epoch)
        history['total'].append(total_loss.item())
        history['pde'].append(loss_pde.item())
        history['terminal'].append(loss_terminal.item())
        history['boundary_zero'].append(loss_boundary_zero.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:04d} | Total Loss: {total_loss.item():.6f} | PDE: {loss_pde.item():.6f} | Terminal: {loss_terminal.item():.6f}")
            
    return history

# Example usage configuration
if __name__ == "__main__":
    # Parameters
    r = 0.05       # 5% Risk-free rate
    sigma = 0.25   # 25% Volatility
    K = 100.0      # Strike Price
    T = 1.0        # 1 Year to expiration

    # Initialize model
    pinn = BlackScholesPINN(r, sigma, K, T)

    # Train the network
    print("Starting PINN Training...")
    history = train_pinn(pinn, epochs=3000, lr=1e-3, num_points=1000)

    # --- Plot 1: PINN Loss Convergence Curves ---
    plt.figure(figsize=(10, 5))
    plt.plot(history['epoch'], history['total'], label='Total Loss', color='black', linewidth=1.5)
    plt.plot(history['epoch'], history['pde'], label='PDE Residual Loss', linestyle='--')
    plt.plot(history['epoch'], history['terminal'], label='Terminal Condition Loss', linestyle='--')
    plt.plot(history['epoch'], history['boundary_zero'], label='S=0 Boundary Loss', linestyle='--')
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Log Scale)")
    plt.title("PINN Loss Convergence Curves")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

    # --- Plot 2: 2D Curve Slice at t = 0 (Today) ---
    S_test = np.linspace(0, 200, 100)
    t_test = np.zeros_like(S_test)
    S_tensor = torch.tensor(S_test, dtype=torch.float32).view(-1, 1)
    t_tensor = torch.tensor(t_test, dtype=torch.float32).view(-1, 1)
    
    pinn.eval()
    with torch.no_grad():
        V_pred = pinn(S_tensor, t_tensor).numpy()
        
    plt.figure(figsize=(8, 5))
    plt.plot(S_test, V_pred, label="PINN Predicted Option Value (t=0)", color='blue')
    plt.axvline(x=K, color='red', linestyle='--', label=f'Strike Price (K={K})')
    plt.xlabel("Stock Price (S)")
    plt.ylabel("Option Price (V)")
    plt.title("Black-Scholes European Call Price via PINN (t=0)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Plot 3: 3D Call Option Price Surface ---
    S_grid = np.linspace(0, 2 * K, 100)
    t_grid = np.linspace(0, T, 100)
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid)

    # Reshape and wrap into tensors for the network
    S_tensor_mesh = torch.tensor(S_mesh.ravel(), dtype=torch.float32).view(-1, 1)
    t_tensor_mesh = torch.tensor(t_mesh.ravel(), dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        V_surface_pred = pinn(S_tensor_mesh, t_tensor_mesh).numpy().reshape(S_mesh.shape)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(S_mesh, t_mesh, V_surface_pred, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel("Stock Price (S)")
    ax.set_ylabel("Time to Maturity (t)")
    ax.set_zlabel("Option Price (V)")
    ax.set_title("Black-Scholes Call Option Price Surface")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    # Adjust viewing perspective angle to cleanly see maturity shape mapping
    ax.view_init(elev=20, azim=-120)
    plt.show()
