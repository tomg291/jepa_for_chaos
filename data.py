import torch
import numpy as np

def generate_chua_batch(batch_size, total_steps, L, dt=0.05):
    """Concise Chua generator. Simulates at 0.01s for stability, outputs at dt."""
    inner_dt = 0.01
    sim_steps = int(total_steps * (dt / inner_dt))
    
    states = np.random.uniform(-0.1, 0.1, (batch_size, 3))
    states[:, 0] += 0.7
    traj = []
    
    def f(s):
        x, y, z = s[:,0], s[:,1], s[:,2]
        fx = -0.714*x + 0.5*(-1.143 - -0.714)*(np.abs(x+1) - np.abs(x-1))
        return np.stack([15.6*(y-x-fx), x-y+z, -28.0*y], axis=1)

    for _ in range(sim_steps):
        k1 = f(states)
        k2 = f(states + 0.5 * inner_dt * k1)
        k3 = f(states + 0.5 * inner_dt * k2)
        k4 = f(states + inner_dt * k3)
        states += (inner_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        traj.append(states.copy())
        
    # Stack and downsample to the requested dt
    traj = np.stack(traj, axis=1)[:, ::int(dt/inner_dt), :]
    
    # Generate timestamps
    t_windows = torch.arange(total_steps // L, dtype=torch.float32) * (L * dt)
    t_tensor = t_windows.unsqueeze(0).repeat(batch_size, 1)
    
    return torch.tensor(traj, dtype=torch.float32), t_tensor