import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import copy
import numpy as np


def train_step(model, x, t, optimiser, tau=0.99):
    s_y_hat, s_y = model(x, t)

    loss = F.l1_loss(s_y_hat, s_y)

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    with torch.no_grad():
        for context_param, target_param in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
            target_param.data.mul_(tau).add_(context_param.data, alpha=1.0 - tau)  # gemini recommended this method to apply computations directly to .data attribute and not mess with computational graph

    return loss.item()
    

BATCH_SIZE = 32
TOTAL_STEPS = 200
L = 10       # time steps per window
D = 128     # latent embedding dimension
EPOCHS = 2000

if __name__ == "__main__":
    from models import ChuaJEPA
    from data import generate_chua_batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = ChuaJEPA(L=L, D=D, n_heads=8, num_layers=4).to(device) # initialise model
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_history = []

    for epoch in range(EPOCHS):

        x_batch, t_batch = generate_chua_batch(BATCH_SIZE, TOTAL_STEPS, L)
        x_batch = x_batch.to(device)
        t_batch = t_batch.to(device)
        loss_val = train_step(model, x_batch, t_batch, optimiser, tau=0.996)

        loss_history.append(loss_val)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Loss: {loss_val:.4f}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color='teal', linewidth=2)
    plt.title("JEPA Latent Loss on Chua's Circuit")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.yscale('log')  # Log scale helps visualize the convergence tail
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    torch.save(model.state_dict(), "chua_jepa.pth")
    print("Model saved to chua_jepa.pth")