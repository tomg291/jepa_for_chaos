import torch
import torch.nn.functional as F
import torch.optim as optim

EPOCHS = 2000
BATCH_SIZE = 64
TOTAL_STEPS = 200
L = 10
D = 128

from models import ChuaJEPA, ReadoutMLP
from data import generate_chua_batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChuaJEPA(L=L, D=D, n_heads=8, num_layers=4)
model.load_state_dict(torch.load("chua_jepa.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

decoder = ReadoutMLP(D=D, L=L, hidden_size=256).to(device)

optimiser = optim.Adam(decoder.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=1000, gamma=0.1)

loss_history = []

for epoch in range(EPOCHS):
    x_batch, t_batch = generate_chua_batch(BATCH_SIZE, TOTAL_STEPS, L)
    x_batch = x_batch.to(device)
    t_batch = t_batch.to(device)

    with torch.no_grad():
        h = model.ts_embedder(x_batch)
        t_embed = model.pos_embedder(t_batch)
        v_t = h + t_embed
        s_true = model.target_encoder(v_t)

    x_hat = decoder(s_true)

    N = TOTAL_STEPS // L
    x_target = x_batch.view(BATCH_SIZE, N, L, 3)

    loss = F.mse_loss(x_hat, x_target)
    loss_history.append(loss.item())

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    scheduler.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

print(f"Final Loss: {loss:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(loss_history, color='teal', linewidth=2)
plt.title("Readout Decoder Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.yscale('log')  # Log scale helps visualize the convergence tail
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


torch.save(decoder.state_dict(), "chua_decoder.pth")
print("Decoder saved!")