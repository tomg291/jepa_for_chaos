import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from train_jepa import ChuaJEPA, generate_chua_batch, L, D

print("Load model and visualise with PCA")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChuaJEPA(L=L, D=D, n_heads=8, num_layers=4)

model.load_state_dict(torch.load("chua_jepa.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

with torch.no_grad():
    x_test, t_test = generate_chua_batch(1, 10000, L)
    x_test, t_test = x_test.to(device), t_test.to(device)
    
    s_y_hat, s_y = model(x_test, t_test)
    
    s_y_np = s_y.squeeze(0).cpu().numpy()
    s_y_hat_np = s_y_hat.squeeze(0).cpu().numpy()

pca = PCA(n_components=3)
true_3d = pca.fit_transform(s_y_np)

pred_3d = pca.transform(s_y_hat_np)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(true_3d[:, 0], true_3d[:, 1], true_3d[:, 2], 
        color='blue', label='True Encoded Future', lw=2)

ax.plot(pred_3d[:, 0], pred_3d[:, 1], pred_3d[:, 2], 
        color='red', label='Predicted Future', lw=2, linestyle='--')

ax.set_title("JEPA Predictor vs Target in Latent Space")
ax.legend()
plt.show()