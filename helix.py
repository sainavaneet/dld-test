import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
print(np.__version__)
print(np.__file__)

import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import torch

def generate_helix_trajectories(n_samples=1000, length=100, radius=1, pitch=0.1, turns=3, noise_level=0.5):
  
    # Adjust the range of t based on the number of turns
    t = np.linspace(0, 2 * np.pi * turns, length)  # Now t makes 'turns' full rotations

    data = []
    for _ in range(n_samples):
        # Adding noise to the radius for variability
        noisy_radius = radius + np.random.normal(0, noise_level, size=(length,))
        x = noisy_radius * np.cos(t)
        y = noisy_radius * np.sin(t)
        z = pitch * t  # Adjust z to scale linearly with t, controlling the separation of turns

        # Stack x, y, z coordinates in the last dimension
        helix = np.stack((x, y, z), axis=1)
        data.append(helix)

    return torch.tensor(data, dtype=torch.float32)

dataset = generate_helix_trajectories(n_samples=100, length=100, turns=1)



# Convert to PyTorch tensor
data_tensor = torch.tensor(dataset, dtype=torch.float32)

# Create TensorDataset and DataLoader
dataset = TensorDataset(dataset)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


loader = DataLoader(dataset, batch_size=32, shuffle=True)

class VectorQuantizer(nn.Module):
    """The Vector Quantizer module for encoding."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # Compute distances between inputs and embeddings
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        # Quantized Vectors
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        # Commitment Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        return loss, quantized , encoding_indices

class Encoder(nn.Module):
    """GRU-based encoder for 3D data."""
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x):
        _, hidden = self.rnn(x)
        return self.linear(hidden.squeeze(0))

class Decoder(nn.Module):
    """GRU-based decoder for 3D data."""
    def __init__(self, embedding_dim, hidden_dim, output_dim, length):
        super().__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.length = length  # Store the length of trajectories

    def forward(self, x):
        x = x.repeat(self.length, 1, 1).transpose(0, 1)  # Repeat encoding for each time step
        output, _ = self.rnn(x)
        return self.linear(output)

def diffuse_embeddings(embeddings, steps, device):
    """Applies a simulated diffusion process to the embeddings."""
    # Create a list to store each diffusion step for analysis if needed
    diffused_embeddings = [embeddings]
    for t in range(1, steps + 1):
        # Increase noise as the step number increases
        noise_level = 0.1 * np.sqrt(t / steps)
        noise = torch.randn_like(embeddings) * noise_level
        embeddings = embeddings + noise
        diffused_embeddings.append(embeddings)
    return embeddings, diffused_embeddings

def reverse_diffuse_embeddings(embeddings, steps, device):
    """Reverses the diffusion process to recover the original embeddings."""
    recovered_embeddings = [embeddings]
    for t in range(steps, 0, -1):
        # Gradually reduce the noise
        noise_level = 0.1 * np.sqrt(t / steps)
        noise = torch.randn_like(embeddings) * noise_level
        embeddings = embeddings - noise
        recovered_embeddings.append(embeddings)
    return embeddings, recovered_embeddings



    


# Adjust the train function to handle diffusion
def train_vq_vae_with_diffusion(loader, model, vector_quantizer, decoder, optimizer, epochs=100, diffusion_steps=10):
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            batch = batch
            optimizer.zero_grad()

            # Encode and quantize
            z = model(batch)
            loss, quantized, _ = vector_quantizer(z)
            
            # Apply diffusion and reverse it during training as a form of regularization
            diffused, _ = diffuse_embeddings(quantized, diffusion_steps, device)
            recovered, _ = reverse_diffuse_embeddings(diffused, diffusion_steps, device)
            
            # Decode from recovered embeddings
            reconstructions = decoder(recovered)
            recon_loss = F.mse_loss(reconstructions, batch)
            
            # Total loss
            total_loss = recon_loss + loss
            total_loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch + 1}, Loss: {total_loss.item()}')
length = 100  # Define this based on your dataset or trajectory length
# Redefine components and optimizer
encoder = Encoder(3, 64, 64).to(device)
decoder = Decoder(64, 64, 3, length).to(device)
vector_quantizer = VectorQuantizer(10, 64, 0.25).to(device)
params = list(encoder.parameters()) + list(decoder.parameters()) + list(vector_quantizer.parameters())
optimizer = optim.Adam(params, lr=0.0001)

# Train the model with diffusion process
train_vq_vae_with_diffusion(loader, encoder, vector_quantizer, decoder, optimizer, epochs=50, diffusion_steps=10)



# Ensure no gradients are computed
with torch.no_grad():
    # Load a sample batch from the data loader
    sample = next(iter(loader)).to(device)
    
    # Encode and quantize the sample
    encoded = encoder(sample)
    _, quantized, encoding_indices = vector_quantizer(encoded)
    
    
    # Decode the recovered embeddings
    reconstructed = decoder(quantized)
    
    # Convert to numpy for plotting
    original_data = sample[0].cpu().numpy()
    reconstructed_data = reconstructed[0].cpu().numpy()

    # Plotting code (same as before)
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(original_data[:, 0], original_data[:, 1], original_data[:, 2], label='Original')
    ax1.legend()
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(reconstructed_data[:, 0], reconstructed_data[:, 1], reconstructed_data[:, 2], label='Reconstructed', color='r')
    ax2.legend()
    plt.show()

