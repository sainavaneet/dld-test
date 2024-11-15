import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import DataLoader

# Import the necessary components from your main.py or wherever your model classes are defined
from main import Encoder, Decoder, VectorQuantizer, generate_helix_trajectories

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Models
encoder = Encoder(3, 64, 64).to(device)
decoder = Decoder(64, 64, 3, 100).to(device)
vector_quantizer = VectorQuantizer(10, 64, 0.25).to(device)

# Ensure the models are set to evaluation mode to deactivate dropout or batch norm layers that behave differently during training
encoder.eval()
decoder.eval()
vector_quantizer.eval()

# Load the trained model weights
encoder.load_state_dict(torch.load('models/Encoder.pth'))
decoder.load_state_dict(torch.load('models/Decoder.pth'))
vector_quantizer.load_state_dict(torch.load('models/VectorQuantizer.pth'))

# Prepare your data loader (assuming you have a dataset function from main.py)
dataset = generate_helix_trajectories(n_samples=100, length=100, turns=3)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# Testing the model
with torch.no_grad():  # Ensures that gradients are not computed, which is important for testing
    for sample in loader:
        sample = sample.to(device)
        encoded = encoder(sample)
        _, quantized, _ = vector_quantizer(encoded)  # Remove unnecessary computation of loss here
        reconstructed = decoder(quantized)
        original_data = sample[0].cpu().numpy()
        reconstructed_data = reconstructed[0].cpu().numpy()

        # Plot the original and reconstructed data
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(original_data[:, 0], original_data[:, 1], original_data[:, 2], label='Original')
        ax1.legend()
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(reconstructed_data[:, 0], reconstructed_data[:, 1], reconstructed_data[:, 2], label='Reconstructed', color='r')
        ax2.legend()
        plt.show()

        break  # Optional: stop after one batch to avoid processing the entire dataset
