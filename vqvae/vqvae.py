import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import h5py
from torch.utils.data import Dataset


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
        '''
        These are the encodings that we got from the encoder'''

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
    """Bidirectional GRU-based encoder with dropout for 3D data."""
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim * 2, embedding_dim)  # *2 for bidirectional

    def forward(self, x):
        _, hidden = self.rnn(x)
        # Concatenate the hidden states from both directions
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.linear(hidden)

class Decoder(nn.Module):
    """GRU-based decoder with attention and teacher forcing for 3D data."""
    def __init__(self, embedding_dim, hidden_dim, output_dim, length, num_layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.length = length
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

    def forward(self, x, encoder_outputs=None, teacher_forcing_ratio=0.5):
        x = x.repeat(self.length, 1, 1).transpose(0, 1)  # Repeat encoding for each time step
        output, _ = self.rnn(x)

        if encoder_outputs is not None:
            output, attn_weights = self.attention(output, encoder_outputs, encoder_outputs)
        
        return self.linear(output)


