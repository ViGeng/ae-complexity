import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_size=28*28, latent_dim=20):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, input_size),
            nn.Sigmoid()  # For MNIST pixel values in [0, 1]
        )
        
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        
        # Encode
        latent = self.encoder(x)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent
    
    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def reconstruct(self, x):
        x = x.view(x.size(0), -1)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def reconstruction_error(self, x):
        """Calculate reconstruction error for each sample"""
        x_flat = x.view(x.size(0), -1)
        reconstructed, _ = self.forward(x)
        error = F.mse_loss(reconstructed, x_flat, reduction='none').sum(dim=1)
        return error
