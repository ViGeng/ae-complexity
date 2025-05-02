import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, input_size=3072, latent_dim=128):
        super(Autoencoder, self).__init__()
        
        # Encoder - expanded architecture for CIFAR-10
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Decoder - expanded architecture for CIFAR-10
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_size),
            nn.Sigmoid()  # For CIFAR-10 pixel values in [0, 1]
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
