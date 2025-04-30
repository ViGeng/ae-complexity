import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from models.autoencoder import Autoencoder
from models.classifier import SimpleClassifier
import utils

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train_autoencoder(model, dataloader, num_epochs=10, device='cpu'):
    """Train the autoencoder model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    print("Training autoencoder...")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            
            # Forward pass
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data.view(data.size(0), -1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    print("Autoencoder training complete!")

def train_classifier(model, dataloader, num_epochs=10, device='cpu'):
    """Train the classifier model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    print("Training classifier...")
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    print("Classifier training complete!")

def evaluate(autoencoder, classifier, dataloader, device='cpu'):
    """Evaluate both models on the test set and analyze results"""
    autoencoder.eval()
    classifier.eval()
    
    all_labels = []
    all_recon_errors = []
    all_predictions = []
    all_confidences = []
    all_latents = []
    
    sample_originals = []
    sample_reconstructions = []
    samples_collected = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            labels = labels.to(device)
            
            # Get reconstructions and latent representations from autoencoder
            reconstructions, latents = autoencoder(data)
            recon_errors = autoencoder.reconstruction_error(data)
            
            # Get classifier predictions and confidences
            preds, probs = classifier.predict(data)
            confidences = torch.max(probs, dim=1)[0]
            
            # Store results
            all_labels.append(labels.cpu().numpy())
            all_recon_errors.append(recon_errors.cpu().numpy())
            all_predictions.append(preds.cpu().numpy())
            all_confidences.append(confidences.cpu().numpy())
            all_latents.append(latents.cpu().numpy())
            
            # Collect some samples for visualization
            if samples_collected < 10:
                n_collect = min(10 - samples_collected, data.size(0))
                sample_originals.append(data[:n_collect].view(n_collect, -1).cpu().numpy())
                sample_reconstructions.append(reconstructions[:n_collect].cpu().numpy())
                samples_collected += n_collect
    
    # Concatenate all results
    all_labels = np.concatenate(all_labels)
    all_recon_errors = np.concatenate(all_recon_errors)
    all_predictions = np.concatenate(all_predictions)
    all_confidences = np.concatenate(all_confidences)
    all_latents = np.concatenate(all_latents)
    
    sample_originals = np.concatenate(sample_originals)[:10]
    sample_reconstructions = np.concatenate(sample_reconstructions)[:10]
    
    # Calculate overall accuracy
    accuracy = np.mean(all_predictions == all_labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    
    # Visualize reconstructions
    utils.plot_reconstructions(sample_originals, sample_reconstructions)
    
    # Visualize latent space colored by prediction correctness
    utils.visualize_latent_space(
        all_latents, all_labels, predictions=all_predictions,
        title="Latent Space - Red: Misclassified, Blue: Correct"
    )
    
    # Analyze the relationship between reconstruction error and classification performance
    utils.plot_error_vs_accuracy(all_recon_errors, all_predictions, all_labels, all_confidences)
    
    # Analyze how well reconstruction error predicts misclassification
    best_threshold, _ = utils.analyze_error_threshold(all_recon_errors, all_predictions, all_labels)
    
    # Calculate statistics for high vs low reconstruction error
    high_error = all_recon_errors > best_threshold
    low_error = ~high_error
    
    print("\nStatistics:")
    print(f"Samples with high reconstruction error: {np.sum(high_error)} ({np.mean(high_error) * 100:.1f}%)")
    print(f"Samples with low reconstruction error: {np.sum(low_error)} ({np.mean(low_error) * 100:.1f}%)")
    print(f"Accuracy on high error samples: {np.mean(all_predictions[high_error] == all_labels[high_error]) * 100:.2f}%")
    print(f"Accuracy on low error samples: {np.mean(all_predictions[low_error] == all_labels[low_error]) * 100:.2f}%")
    
    # Calculate correlation between reconstruction error and confidence
    corr = np.corrcoef(all_recon_errors, all_confidences)[0, 1]
    print(f"Correlation between reconstruction error and confidence: {corr:.4f}")

def main(args):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directory for results if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Initialize models
    autoencoder = Autoencoder(latent_dim=args.latent_dim).to(device)
    classifier = SimpleClassifier().to(device)
    
    # Train models
    train_autoencoder(autoencoder, train_loader, num_epochs=args.ae_epochs, device=device)
    train_classifier(classifier, train_loader, num_epochs=args.cls_epochs, device=device)
    
    # Evaluate and analyze
    evaluate(autoencoder, classifier, test_loader, device=device)
    
    # Save models
    torch.save(autoencoder.state_dict(), 'models/autoencoder.pth')
    torch.save(classifier.state_dict(), 'models/classifier.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AE Complexity Experiment')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--ae-epochs', type=int, default=10, help='Number of epochs for autoencoder training')
    parser.add_argument('--cls-epochs', type=int, default=8, help='Number of epochs for classifier training')
    parser.add_argument('--latent-dim', type=int, default=20, help='Dimension of latent space')
    
    args = parser.parse_args()
    main(args)
