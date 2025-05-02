import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import utils
from models.autoencoder import Autoencoder
from models.classifier import SimpleClassifier

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train_autoencoder(model, dataloader, num_epochs=15, device='cpu'):
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

def train_classifier(model, dataloader, num_epochs=15, device='cpu'):
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

def evaluate(autoencoder, classifier, dataloader, device='cpu', distance_metric='euclidean', use_kmeans=True):
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
    
    # ----------------- RECONSTRUCTION ERROR ANALYSIS -----------------
    print("\n" + "="*50)
    print("RECONSTRUCTION ERROR ANALYSIS")
    print("="*50)
    
    # Analyze the relationship between reconstruction error and classification performance
    utils.plot_error_vs_accuracy(all_recon_errors, all_predictions, all_labels, all_confidences)
    
    # Analyze how well reconstruction error predicts misclassification
    best_threshold, _ = utils.analyze_error_threshold(all_recon_errors, all_predictions, all_labels)
    
    # Calculate statistics for high vs low reconstruction error
    high_error = all_recon_errors > best_threshold
    low_error = ~high_error
    
    print("\nReconstruction Error Statistics:")
    print(f"Samples with high reconstruction error: {np.sum(high_error)} ({np.mean(high_error) * 100:.1f}%)")
    print(f"Samples with low reconstruction error: {np.sum(low_error)} ({np.mean(low_error) * 100:.1f}%)")
    print(f"Accuracy on high error samples: {np.mean(all_predictions[high_error] == all_labels[high_error]) * 100:.2f}%")
    print(f"Accuracy on low error samples: {np.mean(all_predictions[low_error] == all_labels[low_error]) * 100:.2f}%")
    
    # ----------------- LATENT SPACE ANALYSIS WITH TRUE LABELS -----------------
    print("\n" + "="*50)
    print("LATENT SPACE ANALYSIS (WITH TRUE LABELS)")
    print("="*50)
    
    # Compute class centroids using true labels
    centroids = utils.compute_class_centroids(all_latents, all_labels)
    
    # Visualize latent space with centroids
    utils.visualize_latent_space_with_centroids(
        all_latents, all_labels, centroids, predictions=all_predictions,
        title="Latent Space with Class Centroids - Red markers are misclassified samples"
    )
    
    # Compute absolute distances to class centroids
    distances = utils.compute_distances_to_centroids(all_latents, all_labels, centroids, metric=distance_metric)
    
    # Analyze the relationship between latent distance and classification performance
    utils.plot_distance_vs_accuracy(distances, all_predictions, all_labels, all_confidences)
    
    # Analyze how well latent distance predicts misclassification
    best_dist_threshold, _ = utils.analyze_distance_threshold(distances, all_predictions, all_labels)
    
    # Calculate statistics for high vs low latent distances
    high_distance = distances > best_dist_threshold
    low_distance = ~high_distance
    
    print("\nAbsolute Distance Statistics:")
    print(f"Distance metric used: {distance_metric}")
    print(f"Samples with high distance from class centroid: {np.sum(high_distance)} ({np.mean(high_distance) * 100:.1f}%)")
    print(f"Samples with low distance from class centroid: {np.sum(low_distance)} ({np.mean(low_distance) * 100:.1f}%)")
    print(f"Accuracy on high distance samples: {np.mean(all_predictions[high_distance] == all_labels[high_distance]) * 100:.2f}%")
    print(f"Accuracy on low distance samples: {np.mean(all_predictions[low_distance] == all_labels[low_distance]) * 100:.2f}%")
    
    # ----------------- RELATIVE DISTANCE ANALYSIS -----------------
    print("\n" + "="*50)
    print("RELATIVE DISTANCE ANALYSIS")
    print("="*50)
    
    # Compute relative distances (ratio of own-class to other-class distances)
    rel_distances = utils.compute_relative_distances(all_latents, all_labels, centroids, metric=distance_metric)
    
    # Analyze the relationship between relative distance and classification performance
    utils.plot_relative_distance_vs_accuracy(rel_distances, all_predictions, all_labels, all_confidences)
    
    # Analyze how well relative distance predicts misclassification
    best_rel_threshold, _ = utils.analyze_relative_distance_threshold(rel_distances, all_predictions, all_labels)
    
    # Calculate statistics for high vs low relative distances
    high_rel = rel_distances > best_rel_threshold
    low_rel = ~high_rel
    
    print("\nRelative Distance Ratio Statistics:")
    print(f"Samples with high relative distance ratio: {np.sum(high_rel)} ({np.mean(high_rel) * 100:.1f}%)")
    print(f"Samples with low relative distance ratio: {np.sum(low_rel)} ({np.mean(low_rel) * 100:.1f}%)")
    print(f"Accuracy on high relative distance samples: {np.mean(all_predictions[high_rel] == all_labels[high_rel]) * 100:.2f}%")
    print(f"Accuracy on low relative distance samples: {np.mean(all_predictions[low_rel] == all_labels[low_rel]) * 100:.2f}%")
    
    # Calculate correlation between different metrics
    error_distance_corr = np.corrcoef(all_recon_errors, distances)[0, 1]
    error_rel_corr = np.corrcoef(all_recon_errors, rel_distances)[0, 1]
    distance_rel_corr = np.corrcoef(distances, rel_distances)[0, 1]
    
    print("\nCorrelations between metrics:")
    print(f"Reconstruction error vs. absolute distance: {error_distance_corr:.4f}")
    print(f"Reconstruction error vs. relative distance: {error_rel_corr:.4f}")
    print(f"Absolute distance vs. relative distance: {distance_rel_corr:.4f}")
    
    # ----------------- CORRELATION ANALYSIS -----------------
    print("\n" + "="*50)
    print("CORRELATION ANALYSIS")
    print("="*50)
    
    # Convert correctness to binary (1: correct, 0: incorrect)
    correctness = (all_predictions == all_labels).astype(int)
    
    # Calculate correlations
    error_corr = np.corrcoef(all_recon_errors, correctness)[0, 1]
    distance_corr = np.corrcoef(distances, correctness)[0, 1]
    rel_distance_corr = np.corrcoef(rel_distances, correctness)[0, 1]
    
    print("\nCorrelations with classification correctness:")
    print(f"Reconstruction error: {error_corr:.4f}")
    print(f"Absolute distance: {distance_corr:.4f}")
    print(f"Relative distance ratio: {rel_distance_corr:.4f}")
    
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    corr_matrix = np.zeros((5, 5))
    labels = ['Reconstruction Error', 'Absolute Distance', 'Relative Distance', 'Confidence', 'Correctness']
    
    # Fill correlation matrix
    data_matrix = np.vstack([all_recon_errors, distances, rel_distances, all_confidences, correctness])
    corr_matrix = np.corrcoef(data_matrix)
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=labels, yticklabels=labels, cmap='coolwarm')
    plt.title('Correlation Matrix Between Metrics and Classification Correctness')
    plt.tight_layout()
    plt.savefig('results/correlation_matrix.png')
    plt.close()
    
    # Scatter plots of metrics vs correctness (jittered for better visualization)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Add small jitter to correctness for better visualization
    jitter = np.random.normal(0, 0.05, size=len(correctness))
    correctness_jittered = correctness + jitter
    
    # Plot reconstruction error vs correctness
    axes[0].scatter(all_recon_errors, correctness_jittered, alpha=0.5)
    axes[0].set_title(f'Reconstruction Error vs Correctness (r = {error_corr:.3f})')
    axes[0].set_xlabel('Reconstruction Error')
    axes[0].set_ylabel('Correctness (with jitter)')
    
    # Plot absolute distance vs correctness
    axes[1].scatter(distances, correctness_jittered, alpha=0.5)
    axes[1].set_title(f'Absolute Distance vs Correctness (r = {distance_corr:.3f})')
    axes[1].set_xlabel('Absolute Distance')
    
    # Plot relative distance vs correctness
    axes[2].scatter(rel_distances, correctness_jittered, alpha=0.5)
    axes[2].set_title(f'Relative Distance vs Correctness (r = {rel_distance_corr:.3f})')
    axes[2].set_xlabel('Relative Distance')
    
    plt.tight_layout()
    plt.savefig('results/metrics_vs_correctness.png')
    plt.close()
    
    # ----------------- K-MEANS CLUSTERING ANALYSIS -----------------
    if use_kmeans:
        print("\n" + "="*50)
        print("K-MEANS CLUSTERING ANALYSIS (SIMULATING TEST-TIME SCENARIO)")
        print("="*50)
        
        # Perform k-means clustering on latent vectors
        kmeans_labels, kmeans_centroids = utils.perform_kmeans_clustering(all_latents, n_clusters=10)
        
        # Visualize k-means clusters vs true labels
        utils.visualize_kmeans_vs_true(all_latents, all_labels, kmeans_labels,
                                    title="K-Means Clusters vs. True Classes")
        
        # Analyze alignment between clusters and true classes
        mapped_labels, cluster_to_class, kmeans_accuracy = utils.analyze_kmeans_accuracy(all_labels, kmeans_labels)
        
        print(f"K-means clustering accuracy after label mapping: {kmeans_accuracy * 100:.2f}%")
        print("Cluster to class mapping:")
        for cluster, class_label in cluster_to_class.items():
            print(f"  Cluster {cluster} -> Class {class_label}")
        
        # Compute distances using k-means centroids
        kmeans_distances = np.zeros(len(all_latents))
        for i, (sample, cluster) in enumerate(zip(all_latents, kmeans_labels)):
            sample = sample.reshape(1, -1)
            centroid = kmeans_centroids[cluster].reshape(1, -1)
            kmeans_distances[i] = cdist(sample, centroid, metric=distance_metric)[0, 0]
        
        # Compute relative distances using k-means centroids
        kmeans_rel_distances = np.zeros(len(all_latents))
        for i, (sample, cluster) in enumerate(zip(all_latents, kmeans_labels)):
            sample = sample.reshape(1, -1)
            own_centroid = kmeans_centroids[cluster].reshape(1, -1)
            
            # Distance to own cluster centroid
            own_distance = cdist(sample, own_centroid, metric=distance_metric)[0, 0]
            
            # Distances to all other cluster centroids
            other_centroids = np.array([kmeans_centroids[j] for j in range(len(kmeans_centroids)) if j != cluster])
            other_distances = cdist(sample, other_centroids, metric=distance_metric)[0]
            
            # Use minimum distance to other clusters
            min_other_distance = np.min(other_distances)
            
            # Calculate ratio
            epsilon = 1e-10
            kmeans_rel_distances[i] = own_distance / (min_other_distance + epsilon)
        
        # Analyze performance using k-means based metrics
        print("\nK-means distance statistics:")
        corr_kmeans_abs = np.corrcoef(distances, kmeans_distances)[0, 1]
        corr_kmeans_rel = np.corrcoef(rel_distances, kmeans_rel_distances)[0, 1]
        print(f"Correlation between true and k-means absolute distances: {corr_kmeans_abs:.4f}")
        print(f"Correlation between true and k-means relative distances: {corr_kmeans_rel:.4f}")
        
        # Analyze how well k-means derived metrics predict misclassifications
        best_kmeans_rel_threshold, _ = utils.analyze_relative_distance_threshold(
            kmeans_rel_distances, all_predictions, all_labels)
        
        high_kmeans_rel = kmeans_rel_distances > best_kmeans_rel_threshold
        print(f"Accuracy on samples with high k-means relative distance: {np.mean(all_predictions[high_kmeans_rel] == all_labels[high_kmeans_rel]) * 100:.2f}%")
        print(f"Accuracy on samples with low k-means relative distance: {np.mean(all_predictions[~high_kmeans_rel] == all_labels[~high_kmeans_rel]) * 100:.2f}%")
    
    # ----------------- COMBINED APPROACH ANALYSIS -----------------
    print("\n" + "="*50)
    print("COMBINED APPROACH ANALYSIS")
    print("="*50)
    
    # Identify samples with various combinations of metrics
    high_error_high_rel = high_error & high_rel
    
    print("Combined metric statistics:")
    print(f"Samples with both high error and high relative distance: {np.sum(high_error_high_rel)} ({np.mean(high_error_high_rel) * 100:.1f}%)")
    
    if np.sum(high_error_high_rel) > 0:
        accuracy_combined = np.mean(all_predictions[high_error_high_rel] == all_labels[high_error_high_rel])
        print(f"Accuracy on samples flagged by both metrics: {accuracy_combined * 100:.2f}%")
    else:
        print("No samples were flagged by both metrics")
    
    # For comparison, compute standalone metrics one more time
    error_only_accuracy = np.mean(all_predictions[high_error] == all_labels[high_error])
    rel_only_accuracy = np.mean(all_predictions[high_rel] == all_labels[high_rel])
    
    print("\nComparison of different approaches:")
    print(f"Using high reconstruction error only: {error_only_accuracy * 100:.2f}% accuracy")
    print(f"Using high relative distance only: {rel_only_accuracy * 100:.2f}% accuracy")
    if np.sum(high_error_high_rel) > 0:
        print(f"Using both metrics combined: {accuracy_combined * 100:.2f}% accuracy")

def main(args):
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directory for results if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Print CIFAR-10 class names for reference
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print(f"CIFAR-10 classes: {classes}")
    
    # Initialize models with updated input size for CIFAR-10: 32x32x3 = 3072
    autoencoder = Autoencoder(input_size=3072, latent_dim=args.latent_dim).to(device)
    classifier = SimpleClassifier(input_size=3072).to(device)
    
    # Train models - longer training for CIFAR-10
    train_autoencoder(autoencoder, train_loader, num_epochs=args.ae_epochs, device=device)
    train_classifier(classifier, train_loader, num_epochs=args.cls_epochs, device=device)
    
    # Evaluate and analyze
    evaluate(autoencoder, classifier, test_loader, device=device, 
             distance_metric=args.distance_metric, use_kmeans=True)
    
    # Save models
    torch.save(autoencoder.state_dict(), 'models/autoencoder.pth')
    torch.save(classifier.state_dict(), 'models/classifier.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AE Complexity Experiment')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--ae-epochs', type=int, default=15, help='Number of epochs for autoencoder training')
    parser.add_argument('--cls-epochs', type=int, default=15, help='Number of epochs for classifier training')
    parser.add_argument('--latent-dim', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--distance-metric', type=str, default='euclidean', 
                        choices=['euclidean', 'cosine'], help='Distance metric for latent space')
    
    args = parser.parse_args()
    main(args)
