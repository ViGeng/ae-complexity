import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix


def plot_reconstructions(original_images, reconstructed_images, n=10):
    """Plot original and reconstructed images side by side"""
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        # For CIFAR-10: reshape to 32x32x3 and transpose for proper display
        orig_img = original_images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        plt.imshow(np.clip(orig_img, 0, 1))  # Clip to valid range
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        # For CIFAR-10: reshape to 32x32x3 and transpose for proper display
        recon_img = reconstructed_images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        plt.imshow(np.clip(recon_img, 0, 1))  # Clip to valid range
        plt.title("Reconstructed")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/reconstructions.png')
    plt.close()
    
def visualize_latent_space(latent_vectors, labels, predictions=None, title=None):
    """Visualize the latent space using PCA projection"""
    # Project to 2D if latent dimension is > 2
    if latent_vectors.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_vectors)
    else:
        latent_2d = latent_vectors
    
    plt.figure(figsize=(12, 10))
    
    # If predictions are provided, color points based on correctness
    if predictions is not None:
        correct = predictions == labels
        plt.scatter(latent_2d[correct, 0], latent_2d[correct, 1], c='blue', alpha=0.5, label='Correct predictions')
        plt.scatter(latent_2d[~correct, 0], latent_2d[~correct, 1], c='red', alpha=0.8, label='Incorrect predictions')
        plt.legend()
    else:
        # Otherwise, color by class label
        for label in np.unique(labels):
            idx = labels == label
            plt.scatter(latent_2d[idx, 0], latent_2d[idx, 1], label=f'Class {label}', alpha=0.6)
        plt.legend()
    
    if title:
        plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.tight_layout()
    plt.savefig('results/latent_space.png')
    plt.close()

def visualize_latent_space_with_centroids(latent_vectors, labels, centroids, predictions=None, title=None):
    """Visualize the latent space with class centroids using PCA projection"""
    # Project to 2D if latent dimension is > 2
    if latent_vectors.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_vectors)
        centroids_2d = pca.transform(centroids)
    else:
        latent_2d = latent_vectors
        centroids_2d = centroids
    
    plt.figure(figsize=(12, 10))
    
    # Plot class samples
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(latent_2d[idx, 0], latent_2d[idx, 1], alpha=0.6, label=f'Class {label}')
    
    # Plot centroids
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', s=100, marker='X', label='Class Centroids')
    
    # If predictions are provided, mark misclassified points
    if predictions is not None:
        incorrect = predictions != labels
        plt.scatter(latent_2d[incorrect, 0], latent_2d[incorrect, 1], c='red', s=80, marker='o', 
                    edgecolors='white', linewidths=1, alpha=0.8, label='Misclassified')

    plt.legend()
    if title:
        plt.title(title)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.tight_layout()
    plt.savefig('results/latent_space_centroids.png')
    plt.close()

def compute_class_centroids(latent_vectors, labels):
    """Compute centroid of each class in latent space"""
    unique_labels = np.unique(labels)
    centroids = np.zeros((len(unique_labels), latent_vectors.shape[1]))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        centroids[i] = latent_vectors[mask].mean(axis=0)
        
    return centroids

def compute_distances_to_centroids(latent_vectors, labels, centroids, metric='euclidean'):
    """Compute distance of each sample to its class centroid"""
    distances = np.zeros(len(latent_vectors))
    
    for label in np.unique(labels):
        mask = labels == label
        class_latents = latent_vectors[mask]
        centroid = centroids[label].reshape(1, -1)  # Reshape for cdist
        
        # Calculate distances using scipy's cdist
        if metric == 'euclidean':
            class_distances = cdist(class_latents, centroid, metric='euclidean').squeeze()
        elif metric == 'cosine':
            class_distances = cdist(class_latents, centroid, metric='cosine').squeeze()
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
            
        distances[mask] = class_distances
        
    return distances

def compute_relative_distances(latent_vectors, labels, centroids, metric='euclidean'):
    """
    Compute the ratio of distance to own class centroid vs. closest other class centroid
    A lower ratio means the sample is much closer to its own class than to others (good)
    A higher ratio means the sample is relatively close to other classes (likely to be misclassified)
    """
    num_samples = len(latent_vectors)
    relative_distances = np.zeros(num_samples)
    
    for i, (sample, label) in enumerate(zip(latent_vectors, labels)):
        sample = sample.reshape(1, -1)
        own_centroid = centroids[label].reshape(1, -1)
        
        # Distance to own class centroid
        own_distance = cdist(sample, own_centroid, metric=metric)[0, 0]
        
        # Distances to all other class centroids
        other_centroids = np.array([centroids[j] for j in range(len(centroids)) if j != label])
        other_distances = cdist(sample, other_centroids, metric=metric)[0]
        
        # Use the minimum distance to any other class centroid
        min_other_distance = np.min(other_distances)
        
        # Calculate the ratio: distance to own centroid / distance to closest other centroid
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        ratio = own_distance / (min_other_distance + epsilon)
        
        relative_distances[i] = ratio
    
    return relative_distances

def perform_kmeans_clustering(latent_vectors, n_clusters=10, random_state=42):
    """
    Perform k-means clustering in the latent space
    For use in test phase when true labels are not available
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_vectors)
    centroids = kmeans.cluster_centers_
    
    return cluster_labels, centroids

def visualize_kmeans_vs_true(latent_vectors, true_labels, kmeans_labels, title=None):
    """
    Visualize how k-means clusters align with true class labels
    """
    # Project to 2D if latent dimension is > 2
    if latent_vectors.shape[1] > 2:
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(latent_vectors)
    else:
        latent_2d = latent_vectors
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot true labels
    for label in np.unique(true_labels):
        idx = true_labels == label
        ax1.scatter(latent_2d[idx, 0], latent_2d[idx, 1], alpha=0.6, label=f'Class {label}')
    
    ax1.set_title('True Labels')
    ax1.legend()
    
    # Plot k-means clusters
    for cluster in np.unique(kmeans_labels):
        idx = kmeans_labels == cluster
        ax2.scatter(latent_2d[idx, 0], latent_2d[idx, 1], alpha=0.6, label=f'Cluster {cluster}')
    
    ax2.set_title('K-Means Clusters')
    ax2.legend()
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    plt.savefig('results/kmeans_vs_true.png')
    plt.close()

def analyze_kmeans_accuracy(true_labels, kmeans_labels):
    """
    Analyze how well k-means clustering aligns with true class labels
    Returns a mapping between clusters and true classes, plus accuracy
    """
    # Create a contingency table
    contingency = np.zeros((10, 10), dtype=int)  # Assuming 10 classes for MNIST
    
    for i in range(len(true_labels)):
        contingency[true_labels[i], kmeans_labels[i]] += 1
    
    # For each cluster, find the most common true class
    cluster_to_class = {}
    for cluster in range(10):
        most_common_class = np.argmax(contingency[:, cluster])
        cluster_to_class[cluster] = most_common_class
    
    # Map the cluster labels back to class labels
    mapped_labels = np.array([cluster_to_class[label] for label in kmeans_labels])
    
    # Calculate accuracy
    accuracy = np.mean(mapped_labels == true_labels)
    
    return mapped_labels, cluster_to_class, accuracy

def plot_error_vs_accuracy(reconstruction_errors, predictions, labels, confidences=None):
    """Plot reconstruction error vs. classification correctness"""
    correct = (predictions == labels)
    
    data = {
        'Reconstruction Error': reconstruction_errors,
        'Prediction': ['Correct' if c else 'Incorrect' for c in correct]
    }
    
    if confidences is not None:
        data['Confidence'] = confidences
    
    df = pd.DataFrame(data)
    
    # Boxplot of reconstruction error by prediction correctness
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Prediction', y='Reconstruction Error', data=df)
    plt.title('Reconstruction Error by Prediction Correctness')
    plt.tight_layout()
    plt.savefig('results/error_vs_accuracy_boxplot.png')
    plt.close()
    
    # Violin plot for more detailed distribution
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Prediction', y='Reconstruction Error', data=df)
    plt.title('Reconstruction Error Distribution by Prediction Correctness')
    plt.tight_layout()
    plt.savefig('results/error_vs_accuracy_violin.png')
    plt.close()
    
    if confidences is not None:
        # Scatter plot of reconstruction error vs. confidence
        plt.figure(figsize=(10, 8))
        plt.scatter(df['Confidence'], df['Reconstruction Error'], 
                   c=['blue' if c else 'red' for c in correct], alpha=0.5)
        plt.title('Reconstruction Error vs. Classifier Confidence')
        plt.xlabel('Classifier Confidence')
        plt.ylabel('Reconstruction Error')
        plt.tight_layout()
        plt.savefig('results/error_vs_confidence.png')
        plt.close()
        
        # Compute correlation
        correlation = np.corrcoef(reconstruction_errors, confidences)[0, 1]
        print(f"Correlation between reconstruction error and confidence: {correlation:.4f}")

def plot_distance_vs_accuracy(distances, predictions, labels, confidences=None):
    """Plot latent space distance vs. classification correctness"""
    correct = (predictions == labels)
    
    data = {
        'Latent Distance': distances,
        'Prediction': ['Correct' if c else 'Incorrect' for c in correct]
    }
    
    if confidences is not None:
        data['Confidence'] = confidences
    
    df = pd.DataFrame(data)
    
    # Boxplot of distance by prediction correctness
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Prediction', y='Latent Distance', data=df)
    plt.title('Latent Space Distance by Prediction Correctness')
    plt.tight_layout()
    plt.savefig('results/distance_vs_accuracy_boxplot.png')
    plt.close()
    
    # Violin plot for more detailed distribution
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Prediction', y='Latent Distance', data=df)
    plt.title('Latent Space Distance Distribution by Prediction Correctness')
    plt.tight_layout()
    plt.savefig('results/distance_vs_accuracy_violin.png')
    plt.close()
    
    if confidences is not None:
        # Scatter plot of distance vs. confidence
        plt.figure(figsize=(10, 8))
        plt.scatter(df['Confidence'], df['Latent Distance'], 
                   c=['blue' if c else 'red' for c in correct], alpha=0.5)
        plt.title('Latent Space Distance vs. Classifier Confidence')
        plt.xlabel('Classifier Confidence')
        plt.ylabel('Latent Space Distance')
        plt.tight_layout()
        plt.savefig('results/distance_vs_confidence.png')
        plt.close()
        
        # Compute correlation
        correlation = np.corrcoef(distances, confidences)[0, 1]
        print(f"Correlation between latent space distance and confidence: {correlation:.4f}")

def plot_relative_distance_vs_accuracy(rel_distances, predictions, labels, confidences=None):
    """Plot relative distance ratio vs. classification correctness"""
    correct = (predictions == labels)
    
    data = {
        'Relative Distance Ratio': rel_distances,
        'Prediction': ['Correct' if c else 'Incorrect' for c in correct]
    }
    
    if confidences is not None:
        data['Confidence'] = confidences
    
    df = pd.DataFrame(data)
    
    # Boxplot of relative distance by prediction correctness
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Prediction', y='Relative Distance Ratio', data=df)
    plt.title('Relative Distance Ratio by Prediction Correctness')
    plt.tight_layout()
    plt.savefig('results/relative_distance_vs_accuracy_boxplot.png')
    plt.close()
    
    # Violin plot for more detailed distribution
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Prediction', y='Relative Distance Ratio', data=df)
    plt.title('Relative Distance Ratio Distribution by Prediction Correctness')
    plt.tight_layout()
    plt.savefig('results/relative_distance_vs_accuracy_violin.png')
    plt.close()
    
    if confidences is not None:
        # Scatter plot of relative distance vs. confidence
        plt.figure(figsize=(10, 8))
        plt.scatter(df['Confidence'], df['Relative Distance Ratio'], 
                   c=['blue' if c else 'red' for c in correct], alpha=0.5)
        plt.title('Relative Distance Ratio vs. Classifier Confidence')
        plt.xlabel('Classifier Confidence')
        plt.ylabel('Relative Distance Ratio')
        plt.tight_layout()
        plt.savefig('results/relative_distance_vs_confidence.png')
        plt.close()
        
        # Compute correlation
        correlation = np.corrcoef(rel_distances, confidences)[0, 1]
        print(f"Correlation between relative distance ratio and confidence: {correlation:.4f}")

def analyze_error_threshold(reconstruction_errors, predictions, labels, thresholds=None):
    """Analyze how well reconstruction error predicts classification mistakes"""
    if thresholds is None:
        # Create a range of thresholds based on the distribution of errors
        thresholds = np.linspace(np.min(reconstruction_errors), np.max(reconstruction_errors), 100)
    
    results = []
    for threshold in thresholds:
        # Predict as "will be misclassified" if error > threshold
        predicted_mistakes = reconstruction_errors > threshold
        actual_mistakes = predictions != labels
        
        # Calculate metrics
        true_pos = np.sum(predicted_mistakes & actual_mistakes)
        false_pos = np.sum(predicted_mistakes & ~actual_mistakes)
        true_neg = np.sum(~predicted_mistakes & ~actual_mistakes)
        false_neg = np.sum(~predicted_mistakes & actual_mistakes)
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    results_df = pd.DataFrame(results)
    
    # Plot metrics vs. threshold
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['threshold'], results_df['precision'], label='Precision')
    plt.plot(results_df['threshold'], results_df['recall'], label='Recall')
    plt.plot(results_df['threshold'], results_df['f1'], label='F1')
    plt.xlabel('Reconstruction Error Threshold')
    plt.ylabel('Metric Value')
    plt.title('Prediction Metrics by Reconstruction Error Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/threshold_analysis.png')
    plt.close()
    
    # Find best threshold by F1 score
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_metrics = results_df.loc[best_idx].to_dict()
    
    print(f"Best threshold: {best_threshold:.4f}")
    print(f"At this threshold:")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  F1: {best_metrics['f1']:.4f}")
    
    return best_threshold, results_df

def analyze_distance_threshold(distances, predictions, labels, thresholds=None):
    """Analyze how well latent space distance predicts classification mistakes"""
    if thresholds is None:
        # Create a range of thresholds based on the distribution of distances
        thresholds = np.linspace(np.min(distances), np.max(distances), 100)
    
    results = []
    for threshold in thresholds:
        # Predict as "will be misclassified" if distance > threshold
        predicted_mistakes = distances > threshold
        actual_mistakes = predictions != labels
        
        # Calculate metrics
        true_pos = np.sum(predicted_mistakes & actual_mistakes)
        false_pos = np.sum(predicted_mistakes & ~actual_mistakes)
        true_neg = np.sum(~predicted_mistakes & ~actual_mistakes)
        false_neg = np.sum(~predicted_mistakes & actual_mistakes)
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    results_df = pd.DataFrame(results)
    
    # Plot metrics vs. threshold
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['threshold'], results_df['precision'], label='Precision')
    plt.plot(results_df['threshold'], results_df['recall'], label='Recall')
    plt.plot(results_df['threshold'], results_df['f1'], label='F1')
    plt.xlabel('Latent Space Distance Threshold')
    plt.ylabel('Metric Value')
    plt.title('Prediction Metrics by Distance Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/distance_threshold_analysis.png')
    plt.close()
    
    # Find best threshold by F1 score
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_metrics = results_df.loc[best_idx].to_dict()
    
    print(f"\nBest distance threshold: {best_threshold:.4f}")
    print(f"At this threshold:")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  F1: {best_metrics['f1']:.4f}")
    
    return best_threshold, results_df

def analyze_relative_distance_threshold(rel_distances, predictions, labels, thresholds=None):
    """Analyze how well relative distance ratio predicts classification mistakes"""
    if thresholds is None:
        # Create a range of thresholds based on the distribution of distances
        thresholds = np.linspace(np.min(rel_distances), np.max(rel_distances), 100)
    
    results = []
    for threshold in thresholds:
        # Predict as "will be misclassified" if ratio > threshold
        predicted_mistakes = rel_distances > threshold
        actual_mistakes = predictions != labels
        
        # Calculate metrics
        true_pos = np.sum(predicted_mistakes & actual_mistakes)
        false_pos = np.sum(predicted_mistakes & ~actual_mistakes)
        true_neg = np.sum(~predicted_mistakes & ~actual_mistakes)
        false_neg = np.sum(~predicted_mistakes & actual_mistakes)
        
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    results_df = pd.DataFrame(results)
    
    # Plot metrics vs. threshold
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['threshold'], results_df['precision'], label='Precision')
    plt.plot(results_df['threshold'], results_df['recall'], label='Recall')
    plt.plot(results_df['threshold'], results_df['f1'], label='F1')
    plt.xlabel('Relative Distance Ratio Threshold')
    plt.ylabel('Metric Value')
    plt.title('Prediction Metrics by Relative Distance Threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/relative_threshold_analysis.png')
    plt.close()
    
    # Find best threshold by F1 score
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_metrics = results_df.loc[best_idx].to_dict()
    
    print(f"\nBest relative distance threshold: {best_threshold:.4f}")
    print(f"At this threshold:")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  F1: {best_metrics['f1']:.4f}")
    
    return best_threshold, results_df
