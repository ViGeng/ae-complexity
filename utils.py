import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def plot_reconstructions(original_images, reconstructed_images, n=10):
    """Plot original and reconstructed images side by side"""
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
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
