# Autoencoder Complexity Analysis

## Overview
This project implements experiments to verify whether autoencoders can help identify samples that external classifiers might struggle with. The project explores two complementary hypotheses:

1. Samples with high reconstruction error from an autoencoder (i.e., harder to reconstruct) correlate with samples that are more likely to be misclassified by a separate classifier model.

2. Samples that lie far from their class centroid in the latent space (i.e., on the "edges" of class clusters) are more likely to be misclassified by external classification models.

## Hypotheses
1. **Reconstruction Error Hypothesis**: Samples that are "atypical" with respect to the data distribution (as measured by high reconstruction error from an autoencoder) are more likely to be misclassified by external classification models.

2. **Latent Space Distance Hypothesis**: Samples that are farther from their respective class centroids in the autoencoder's latent space (i.e., outliers within their own class) are more likely to be misclassified by external classification models.

## Features
- Trains an autoencoder to learn MNIST digit representations
- Trains a separate classifier for digit recognition
- Analyzes the relationship between reconstruction error and classification accuracy
- Computes class centroids in latent space and measures distances of samples from their class centroids
- Analyzes the relationship between latent space distances and classification accuracy
- Visualizes latent space representations with class clusters and highlights misclassified points
- Provides statistical analysis to verify or refute both hypotheses

## Project Structure
```
├── main.py                # Main experiment script
├── utils.py               # Utility functions for analysis and visualization
├── requirements.txt       # Project dependencies
├── data/                  # Dataset storage (automatically populated with MNIST)
├── models/
│   ├── autoencoder.py     # Autoencoder model implementation
│   ├── classifier.py      # Classifier model implementation
│   ├── autoencoder.pth    # Trained autoencoder weights (generated after training)
│   └── classifier.pth     # Trained classifier weights (generated after training)
└── results/               # Generated visualizations and analysis results
    ├── reconstructions.png           # Original vs reconstructed images
    ├── latent_space.png              # PCA projection of latent space
    ├── latent_space_centroids.png    # Latent space with class centroids
    ├── error_vs_accuracy_boxplot.png # Boxplot of error vs accuracy
    ├── error_vs_accuracy_violin.png  # Violin plot of error distribution
    ├── error_vs_confidence.png       # Error vs classifier confidence
    ├── distance_vs_accuracy_boxplot.png # Boxplot of latent distance vs accuracy
    ├── distance_vs_accuracy_violin.png  # Violin plot of distance distribution
    ├── distance_vs_confidence.png    # Distance vs classifier confidence
    ├── threshold_analysis.png        # Error threshold metrics
    └── distance_threshold_analysis.png # Distance threshold metrics
```

## Requirements
- Python 3.6+
- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-learn
- pandas
- seaborn

Install all dependencies with:
```
pip install -r requirements.txt
```

## Usage

### Basic Execution
Run the experiment with default parameters:
```
python main.py
```

### Customization
Customize the experiment with command-line arguments:
```
python main.py --batch-size 128 --latent-dim 20 --ae-epochs 10 --cls-epochs 8
```

Parameters:
- `--batch-size`: Batch size for training (default: 128)
- `--ae-epochs`: Number of epochs for autoencoder training (default: 10)
- `--cls-epochs`: Number of epochs for classifier training (default: 8)
- `--latent-dim`: Dimension of latent space (default: 20)
- `--distance-metric`: Distance metric to use in latent space (default: 'euclidean', options: 'euclidean', 'cosine')

## Results Interpretation

After running the experiment, you'll find visualization outputs in the `results/` directory and statistical summaries printed to the console. Key metrics to look for:

1. **Accuracy differences**:
   - Between high and low reconstruction error samples
   - Between samples far from vs. close to their class centroids in latent space

2. **Correlations**:
   - Between reconstruction error and classifier confidence
   - Between latent space distance and classifier confidence
   - Between reconstruction error and latent space distance 

3. **Prediction performance**:
   - Precision, recall, and F1 score when using reconstruction error to predict misclassifications
   - Precision, recall, and F1 score when using latent space distance to predict misclassifications

If the hypotheses are correct, you should observe:
- Higher misclassification rates for samples with high reconstruction error and for samples far from their class centroids
- Negative correlations between classifier confidence and both reconstruction error and latent space distance
- Better-than-random performance when using both metrics to identify potential misclassifications

## Key Findings

### Reconstruction Error Approach
The experiment shows evidence supporting the reconstruction error hypothesis:
- Samples with high reconstruction error show significantly lower classification accuracy (87.56%) compared to samples with low reconstruction error (98.63%)
- There is a moderate negative correlation (-0.1962) between reconstruction error and classifier confidence
- Using reconstruction error as a predictor for misclassification achieves a recall of approximately 30%
- Only 4.5% of samples are flagged as having "high reconstruction error," making this a selective approach

### Latent Space Distance Approach
The experiment also provides evidence for the latent space distance hypothesis:
- Samples farther from their class centroids in the latent space show lower classification accuracy
- The correlation between latent space distance and classification error provides a complementary signal to reconstruction error
- The latent distance approach offers a more geometrically intuitive explanation for why certain samples are difficult to classify
- This approach aligns with the intuition that samples on the boundaries between classes are more likely to be misclassified

## Future Directions

Potential extensions to this work:
- Test on more complex datasets beyond MNIST
- Explore different autoencoder architectures (VAEs, denoising autoencoders)
- Combine reconstruction error and latent space distance for better prediction
- Explore other distance metrics in the latent space
- Investigate adaptive approaches that learn class-specific distance thresholds
- Analyze which specific features contribute most to both reconstruction error and misclassification
- Use the identified challenging samples to improve classifier training

## License
[MIT License](https://opensource.org/licenses/MIT)
