# Autoencoder Complexity Analysis

## Overview
This project implements an experiment to verify whether autoencoders can help identify samples that external classifiers might struggle with. The core hypothesis is that samples with high reconstruction error from an autoencoder (i.e., harder to reconstruct) correlate with samples that are more likely to be misclassified by a separate classifier model.

## Hypothesis
Samples that are "atypical" with respect to the data distribution (as measured by high reconstruction error from an autoencoder) are more likely to be misclassified by external classification models.

## Features
- Trains an autoencoder to learn MNIST digit representations
- Trains a separate classifier for digit recognition
- Analyzes the relationship between reconstruction error and classification accuracy
- Visualizes latent space representations and highlights misclassified points
- Provides statistical analysis to verify or refute the core hypothesis

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
    ├── error_vs_accuracy_boxplot.png # Boxplot of error vs accuracy
    ├── error_vs_accuracy_violin.png  # Violin plot of error distribution
    ├── error_vs_confidence.png       # Error vs classifier confidence
    └── threshold_analysis.png        # Metrics across error thresholds
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

## Results Interpretation

After running the experiment, you'll find visualization outputs in the `results/` directory and statistical summaries printed to the console. Key metrics to look for:

1. **Accuracy difference** between high and low reconstruction error samples
2. **Correlation** between reconstruction error and classifier confidence
3. **Precision, recall, and F1 score** when using reconstruction error to predict misclassifications

If the hypothesis is correct, you should observe:
- Higher misclassification rates for samples with high reconstruction error
- Negative correlation between reconstruction error and classifier confidence
- Better-than-random performance when using reconstruction error to identify potential misclassifications

## Key Findings

The experiment shows evidence supporting the hypothesis:
- Samples with high reconstruction error show significantly lower classification accuracy (87.56%) compared to samples with low reconstruction error (98.63%)
- There is a moderate negative correlation (-0.1962) between reconstruction error and classifier confidence
- Using reconstruction error as a predictor for misclassification achieves a recall of approximately 30%
- Only 4.5% of samples are flagged as having "high reconstruction error," making this a selective approach

## Future Directions

Potential extensions to this work:
- Test on more complex datasets beyond MNIST
- Explore different autoencoder architectures (VAEs, denoising autoencoders)
- Combine reconstruction error with other signals for better prediction
- Analyze which specific features contribute most to both reconstruction error and misclassification

## License
[MIT License](https://opensource.org/licenses/MIT)
