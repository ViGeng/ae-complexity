import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, input_size=28*28, num_classes=10):
        super(SimpleClassifier, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        logits = self.model(x)
        return logits
    
    def predict(self, x):
        """Return class predictions and probabilities"""
        x = x.view(x.size(0), -1)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        _, preds = torch.max(probs, 1)
        return preds, probs
    
    def confidence(self, x):
        """Return confidence scores (max probability) for each input"""
        _, probs = self.predict(x)
        confidence, _ = torch.max(probs, dim=1)
        return confidence
