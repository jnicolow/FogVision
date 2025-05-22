"""
This module contains code to create a fog classifier class

Author: Joel Nicolow, Information and Computer Science, University of Hawaii at Manoa (September 25, 2024)
"""

import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, num_ftrs=2048):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_ftrs, 1024),  # input size is 2048 (assuming your embeddings have 2048 features)
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # output layer: 2 classes (binary classification)
        )

    def forward(self, x):
        return self.model(x)


class FogClassifier(nn.Module):
    def __init__(self, base_model, fine_tune_base=False):
        super(FogClassifier, self).__init__()
        num_ftrs = base_model.fc.in_features

        # get output at penultimate layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = fine_tune_base # freeze basemodel if fine_tune is False
        # Add a new classification head
        self.classifier = SimpleNN(num_ftrs)
    
    def forward(self, x):
        # Extract embeddings from the penultimate layer
        embeddings = self.features(x)
        embeddings = embeddings.view(embeddings.size(0), -1)  # Flatten the output
        # Pass the embeddings through the classifier
        out = self.classifier(embeddings)
        return out, embeddings

# Instantiate the model
# import torchvision.models as models

# model = FogClassifier(models.resnet50(pretrained=True))


