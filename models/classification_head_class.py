"""
"""

import torch
import torch.nn as nn

class PyramidEndDropout(nn.Module):
    def __init__(self, input_size=2048, output_size=2, dropout_prob=0.5):
        super(PyramidEndDropout, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.fc1(x); x = self.relu(x)
        x = self.fc2(x); x = self.relu(x)
        x = self.fc3(x); x = self.relu(x)
        x = self.fc4(x); x = self.relu(x); x = self.dropout(x)
        x = self.fc5(x); x = self.relu(x); x = self.dropout(x)
        x = self.fc6(x)
        return x