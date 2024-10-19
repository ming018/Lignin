import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 1D CNN 모델 정의
class TemperatureToDataPredictorCNN(nn.Module):
    def __init__(self, input_size=1, output_size=3476):
        super(TemperatureToDataPredictorCNN, self).__init__()
        # 1D Convolution layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * input_size, 1024)
        self.fc2 = nn.Linear(1024, output_size)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
