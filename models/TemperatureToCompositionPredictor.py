import torch
import torch.nn as nn
import torch.optim as optim

# Fully Connected Neural Network with Softmax output for percentage prediction
class TemperatureToCompositionPredictor(nn.Module):
    def __init__(self, input_size=1, output_size=10, hidden_channels=64):
        super(TemperatureToCompositionPredictor, self).__init__()
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_channels, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Ensure the output is a probability distribution

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Global Average Pooling to reduce to the channel dimension
        x = x.mean(dim=-1)  # Apply global average pooling

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))  # Output is a percentage (sum to 1)
        return x