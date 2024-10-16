import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 사용자 정의 Dataset 클래스
class ByproductDataset(Dataset):
    def __init__(self, temperature_data, byproduct_data):
        """
        temperature_data: (N, input_size) 형태의 온도 데이터
        byproduct_data: (N, output_size) 형태의 부산물 데이터
        """
        self.temperature_data = temperature_data
        self.byproduct_data = byproduct_data

    def __len__(self):
        return len(self.temperature_data)

    def __getitem__(self, idx):
        temperature = self.temperature_data[idx]
        byproduct = self.byproduct_data[idx]
        return temperature, byproduct

# CNN 모델 정의
class ByproductPredictorCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ByproductPredictorCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)  # 1차원 컨볼루션
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(input_size * 32, 1024)  # fully connected layer
        self.fc2 = nn.Linear(1024, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flattening
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x