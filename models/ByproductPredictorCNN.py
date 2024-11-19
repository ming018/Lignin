import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchviz import make_dot


# Dataset 클래스 정의
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

model = ByproductPredictorCNN(1, 761)

print(repr(model))


# 입력 크기와 출력 크기 설정
input_size = 1 # 예시 입력 크기 (변경 가능)
output_size = 761  # 예시 출력 크기 (변경 가능)

# 모델 인스턴스 생성
model = ByproductPredictorCNN(input_size=input_size, output_size=output_size)

# 임의의 샘플 입력 데이터 생성 (배치 크기 1, 채널 1, 입력 길이 input_size)
sample_input = torch.randn(1, 1, input_size)

# 모델의 출력 계산
output = model(sample_input)

# 모델의 계산 그래프 시각화
graph = make_dot(output, params=dict(model.named_parameters()))
graph.render("byproduct_predictor_cnn", format="png")  # PNG 파일로 저장