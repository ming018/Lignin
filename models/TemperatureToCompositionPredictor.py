import torch
import torch.nn as nn
from torchsummary import summary
from torchviz import make_dot


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


# model = TemperatureToCompositionPredictor()
# print(repr(model))
#
#
# # 입력 크기와 출력 크기 설정
# input_size = 1 # 예시 입력 크기 (변경 가능)
# output_size = 10  # 예시 출력 크기 (변경 가능)
#
# # 모델 인스턴스 생성
# model = TemperatureToCompositionPredictor(input_size=input_size, output_size=output_size)
#
# # 임의의 샘플 입력 데이터 생성 (배치 크기 1, 채널 1, 입력 길이 input_size)
# sample_input = torch.randn(1, 1, input_size)
#
# # 모델의 출력 계산
# output = model(sample_input)
#
# # 모델의 계산 그래프 시각화
# graph = make_dot(output, params=dict(model.named_parameters()))
# graph.render("GCMS", format="png")  # PNG 파일로 저장
#
