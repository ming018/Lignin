import torch
import torch.nn as nn
from torchviz import make_dot


# 1D CNN 모델 정의
class TemperatureToDataPredictorCNN(nn.Module):
    def __init__(self, input_size=1, output_size=3476):
        super(TemperatureToDataPredictorCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * input_size, 1024)
        self.fc2 = nn.Linear(1024, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = TemperatureToDataPredictorCNN()

# Print the model summary
# Input size is (channels, sequence length), e.g., (1, 10)
# summary(model, input_size=(1, 10))
#
print(repr(model))


# 입력 크기와 출력 크기 설정
input_size = 1 # 예시 입력 크기 (변경 가능)
output_size = 3476  # 예시 출력 크기 (변경 가능)

# 모델 인스턴스 생성
model = TemperatureToDataPredictorCNN(input_size=input_size, output_size=output_size)

# 임의의 샘플 입력 데이터 생성 (배치 크기 1, 채널 1, 입력 길이 input_size)
sample_input = torch.randn(1, 1, input_size)

# 모델의 출력 계산
output = model(sample_input)

# 모델의 계산 그래프 시각화
graph = make_dot(output, params=dict(model.named_parameters()))
graph.render("TemperatureToDataPredictorCNN", format="png")  # PNG 파일로 저장

