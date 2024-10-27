import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from models.ByproductPredictorCNN import ByproductDataset
from postprocessing import gaussian_smooth_data
from preprocessing import reduce_by_temperature, interpolate_temperature, reduce_to_one_degree_interval


def process_TGA_data(TGA_data, cat, target_temp):
    """입력 값에 따라 TGA 데이터를 처리하고 보간을 진행."""
    # 데이터 처리 및 보간
    from TGA import group
    data_for_return, temp1, temp2 = group.process_group_for_TGA(TGA_data, cat, target_temp)
    data = [reduce_by_temperature(d) for d in data_for_return]
    data = [interpolate_temperature(d, 40, 800) for d in data]
    data = [reduce_to_one_degree_interval(d) for d in data]

    return data, temp1, temp2


def prepare_dataloader(data, computer_device):
    """Dataset과 DataLoader를 생성."""
    temperature_data = torch.tensor([[250], [300], [350], [400]], dtype=torch.float).to(computer_device)
    byproduct_data = torch.tensor(data, dtype=torch.float).to(computer_device)

    # Dataset 및 DataLoader 생성
    batch_size = 16
    dataset = ByproductDataset(temperature_data, byproduct_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def train_model(model, dataloader, model_path, computer_device, epochs=10000, learning_rate=0.001):
    """모델 학습을 수행하고, 완료되면 모델을 저장."""
    model.to(computer_device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (temperatures, byproducts) in enumerate(dataloader):
            temperatures = temperatures.unsqueeze(1)  # (batch_size, 1, input_size)로 변환

            # 순전파 및 역전파 계산
            outputs = model(temperatures)
            loss = criterion(outputs, byproducts[:, 2, :])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

    # 학습이 완료된 모델 저장
    print(f"Training complete. Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path, computer_device):
    """학습된 모델을 불러옵니다."""
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(computer_device)
        model.eval()
        return True
    return False


def evaluate_model(model, computer_device):
    """새로운 온도에서 모델을 평가하고 결과를 반환."""
    model.eval()
    with torch.no_grad():
        new_temperatures = torch.tensor([[260], [320], [370], [420]], dtype=torch.float).to(computer_device)
        new_temperatures = new_temperatures.unsqueeze(1)
        predicted_byproducts = model(new_temperatures)

    return predicted_byproducts.detach().cpu().numpy()


def smooth_data(predicted_byproducts, sigma=2):
    """Gaussian smoothing을 적용."""
    return np.array([gaussian_smooth_data(byproduct, sigma) for byproduct in predicted_byproducts])
