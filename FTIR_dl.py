from preprocessing import group_and_average_data, group_preprocessed_data, clip_data_to_100, process_data_with_log
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def preprocess_FTIR_data(FTIR_data):
    preprocessed_data = group_and_average_data(FTIR_data)
    preprocessed_data = group_preprocessed_data(preprocessed_data)
    preprocessed_data = clip_data_to_100(preprocessed_data)
    preprocessed_data = process_data_with_log(preprocessed_data)
    return preprocessed_data

# 학습 및 평가 함수
def train_and_evaluate(model, temperatures, outputs, learning_rate=0.0008, epochs=10000):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        predicted_outputs = model(temperatures)
        loss = criterion(predicted_outputs, outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# 예측 및 시각화 함수
def predict_and_plot(model, preprocessed_data, new_temperatures):
    model.eval()
    with torch.no_grad():
        predicted_new_outputs = model(new_temperatures)

        # 기존 데이터와 예측 데이터 시각화
        plt.plot(100 - np.exp(preprocessed_data[0][0, 1, :]))
        plt.plot(100 - np.exp(preprocessed_data[0][1, 1, :]))
        plt.plot(100 - np.exp(preprocessed_data[0][2, 1, :]))
        plt.plot(100 - np.exp(preprocessed_data[0][3, 1, :]))
        plt.plot(100 - np.exp(predicted_new_outputs[0].cpu().detach().numpy()))
        plt.show()

        plt.plot(100 - np.exp(predicted_new_outputs[0].cpu().detach().numpy()))
        plt.show()
        print("A")

    return predicted_new_outputs