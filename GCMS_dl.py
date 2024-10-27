import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
import torch
import os

def train_and_evaluate(model, temperatures, composition_data, target_temp, device, learning_rate=0.001, epochs=5000):
    MODEL_PATH = "composition_model.pth"

    if os.path.exists(MODEL_PATH):
        print("Loading the existing model...")
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))  # 저장된 모델 가중치 불러오기
    else:
        print("Training the model...")

        criterion = nn.MSELoss()  # 손실 함수: MSE
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

        for epoch in range(epochs):
            model.train()
            predicted_composition = model(temperatures)
            loss = criterion(predicted_composition, composition_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

                # 학습 후 모델 저장
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    # 예측 예시
    model.eval()
    with torch.no_grad():
        new_temperatures = np.array(target_temp, dtype=np.float32).reshape(-1, 1)
        new_temperatures = torch.tensor(new_temperatures).unsqueeze(1).to(device)
        predicted_compositions = model(new_temperatures)

    return predicted_compositions


# 실제와 예측 데이터를 비교하는 바 그래프 시각화 함수
def plot_comparison(actual, predicted, temp, component_labels):
    x = np.arange(len(actual))  # 구성 요소 개수만큼 x축 생성

    # 그래프 크기 설정
    plt.figure(figsize=(10, 6))

    # 실제 값
    plt.bar(x - 0.2, actual, 0.4, label='Actual', color='blue')

    # 예측 값
    plt.bar(x + 0.2, predicted, 0.4, label='Predicted', color='orange')

    # 그래프 제목 및 레이블 설정
    plt.xlabel('Components')
    plt.ylabel('Percentage')
    plt.title(f'Actual vs Predicted Composition at {temp}°C')

    # x축 눈금과 레이블
    plt.xticks(x, component_labels)

    # 범례 추가
    plt.legend()

    # 그래프 보여주기
    plt.show()

def plot_comparison(actual, predicted, temp, component_labels):
    x = np.arange(len(actual))  # 구성 요소 개수만큼 x축 생성

    # 그래프 크기 설정
    plt.figure(figsize=(10, 6))

    # 실제 값
    plt.bar(x - 0.2, actual, 0.4, label='Actual', color='blue')

    # 예측 값
    plt.bar(x + 0.2, predicted, 0.4, label='Predicted', color='orange')

    # 그래프 제목 및 레이블 설정
    plt.xlabel('Components')
    plt.ylabel('Percentage')
    plt.title(f'Actual vs Predicted Composition at {temp}°C')

    # x축 눈금과 레이블
    plt.xticks(x, component_labels)

    # 범례 추가
    plt.legend()

    # 그래프 보여주기
    plt.show()