import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim

def train_and_evaluate(model, temperatures, composition_data, learning_rate=0.001, epochs=500):
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

    return predicted_composition


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