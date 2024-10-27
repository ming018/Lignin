import numpy as np
from matplotlib import pyplot as plt

def plot_grouped_bar_chart(tensor_data_np, target_temp):
    # 막대그래프로 보여주는 함수, 수정 필요함

    bar_width = 0.1  # 컴포넌트는 붙여서 표현하기 위한 막대 폭 설정
    group_gap = 0.3  # 온도 그룹 간 간격 확보

    # target_temp에 대해서만 tensor_data_np로 하고, '2'와 '3'은 임의의 값으로 설정
    categories = [target_temp, '250°C (Dummy)', '300°C (Dummy)']

    # 임의의 데이터 생성 ('250°C (Dummy)'와 '300°C (Dummy)')
    dummy_data_1 = np.random.rand(tensor_data_np.shape[1]) * 100
    dummy_data_2 = np.random.rand(tensor_data_np.shape[1]) * 100

    index = np.arange(len(categories))

    # 색상 리스트 추가
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#A133FF', '#33FFF0', '#FFC733', '#8D33FF', '#FF3333', '#3380FF']

    # 막대그래프를 통해 각 범주(온도)에서의 값을 비교
    plt.figure(figsize=(14, 8))
    for i in range(tensor_data_np.shape[1]):  # 컴포넌트별로 시각화
        plt.bar(index[0] * (tensor_data_np.shape[1] * bar_width + group_gap) + i * bar_width, tensor_data_np[:, i] * 100,
                width=bar_width, label=f'Component {i}', color=colors[i % len(colors)])
        # 임의의 값으로 다른 두 그룹의 막대그래프 생성
        plt.bar(index[1] * (tensor_data_np.shape[1] * bar_width + group_gap) + i * bar_width, dummy_data_1[i],
                width=bar_width, color=colors[i % len(colors)], alpha=0.5)
        plt.bar(index[2] * (tensor_data_np.shape[1] * bar_width + group_gap) + i * bar_width, dummy_data_2[i],
                width=bar_width, color=colors[i % len(colors)], alpha=0.5)

    plt.xlabel('Temperature')
    plt.ylabel('Area %')
    plt.title('Tensor Data Grouped by Temperature with Separated Temperature Groups')
    plt.xticks(index * (tensor_data_np.shape[1] * bar_width + group_gap) + (bar_width * tensor_data_np.shape[1] / 2), labels=categories, fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()