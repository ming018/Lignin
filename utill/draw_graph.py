import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def read_and_process_csv(csv_file):
    """CSV 파일을 읽고 중복 데이터를 제거한 후 처리된 DataFrame 반환"""
    df = pd.read_csv(csv_file)
    df = df.drop_duplicates(subset=['Temperature', 'Weight'])
    return df


def calculate_derivatives(C, weight, sample_mass):
    """파생 변수 계산 (벡터화된 연산 사용)"""
    delta_C = np.diff(C)
    delta_weight = np.diff(weight)

    derivative = delta_weight / delta_C
    percentage_change = np.abs(derivative / sample_mass) * 100

    return np.concatenate([[percentage_change[0]], percentage_change]), np.concatenate([[derivative[0]], derivative])


def smooth_data(data, sigma):
    """Gaussian smoothing"""
    return gaussian_filter1d(data, sigma=sigma)


def plot_graph(C, test, test2, derW, per, starting, smoothing_sigma):
    """스무딩된 데이터와 원본 데이터를 그래프로 시각화"""
    smoothed_test = smooth_data(test, smoothing_sigma)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Derivative Weight', color='tab:blue')

    ax1.plot(C[starting:], smoothed_test[starting:], color='tab:green', label='Smooth Derivative')
    ax1.plot(C[starting:], test2[starting:], color='tab:pink', label='Raw Derivative')
    ax1.plot(C[starting:], derW[starting:], color='tab:blue', label='derW')

    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.set_ylabel('Weight %', color='tab:red')
    ax2.plot(C[starting:], per[starting:], color='tab:red', label='Weight %')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()


def graph_test(csv_file, sample_mass, starting=1500, smoothing_sigma=2):
    """전체 프로세스를 수행하는 메인 함수"""
    df = read_and_process_csv(csv_file)
    C = np.array(df['Detail_Temperature'])
    weight = np.array(df['Detail_Weight'])
    derW = np.abs(np.array(df['Deriv. Weight']))
    per = np.abs(np.array(df['Weight_%']))

    test, test2 = calculate_derivatives(C, weight, sample_mass)
    plot_graph(C, test, test2, derW, per, starting, smoothing_sigma)
