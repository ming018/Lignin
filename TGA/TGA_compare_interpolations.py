from scipy.interpolate import CubicSpline, PchipInterpolator, Rbf, Akima1DInterpolator, interp1d
import numpy as np
from matplotlib import pyplot as plt
from TGA import TGA_evaluate
import time

def interpolate_data(data1, data2, method):
    """
    두 리스트의 각 요소에 대해 보간을 수행하여 새로운 리스트 data3을 생성합니다.

    Parameters:
    data1: 첫 번째 리스트
    data2: 두 번째 리스트
    method: 'spline', 'pchip', 'Rbf', 'akima', 'linear' 중 보간 방법 선택

    Returns:
    data3: data1과 data2의 각 대응하는 값을 보간한 새로운 리스트
    """

    # data1과 data2가 같은 길이여야 함
    if len(data1) != len(data2):
        raise ValueError("data1과 data2의 길이는 같아야 합니다.")

    # x 좌표 생성 (index 0, 1)
    x = np.array([0, 1])

    # 새로운 리스트를 저장할 data3
    data3 = []

    # data1과 data2가 2차원 배열일 경우 각 열마다 보간 수행
    if isinstance(data1[0], (list, np.ndarray)):
        # data1과 data2가 2차원 배열인 경우 각 열마다 보간
        for i in range(len(data1[0])):
            y = np.array([data1[:, i], data2[:, i]])

            # 보간 방법 선택
            if method == 'spline':
                interp_func = CubicSpline(x, y)
            elif method == 'pchip':
                interp_func = PchipInterpolator(x, y)
            elif method == 'Rbf':
                interp_func = Rbf(x, y, function='multiquadric')
            elif method == 'akima':
                interp_func = Akima1DInterpolator(x, y)
            elif method == 'linear':
                interp_func = interp1d(x, y, kind='linear')
            else:
                raise ValueError("method는 'spline', 'pchip', 'Rbf', 'akima', 'linear' 중 하나여야 합니다.")

            # x=0.5에서 중간 값 계산 (0과 1 사이에서 중간 값)
            middle_value = interp_func(0.5)
            data3.append(middle_value)

    else:
        # 1차원 데이터일 경우 기존 방법
        for i in range(len(data1)):
            y = np.array([data1[i], data2[i]])

            # 보간 방법 선택
            if method == 'spline':
                interp_func = CubicSpline(x, y)
            elif method == 'pchip':
                interp_func = PchipInterpolator(x, y)
            elif method == 'Rbf':
                interp_func = Rbf(x, y, function='multiquadric')
            elif method == 'akima':
                interp_func = Akima1DInterpolator(x, y)
            elif method == 'linear':
                interp_func = interp1d(x, y, kind='linear')
            else:
                raise ValueError("method는 'spline', 'pchip', 'Rbf', 'akima', 'linear' 중 하나여야 합니다.")

            # x=0.5에서 중간 값 계산 (0과 1 사이에서 중간 값)
            middle_value = interp_func(0.5)
            data3.append(middle_value)

    return data3



def smooth_data(data, window_size=3):
    """
    데이터를 스무딩하는 함수 (이동 평균 방식)

    Parameters:
    data: 스무딩할 데이터 리스트
    window_size: 이동 평균의 윈도우 크기 (기본값은 3)

    Returns:
    스무딩된 리스트
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_five_lists(list1, list2, list3, list4, list5, labels=None, smooth=True, window_size=10):
    """
    다섯 개의 리스트를 한 그래프에 그려주는 함수, list1에 스무딩 적용 가능

    Parameters:
    list1, list2, list3, list4, list5: 그릴 리스트들
    labels: 각 리스트에 대한 레이블 (기본값은 None, 리스트로 제공 가능)
    smooth: True일 경우 list1에 스무딩 적용
    window_size: 스무딩 윈도우 크기 (기본값은 3)
    """
    # list1에 스무딩 적용 여부 확인
    if smooth:
        list1_smoothed = smooth_data(list1, window_size)
        x1 = range(len(list1_smoothed))  # 스무딩 적용 후 x 길이가 줄어듦
    else:
        list1_smoothed = list1
        x1 = range(len(list1))

    plt.figure(figsize=(10, 6))

    # 각 리스트에 대한 plot 그리기 (투명도 조정, 마커 없음)
    plt.plot(x1, list1_smoothed, label=labels[0] if labels else 'Linear Interpolation' if smooth else 'Linear Interpolation', linestyle='-', alpha=0.3, color='blue')
    plt.plot(range(len(list2)), list2, label=labels[1] if labels else 'Cubic Spline', linestyle='--', alpha=0.3,
             color='green')
    plt.plot(range(len(list3)), list3, label=labels[2] if labels else 'PCHIP', linestyle='-.', alpha=0.3, color='red')
    plt.plot(range(len(list4)), list4, label=labels[3] if labels else 'Rbf Interpolation', linestyle=':', alpha=0.3,
             color='purple')
    plt.plot(range(len(list5)), list5, label=labels[4] if labels else 'akima', linestyle='-', alpha=0.3,
             color='orange')

    # x, y축 레이블 변경
    plt.title("Draw Predict Data")
    plt.xlabel("Temperature")  # x축 레이블을 Temperature로 변경
    plt.ylabel("Deriv. Weight")  # y축 레이블을 Deriv. Weight로 변경
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # 범례를 그래프 밖으로 이동
    plt.grid(True)
    plt.tight_layout()  # 그래프를 레이아웃에 맞게 조정
    plt.show()


def compare_main(data1, data2) :
    check_time = []

    data1 = np.abs(data1[4])
    data2 = np.abs(data2[4])

    # 각 보간 방법에 대해 결과 계산
    start_time = time.time()
    linear = interpolate_data(data1, data2, 'linear')
    end_time = time.time()
    check_time.append(end_time - start_time)

    start_time = time.time()
    cubic = interpolate_data(data1, data2, 'spline')
    end_time = time.time()
    check_time.append(end_time - start_time)

    start_time = time.time()
    pchip = interpolate_data(data1, data2, 'pchip')
    end_time = time.time()
    check_time.append(end_time - start_time)

    start_time = time.time()
    polynomial = interpolate_data(data1, data2, 'Rbf')
    end_time = time.time()
    check_time.append(end_time - start_time)

    start_time = time.time()
    akima = interpolate_data(data1, data2, 'akima')
    end_time = time.time()
    check_time.append(end_time - start_time)

    # 각 보간을 그래프로 시각화
    plot_five_lists(linear, cubic, pchip, polynomial, akima)

    print(TGA_evaluate.dtw_distance(data1, data2, linear))
    print(TGA_evaluate.dtw_distance(data1, data2, cubic))
    print(TGA_evaluate.dtw_distance(data1, data2, pchip))
    print(TGA_evaluate.dtw_distance(data1, data2, polynomial))
    print(TGA_evaluate.dtw_distance(data1, data2, akima))

    print('linear')
    TGA_evaluate.TGA_evaluate(275, 'No', data1, data2, linear, 250, 300)
    print(f"보간 소요 시간 {check_time[0]}\n")

    print('cubic')
    TGA_evaluate.TGA_evaluate(275, 'No', data1, data2, cubic, 250, 300)
    print(f"보간 소요 시간 {check_time[1]}\n")

    print('pchip')
    TGA_evaluate.TGA_evaluate(275, 'No', data1, data2, pchip, 250, 300)
    print(f"보간 소요 시간 {check_time[2]}\n")

    print('polynomial')
    TGA_evaluate.TGA_evaluate(275, 'No', data1, data2, polynomial, 250, 300)
    print(f"보간 소요 시간 {check_time[3]}\n")

    print('akima')
    TGA_evaluate.TGA_evaluate(275, 'No', data1, data2, akima, 250, 300)
    print(f"보간 소요 시간 {check_time[4]}\n")

    print()