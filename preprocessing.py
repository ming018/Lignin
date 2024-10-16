import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def interpolate_temperature(data, start_temp=40, end_temp=800, step=0.01):
    """
    온도 값을 0.01 단위로 40도에서 800도까지 보간하는 함수.

    :param data: (3, length) 형태의 ndarray, [0]에 온도 값이 있음
    :param start_temp: 보간할 온도의 시작 값 (기본값: 40)
    :param end_temp: 보간할 온도의 종료 값 (기본값: 800)
    :param step: 보간할 온도의 간격 (기본값: 0.01)
    :return: 보간된 데이터를 가진 (3, new_length) 형태의 ndarray
    """
    # 데이터 분리: 온도 값 (data[0])과 나머지 값 (data[1], data[2])
    temperature = data[0]
    remaining_data = data[1:]

    # 새로운 온도 범위를 40도에서 800도까지 0.01 간격으로 생성
    new_temperatures = np.arange(start_temp, end_temp, step)

    # 보간(interpolation) 함수 생성 (온도를 기준으로 보간)
    interpolated_data = []

    for i in range(remaining_data.shape[0]):
        # 보간 함수 설정
        interpolation_function = interp1d(temperature, remaining_data[i], fill_value="extrapolate")
        interpolated_values = interpolation_function(new_temperatures)
        interpolated_data.append(interpolated_values)

    interpolated_data = np.array(interpolated_data)

    # 온도가 40도보다 높은 값에서 시작하는 경우, 40도 이하 구간을 첫 번째 값으로 채움
    min_temp_in_data = temperature.min()
    if min_temp_in_data > start_temp:
        interpolated_data[:, new_temperatures < min_temp_in_data] = remaining_data[:, 0].reshape(-1, 1)

    # 결과적으로 온도 값과 보간된 데이터 결합 (3, new_length) 형태로 반환
    final_data = np.vstack([new_temperatures, interpolated_data])

    return final_data


def reduce_by_temperature(data, method='mean'):
    """
    온도 값을 기준으로 각 온도별 대표 값을 뽑아 길이를 줄이는 함수.

    :param data: (3, length) 형태의 ndarray, [0]에 온도 값이 있음
    :param method: 대표 값을 계산하는 방법 ('mean', 'median', 'max' 등)
    :return: 온도별 대표 값을 추출한 데이터
    """
    # 데이터 분리: 온도 값 (data[0])과 나머지 값 (data[1], data[2])
    temperature = data[0]
    remaining_data = data[1:]

    # 데이터프레임으로 변환하여 처리
    df = pd.DataFrame(np.vstack([temperature, remaining_data]).T, columns=['Temperature', 'Data1', 'Data2'])

    # 온도 값을 기준으로 그룹화한 뒤, 각 그룹의 대표 값을 계산
    if method == 'mean':
        df_reduced = df.groupby('Temperature').mean()
    elif method == 'median':
        df_reduced = df.groupby('Temperature').median()
    elif method == 'max':
        df_reduced = df.groupby('Temperature').max()
    else:
        raise ValueError(f"허용되지 않은 method: {method}")

    # 결과를 다시 (3, new_length) 형태로 변환하여 반환
    reduced_data = df_reduced.reset_index().T.values

    return reduced_data

def reduce_to_one_degree_interval(data):
    """
    보간된 데이터를 1도 단위로 줄이는 함수.

    :param data: (3, new_length) 형태의 보간된 ndarray, [0]에 온도 값이 있음
    :return: 1도 단위로 줄인 (3, reduced_length) 형태의 ndarray
    """
    # 온도 데이터 (data[0])를 가져옴
    temperature = data[0]

    # 1도 단위로 새로운 온도 배열 생성 (온도의 범위는 40도에서 800도까지로 가정)
    reduced_temperatures = np.arange(np.floor(temperature.min()), np.ceil(temperature.max()) + 1, 1)

    # 각 1도에 대응하는 값을 추출
    reduced_data = []
    for temp in reduced_temperatures:
        # 현재 온도 값에 가장 가까운 값을 찾음
        idx = np.argmin(np.abs(temperature - temp))
        reduced_data.append(data[:, idx])

    # 결과를 (3, reduced_length) 형태로 변환하여 반환
    reduced_data = np.array(reduced_data).T

    return reduced_data
