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


def group_and_average_data(data_list):
    """
    동일한 조건에서 수행된 세 번의 실험 결과를 그룹화하고, 평균을 구하여 대표 데이터를 생성하는 함수.

    :param data_list: 길이 48의 리스트, 각 요소는 (2, 3476) 크기의 ndarray
    :return: 그룹화된 실험 조건의 평균 데이터를 담은 리스트
    """
    # 각 그룹별 인덱스 (1-3, 4-6, ... 형태로 그룹화)
    groups = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
        [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23],
        [24, 25, 26], [27, 28, 29], [30, 31, 32], [33, 34, 35],
        [36, 37, 38], [39, 40, 41], [42, 43, 44], [45, 46, 47]
    ]

    processed_data = []

    for group in groups:
        # 각 그룹의 실험 결과 추출 (ndarray[0] = 주파수, ndarray[1] = 광 투과율)
        freq_data = np.array([data_list[i][0] for i in group])
        transmittance_data = np.array([data_list[i][1] for i in group])

        # 주파수와 광 투과율의 평균 계산
        avg_freq = np.mean(freq_data, axis=0)
        avg_transmittance = np.mean(transmittance_data, axis=0)

        # 결과 저장 (주파수와 광 투과율 평균값을 한 쌍으로)
        processed_data.append((avg_freq, avg_transmittance))

    return processed_data


def group_preprocessed_data(data_list):
    """
    preprocessed_data 리스트를 (0/4/8/12), (1/5/9/13), (2/6/10/14), (3/7/11/15) 형식으로 그룹화하는 함수.

    :param data_list: 길이 16인 preprocessed_data 리스트
    :return: 그룹화된 데이터를 담은 리스트 (각 그룹은 4개의 데이터로 구성됨)
    """
    # 그룹 인덱스 정의
    groups = [
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15]
    ]

    grouped_data = []

    # 각 그룹 인덱스에 맞춰 데이터 그룹화
    for group in groups:
        grouped_data.append(np.asarray([data_list[i] for i in group]))

    return grouped_data

def clip_data_to_100(data_list):
    """
    각 ndarray의 값을 100으로 클리핑하는 함수.

    :param data_list: 길이 16인 preprocessed_data 리스트, 각 요소는 ndarray로 구성
    :return: 각 ndarray의 값이 100을 넘지 않도록 클리핑된 데이터 리스트
    """
    clipped_data = []

    for data in data_list:
        # ndarray의 값들을 100으로 클리핑
        clipped_ndarray = np.clip(data, None, 100)
        clipped_data.append(clipped_ndarray)

    return clipped_data


def process_data_with_log(data_list):
    """
    각 ndarray의 값을 100에서 뺀 후, 그 결과에 로그를 취하는 함수.

    :param data_list: 길이 16인 preprocessed_data 리스트, 각 요소는 ndarray로 구성
    :return: 100에서 값을 뺀 후 로그를 취한 데이터 리스트
    """
    processed_data = []

    for data in data_list:
        # 100에서 값을 빼고, 0 이하인 값이 없도록 처리
        subtracted_data = 101 - data

        # 음수나 0을 방지하기 위해 subtracted_data에서 최소 1e-6을 유지
        safe_data = np.clip(subtracted_data, 1e-6, None)

        # 로그를 취함
        log_data = np.log(safe_data)

        processed_data.append(log_data)

    return processed_data
