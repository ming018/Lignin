import os

import pandas as pd
import numpy as np
import torch
from fastdtw import fastdtw
from pandas import read_csv
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.metrics.pairwise import cosine_similarity

import GCMS_to_csv
import data_loader
from GCMS import GCMS_add_Condition, GCMS_combine
from TGA_dl import load_model, smooth_data, load_model_ignore_mismatch
from FTIR_dl import preprocess_FTIR_data, train_and_evaluate, predict_and_plot
from main import TGA_augmentation, MVAE_MoE
from models.ByproductPredictorCNN import ByproductPredictorCNN
from models.TemperatureToCompositionPredictor import TemperatureToCompositionPredictor
from models.TemperatureToDataPredictorCNN import TemperatureToDataPredictorCNN
from models.ml import polynomial_regression, random_forest, support_vector_regression, k_nearest_neighbors, \
    linear_regression, compare_models

from scipy.interpolate import griddata, PchipInterpolator, Akima1DInterpolator, Rbf, interp1d
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from scipy.signal import savgol_filter

from scipy.stats import pearsonr

from preprocessing import reduce_by_temperature, interpolate_temperature, reduce_to_one_degree_interval


def process_TGA_data(TGA_data, cat, target_temp):
    """입력 값에 따라 TGA 데이터를 처리하고 보간을 진행."""
    # 데이터 처리 및 보간
    from TGA import group
    data_for_return, temp1, temp2 = group.process_group_for_TGA(TGA_data, cat, target_temp)
    data = [reduce_by_temperature(d) for d in data_for_return]

    print()

    # 0: 온도, 1: Weight%, 2: deriv. Weight
    data = [interpolate_temperature(d, 40, 800) for d in data]

    data = [reduce_to_one_degree_interval(d) for d in data]

    return np.array(data), temp1, temp2

def process_FTIR_data(FTIR_data, cat, target_tem):
    processed_FTIR = []

    group_number = {
        'NoCat': [0, 4, 8, 12],
        'PtC': [1, 5, 9, 13],
        'RuC': [2, 6, 10, 14],
        'RN': [3, 7, 11, 15]
    }

    if cat not in group_number:
        raise ValueError("허용되지 않은 촉매")

    if 250 < target_tem < 300:
        tem1, tem2 = 0, 1
    elif 300 < target_tem < 350:
        tem1, tem2 = 1, 2
    elif 350 < target_tem < 400:
        tem1, tem2 = 2, 3
    else:
        raise ValueError("허용되지 않은 온도")

    # 촉매, 온도에 따른 데이터들만 추출
    for tem in [tem1, tem2]:
        for i in range(3):  # 0, 1, 2를 반복하여 * 3 + i 부분을 처리
            processed_FTIR.append(FTIR_data[(group_number[cat][tem] * 3 + i)][1])

    # return np.array(processed_FTIR), tem1, tem2, FTIR_data[0][0]
    return tem1, tem2, FTIR_data[0][0]



# 온도 간의 비율 계산
def calculate_ratio(target_temp, X1_temp, X2_temp):
    return (target_temp - X1_temp) / (X2_temp - X1_temp)

# 예측값이 주어진 목표 온도와 그 근처의 온도 간의 비율을 얼마나 잘 따르는지를 평가
def consistency_score(predictions, X1, X2, ratio):
    # 예측값과 기준 온도들 사이의 거리 계산
    D1 = np.linalg.norm(predictions - X1)
    D2 = np.linalg.norm(predictions - X2)
    D_total = D1 + D2

    if D_total == 0:
        predicted_ratio = 0.5
    else:
        predicted_ratio = D2 / D_total

    # 일관성 점수 계산
    consistency = 1 - abs(predicted_ratio - ratio)
    return consistency

# 정확성과 일관성 평가 결과를 결합하여 하이브리드 점수를 계산합니다.
def hybrid_score(predictions, X1, X2, ratio, alpha=0.1):
    # 정확성 평가 (MSE 사용)
    accuracy_mse = mean_squared_error(predictions, (X1 + X2) / 2)

    # 일관성 평가
    consistency = consistency_score(predictions, X1, X2, ratio)

    # 최종 점수 계산
    result = alpha * (1 / (1 + accuracy_mse)) + (1 - alpha) * consistency

    return result

# 주어진 목표 온도와 그에 맞는 예측값을 기준으로 정확성과 일관성을 계산해 최종 평가 점수를 도출
def evaluate_predictions_with_ratio(results, data, alpha=0.8):
    """
    주어진 결과와 데이터를 비교하여 각 모델의 성능을 평가합니다.

    Parameters:
    - results: 각 목표 온도에 대해 모델의 예측값을 포함한 리스트
    - TGA_data: 250°C, 300°C, 350°C, 400°C에서의 실제 TGA 데이터
    - alpha: 정확성과 일관성 간의 가중치를 조정하는 파라미터 (기본값 0.8)

    Returns:
    - evaluation_results: 각 목표 온도에 대한 모델 평가 점수를 포함한 리스트
    """


    # 이거 변수명이 TGA_data인데, GCMS_data가 맞는거 같아요오오오오오 여쭤보기


    evaluation_results = []  # 최종 평가 결과를 저장할 리스트

    for result in results:
        target_temp = result['temperature']  # result 딕셔너리에서 목표 온도를 가져옴

        # 목표 온도에 따라 기준 온도와 해당하는 TGA 데이터를 설정
        if target_temp <= 300:
            X1_temp, X2_temp = 250, 300  # 기준 온도: 250°C, 300°C
            X1, X2 = data[0], data[1]  # 해당 온도에서의 실제 TGA 데이터

        elif target_temp <= 350:
            X1_temp, X2_temp = 300, 350  # 기준 온도: 300°C, 350°C
            X1, X2 = data[1], data[2]  # 해당 온도에서의 실제 TGA 데이터

        elif target_temp <= 400:
            X1_temp, X2_temp = 350, 400  # 기준 온도: 350°C, 400°C
            X1, X2 = data[2], data[3]  # 해당 온도에서의 실제 TGA 데이터

        else:
            # 목표 온도가 허용 범위를 벗어난 경우 예외 처리
            raise ValueError("Target temperature is out of range. Must be <= 400°C.")

        # 목표 온도와 기준 온도들 간의 비율 계산
        ratio = calculate_ratio(target_temp, X1_temp, X2_temp)

        models = result['models']  # result 딕셔너리에서 모델별 예측값을 가져옴
        model_evaluations = {'temperature': target_temp, 'evaluations': {}}  # 평가 결과 저장을 위한 딕셔너리

        # 각 모델의 예측값을 기반으로 성능 점수 계산
        for model_name, predictions in models.items():
            score = hybrid_score(predictions, X1, X2, ratio, alpha)  # 하이브리드 점수 계산
            model_evaluations['evaluations'][model_name] = score  # 모델 이름과 점수를 저장

        evaluation_results.append(model_evaluations)  # 최종 평가 결과 리스트에 추가

    return evaluation_results, X1.reshape(1, data.shape[1]), X2.reshape(1, data.shape[1]) # 각 목표 온도와 모델별 평가 점수를 반환, 평가에 사용된 데이터들 반환

# 주어진 목표 온도와 그에 맞는 예측값을 기준으로 정확성과 일관성을 계산해 최종 평가 점수를 도출
def evaluate_predictions_with_ratio_MoE(results, data, names, target_temp, alpha=0.8):
    """
    주어진 결과와 데이터를 비교하여 각 모델의 성능을 평가합니다.

    Parameters:
    - results: 각 목표 온도에 대해 모델의 예측값을 포함한 리스트
    - TGA_data: 250°C, 300°C, 350°C, 400°C에서의 실제 TGA 데이터
    - alpha: 정확성과 일관성 간의 가중치를 조정하는 파라미터 (기본값 0.8)

    Returns:
    - evaluation_results: 각 목표 온도에 대한 모델 평가 점수를 포함한 리스트
    """


    # 이거 변수명이 TGA_data인데, GCMS_data가 맞는거 같아요오오오오오 여쭤보기


    evaluation_results = []  # 최종 평가 결과를 저장할 리스트

    for i in range(len(results)):

        # 목표 온도에 따라 기준 온도와 해당하는 TGA 데이터를 설정
        if target_temp <= 300:
            X1_temp, X2_temp = 250, 300  # 기준 온도: 250°C, 300°C
            X1, X2 = data[0], data[1]  # 해당 온도에서의 실제 TGA 데이터

        elif target_temp <= 350:
            X1_temp, X2_temp = 300, 350  # 기준 온도: 300°C, 350°C
            X1, X2 = data[1], data[2]  # 해당 온도에서의 실제 TGA 데이터

        elif target_temp <= 400:
            X1_temp, X2_temp = 350, 400  # 기준 온도: 350°C, 400°C
            X1, X2 = data[2], data[3]  # 해당 온도에서의 실제 TGA 데이터

        else:
            # 목표 온도가 허용 범위를 벗어난 경우 예외 처리
            raise ValueError("Target temperature is out of range. Must be <= 400°C.")

        # 목표 온도와 기준 온도들 간의 비율 계산
        ratio = calculate_ratio(target_temp, X1_temp, X2_temp)

        model_evaluations = {'temperature': target_temp, 'evaluations': {}}  # 평가 결과 저장을 위한 딕셔너리

        # 각 모델의 예측값을 기반으로 성능 점수 계산
        score = hybrid_score(results[i], X1, X2, ratio, alpha)  # 하이브리드 점수 계산
        model_evaluations['evaluations'][names[i]] = score  # 모델 이름과 점수를 저장

        evaluation_results.append(model_evaluations)  # 최종 평가 결과 리스트에 추가

    return evaluation_results, X1.reshape(1, data.shape[1]), X2.reshape(1, data.shape[1]) # 각 목표 온도와 모델별 평가 점수를 반환, 평가에 사용된 데이터들 반환


# 정규화 함수
def normalize():
    data = read_csv('dataset/combined_GCMS.csv')
    data = data[data['Catalyst'] == cat]
    temps = [250, 300, 350, 400]

    combined_data = []

    for temp in temps:
        # 현재 온도에 해당하는 데이터 선택
        temp_data = data[data['temp'] == temp]

        # 'Value' 열에서 필요한 데이터를 추출하고, 마지막 값 제외
        values = temp_data['Value'].values[:-1]

        # 값의 범위가 너무 크면, 스케일을 줄이기 위한 정규화 적용 (예: 값을 로그 스케일로 변환)
        scaled_values = np.log1p(values)  # 로그 변환으로 큰 값을 줄이는 방식

        # 소프트맥스 적용 (안정성을 위해 최대값을 빼는 방식 사용)
        exp_values = np.exp(scaled_values - np.max(scaled_values))
        softmax_values = exp_values / np.sum(exp_values)

        # 결과를 리스트에 추가
        combined_data.append(softmax_values)

    return np.array(combined_data)

def rank_models(evaluation_results):
    """
    모델의 성능 평가 결과를 받아 모델별 성능을 순위로 매깁니다.
    각 모델의 순위와 최종 점수를 함께 반환합니다.
    """
    model_scores = {}

    # 평가 결과를 각 모델별로 정리
    for result in evaluation_results:
        models = result['evaluations']
        for model_name, score in models.items():
            if model_name not in model_scores:
                model_scores[model_name] = []
            model_scores[model_name].append(score)

    # 각 모델의 평균 점수 계산 (Hybrid Score)
    final_scores = {}
    for model_name, scores in model_scores.items():
        # 각 모델의 Hybrid Score 평균을 계산
        avg_score = np.mean(scores)
        final_scores[model_name] = avg_score

    # 모델별 최종 점수 계산 및 정규화는 필요 없음 (이미 Hybrid Score를 사용하므로)

    # 점수 순서로 모델을 정렬하고 순위와 점수를 함께 반환 (1위부터 시작)
    ranked_models = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    ranked_models_with_positions = {model[0]: {'rank': rank + 1, 'score': model[1]} for rank, model in
                                    enumerate(ranked_models)}

    return ranked_models_with_positions


# 모델별 결과 플롯 함수
def plot_results(results, X, option=None):
    """
    results 리스트를 받아서 각 온도에 대한 모델별 예측 결과를 플롯팅합니다.
    또한 X[0], X[1], X[2], X[3]에는 각각 250도, 300도, 350도, 400도의 실제 값을 저장하여 점선으로 플롯합니다.

    Parameters:
    - results: generate_data 함수에서 생성된 결과 리스트
    - X: 실제 값들 (250도, 300도, 350도, 400도에서의 실제 데이터 리스트)
    """
    # X에 있는 실제 데이터는 250도, 300도, 350도, 400도에 대응
    actual_temps = [250, 300, 350, 400]

    for result in results:
        temp = result['temperature']
        models = result['models']

        plt.figure(figsize=(10, 6))

        # 모델 예측 값 플롯
        for model_name, predictions in models.items():
            time_range = np.arange(len(predictions))

            if option == 'TGA':
                predictions = np.clip(predictions, None, 0.2)

            plt.plot(time_range, predictions, label=model_name)

        # 실제 값(X)를 점선으로 플롯 (각각 250도, 300도, 350도, 400도 데이터)
        for i, actual_temp in enumerate(actual_temps):
            if len(X) > i:  # X 배열이 해당 데이터를 가지고 있는 경우만 플롯
                time_range = np.arange(len(X[i]))
                plt.plot(time_range, X[i], linestyle='--', label=f'Actual {actual_temp}°C')

        plt.title(f'Predicted Byproducts at Temperature: {temp}°C')
        plt.xlabel('Time')
        plt.ylabel('Predicted Byproducts')
        plt.legend()
        plt.grid(True)
        plt.show()


# PyTorch 모델 평가 함수
def evaluate_model(model, device, desired_temps):
    """desired_temps 리스트를 받아서 모델을 평가하고 결과를 반환."""
    model.eval()

    # 리스트를 먼저 하나의 numpy 배열로 변환
    if isinstance(desired_temps, list):
        desired_temps = np.array(desired_temps)

    # numpy 배열을 tensor로 변환
    new_temperatures = torch.tensor(desired_temps, dtype=torch.float32).to(device)

    # (batch_size, 1) 형태로 변환 (2D 형태)
    new_temperatures = new_temperatures.unsqueeze(1)

    with torch.no_grad():
        predicted_byproducts = model(new_temperatures)

    # 결과를 numpy로 변환
    predicted_byproducts = predicted_byproducts.detach().cpu().numpy()

    # 결과를 부드럽게 처리
    # predicted_byproducts = smooth_data(predicted_byproducts, sigma=2)

    return predicted_byproducts


# 보간법 추가 (Linear, PCHIP, RBF, Akima)
def linear_spline_interpolation(X, y, X_pred):
    # 보간 가능한 범위로 제한
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)

    # X_pred의 값을 X 범위 내로 제한
    X_pred_clipped = np.clip(X_pred, X_min, X_max)

    # 2차원 보간 수행
    result = griddata(X, y, X_pred_clipped, method='linear')

    # NaN 값은 가장 가까운 값을 사용하도록 처리
    nan_mask = np.isnan(result)
    if np.any(nan_mask):
        result[nan_mask] = griddata(X, y, X_pred_clipped[nan_mask], method='nearest')

    return result


def pchip_interpolation(X, y, X_pred):
    y_pred = []
    for t in np.unique(X[:, 1]):
        indices = X[:, 1] == t
        X_t = X[indices, 0]
        y_t = y[indices]
        interp_func = PchipInterpolator(X_t, y_t)
        X_pred_t = X_pred[X_pred[:, 1] == t, 0]
        y_pred.extend(interp_func(X_pred_t))
    return np.array(y_pred)


def rbf_interpolation(X, y, X_pred):
    interp_func = Rbf(X[:, 0], X[:, 1], y, function='linear')
    return interp_func(X_pred[:, 0], X_pred[:, 1])


def akima_interpolation(X, y, X_pred):
    y_pred = []
    for t in np.unique(X[:, 1]):
        indices = X[:, 1] == t
        X_t = X[indices, 0]
        y_t = y[indices]

        if len(X_t) < 2:
            y_pred.extend([np.nan] * len(X_pred[X_pred[:, 1] == t]))
            continue

        # Akima 보간 함수 정의
        interp_func = Akima1DInterpolator(X_t, y_t)
        X_pred_t = X_pred[X_pred[:, 1] == t, 0]

        # X_pred_t 값을 X_t의 범위 내로 클리핑
        X_pred_t_clipped = np.clip(X_pred_t, np.min(X_t), np.max(X_t))

        y_pred_t = interp_func(X_pred_t_clipped)

        # NaN 값 처리 (가장 가까운 값으로 대체)
        nan_mask = np.isnan(y_pred_t)
        if np.any(nan_mask):
            nearest_interp = Akima1DInterpolator(X_t, y_t)
            y_pred_t[nan_mask] = nearest_interp(X_pred_t_clipped[nan_mask])

        y_pred.extend(y_pred_t)

    return np.array(y_pred)


# 데이터 생성 함수 (모델 학습 및 보간 적용)
def generate_data_GCMS(data, model, desired_temps, device):
    results = []  # 결과를 저장할 리스트

    time = np.arange(data.shape[1])
    X = []
    y = []

    temperatures = [250, 300, 350, 400]

    # 데이터 준비
    for idx, t in enumerate(time):
        for i, temp in enumerate(temperatures):
            X.append([temp, t])
            y.append(data[i][idx])

    X = np.array(X)
    y = np.array(y)

    # 여러 온도 조건에서 예측
    for desired_temp in desired_temps:

        if desired_temp in [250, 300, 350, 400]:
            continue

        if isinstance(desired_temp, (int, float)):
            X_pred = np.array([[desired_temp, t] for t in time])
        else:
            X_pred = np.array([[temp, t] for temp in desired_temp for t in time])

        temp_results = {'temperature': desired_temp, 'models': {}}

        # 선형 회귀
        y_pred = linear_regression(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['Linear Regression'] = y_pred

        # 다항 회귀
        y_pred = polynomial_regression(X, y, X_pred, degree=3)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['Polynomial Regression (degree=3)'] = y_pred

        # 랜덤 포레스트
        y_pred = random_forest(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['Random Forest'] = y_pred

        # 서포트 벡터 회귀
        y_pred = support_vector_regression(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['SVR'] = y_pred

        # K-최근접 이웃
        y_pred = k_nearest_neighbors(X, y, X_pred, n_neighbors=5)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['K-Nearest Neighbors'] = y_pred

        # # Linear Spline
        # y_pred = linear_spline_interpolation(X, y, X_pred)
        # y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        # temp_results['models']['Linear Spline'] = y_pred


        # # PCHIP 보간
        # y_pred = pchip_interpolation(X, y, X_pred)
        # y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        # temp_results['models']['PCHIP Interpolation'] = y_pred
        #
        # # RBF 보간
        # y_pred = rbf_interpolation(X, y, X_pred)
        # y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        # temp_results['models']['RBF Interpolation'] = y_pred
        #
        # # Akima 보간
        # y_pred = akima_interpolation(X, y, X_pred)
        # y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        # temp_results['models']['Akima Interpolation'] = y_pred

        if model is not None and device is not None:
            torch_pred = evaluate_model(model, device, [desired_temp])
            torch_pred = np.clip(torch_pred.flatten(), None, 0.2)  # 0.2로 값 제한
            temp_results['models']['PyTorch Model'] = torch_pred

        results.append(temp_results)

    return results

# 데이터 생성 함수 (모델 학습 및 보간 적용)
def generate_data_TGA(tga_data, data, model, desired_temps, cat, temps, device):
    catalyst = {
        "NoCat": 0,
        "PtC": 1,
        "RuC": 2,
        "RN": 3
    }
    results = []  # 결과를 저장할 리스트

    time = np.arange(tga_data.shape[1])
    X = []
    y = []

    temperatures = [250, 300, 350, 400]

    # 데이터 준비
    for idx, t in enumerate(time):
        for i, temp in enumerate(temperatures):
            X.append([temp, t])
            y.append(tga_data[i][idx])

    X = np.array(X)
    y = np.array(y)

    # 여러 온도 조건에서 예측
    for desired_temp in desired_temps:

        if desired_temp in [250, 300, 350, 400]:
            continue

        if isinstance(desired_temp, (int, float)):
            X_pred = np.array([[desired_temp, t] for t in time])
        else:
            X_pred = np.array([[temp, t] for temp in desired_temp for t in time])

        temp_results = {'temperature': desired_temp, 'models': {}}

        # 선형 회귀
        y_pred = linear_regression(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['Linear Regression'] = y_pred

        # 다항 회귀
        y_pred = polynomial_regression(X, y, X_pred, degree=3)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['Polynomial Regression (degree=3)'] = y_pred

        # 랜덤 포레스트
        y_pred = random_forest(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['Random Forest'] = y_pred

        # 서포트 벡터 회귀
        y_pred = support_vector_regression(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['SVR'] = y_pred

        # K-최근접 이웃
        y_pred = k_nearest_neighbors(X, y, X_pred, n_neighbors=5)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['K-Nearest Neighbors'] = y_pred

        # # Linear Spline
        # y_pred = linear_spline_interpolation(X, y, X_pred)
        # y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        # temp_results['models']['Linear Spline'] = y_pred


        # # PCHIP 보간
        # y_pred = pchip_interpolation(X, y, X_pred)
        # y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        # temp_results['models']['PCHIP Interpolation'] = y_pred
        #
        # # RBF 보간
        # y_pred = rbf_interpolation(X, y, X_pred)
        # y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        # temp_results['models']['RBF Interpolation'] = y_pred
        #
        # # Akima 보간
        # y_pred = akima_interpolation(X, y, X_pred)
        # y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        # temp_results['models']['Akima Interpolation'] = y_pred

        if model is not None and device is not None:
            torch_pred = evaluate_model(model, device, [desired_temp])
            torch_pred = np.clip(torch_pred.flatten(), None, 0.2)  # 0.2로 값 제한
            temp_results['models']['PyTorch Model'] = torch_pred

        results.append(temp_results)

    return results

def generate_data_FTIR(ftir_data, data, model, desired_temps, cat, temps, device):

    results = []  # 결과를 저장할 리스트

    time = np.array(ftir_data[0])
    X = []
    y = []

    temperatures = [250, 300, 350, 400]

    # 데이터 준비
    for idx, t in enumerate(time):
        for i, temp in enumerate(temperatures):
            X.append([temp, t])
            y.append(ftir_data[i][idx])

    X = np.array(X)
    y = np.array(y)

    # 여러 온도 조건에서 예측
    for desired_temp in desired_temps:

        if desired_temp in [250, 300, 350, 400]:
            continue

        if isinstance(desired_temp, (int, float)):
            X_pred = np.array([[desired_temp, t] for t in time])
        else:
            X_pred = np.array([[temp, t] for temp in desired_temp for t in time])

        temp_results = {'temperature': desired_temp, 'models': {}}

        # 선형 회귀
        y_pred = linear_regression(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['Linear Regression'] = y_pred

        # 다항 회귀
        y_pred = polynomial_regression(X, y, X_pred, degree=3)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['Polynomial Regression (degree=3)'] = y_pred

        # 랜덤 포레스트
        y_pred = random_forest(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['Random Forest'] = y_pred

        # 서포트 벡터 회귀
        y_pred = support_vector_regression(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['SVR'] = y_pred

        # K-최근접 이웃
        y_pred = k_nearest_neighbors(X, y, X_pred, n_neighbors=5)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['K-Nearest Neighbors'] = y_pred

        if model is not None and device is not None:
            torch_pred = evaluate_model(model, device, [desired_temp])
            torch_pred = np.clip(torch_pred.flatten(), None, 0.2)  # 0.2로 값 제한
            temp_results['models']['PyTorch Model'] = torch_pred

        results.append(temp_results)

    return results


def train_and_evaluate_models(TGA_data, FTIR_data, GCMS_data, target_temp):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.decomposition import PCA

    # PyTorch 텐서를 numpy 배열로 변환 (detach 필요 시 수행)
    if isinstance(TGA_data, torch.Tensor):
        TGA_data = TGA_data.detach().cpu().numpy()
    if isinstance(FTIR_data, torch.Tensor):
        FTIR_data = FTIR_data.detach().cpu().numpy()
    if isinstance(GCMS_data, torch.Tensor):
        GCMS_data = GCMS_data.detach().cpu().numpy()

    # 데이터 결합
    X = np.hstack((TGA_data, FTIR_data)).astype(np.float32)  # 입력 변수
    y = GCMS_data.astype(np.float32)  # 타겟 변수

    # 데이터 분할
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 데이터 샘플링 (예: 5000개 샘플 사용)
    sample_size = 5000
    X_train = X_train_full[:sample_size]
    y_train = y_train_full[:sample_size]

    # 입력 변수 스케일링
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # PCA 적용
    pca = PCA(n_components=100)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    # 결과를 저장할 딕셔너리
    MoE_results = {'temperature': target_temp, 'models': {}}

    # 선형 회귀 모델 훈련 및 예측
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # 예측값을 처리하여 평균을 구함
    processing = []
    for i in range(10):
        processing.append(pd.DataFrame(y_pred_lr[i]).mean().values[0])  # .values[0]로 단일 값 추출

    # 처리된 데이터를 MoE_results에 저장
    MoE_results['models']['LinearRegression'] = processing

    # LinearRegression 모델의 예측값이 단일 값일 경우 배열로 변환
    linear_regression_floats = [pred for pred in MoE_results['models']['LinearRegression']]

    # 배열로 변환
    MoE_results['models']['LinearRegression'] = np.array(linear_regression_floats)


    # 다항 회귀
    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)

    processing = []
    for i in range(10):
        processing.append(pd.DataFrame(y_pred_poly)[i].mean())

    MoE_results['models']['PolynomialRegression'] = processing



    # SVR (데이터 샘플링으로 인해 시간 단축)
    y_pred_svr = np.zeros_like(y_test)
    for i in range(y_train.shape[1]):
        svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        svr_model.fit(X_train, y_train[:, i])
        y_pred_svr[:, i] = svr_model.predict(X_test)

    processing = []
    for i in range(10):
        processing.append(pd.DataFrame(y_pred_svr)[i].mean())

    MoE_results['models']['SVR'] = processing



    # 랜덤 포레스트 (n_estimators 감소)
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    processing = []
    for i in range(10):
        processing.append(pd.DataFrame(y_pred_rf)[i].mean())

    MoE_results['models']['RandomForestRegression'] = processing



    # KNN 회귀
    knn_model = KNeighborsRegressor(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    processing = []
    for i in range(10):
        processing.append(pd.DataFrame(y_pred_knn)[i].mean())

    MoE_results['models']['KNN'] = processing


    return MoE_results


def train_model_FTIR(X_train, y_train):

    """
    다양한 회귀 모델을 학습하고 학습된 모델을 반환합니다.

    X_train: 학습 데이터의 입력 값 (N, 1) 형태의 numpy 배열
    y_train: 학습 데이터의 출력 값 (N,) 형태의 numpy 배열
    """

    models = {}

    # 선형 회귀 모델
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)  # 선형 회귀 모델 학습
    models['linear_model'] = linear_model  # 모델 저장

    # 3차 다항 회귀 모델
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X_train)  # X 데이터를 다항 특징으로 변환
    poly_model = LinearRegression().fit(X_poly, y_train)  # 다항 회귀 모델 학습
    models['poly_model'] = (poly_model, poly)  # 모델과 다항 변환기를 저장

    # 랜덤 포레스트 회귀 모델
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)  # 랜덤 포레스트 모델 학습
    models['rf_model'] = rf_model  # 모델 저장

    # 서포트 벡터 회귀 모델
    svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr_model.fit(X_train, y_train)  # 서포트 벡터 회귀 모델 학습
    models['svr'] = svr_model  # 모델 저장

    # K-최근접 이웃 회귀 모델
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)  # KNN 모델 학습
    models['knn'] = knn_model  # 모델 저장

    return models

def plot_predictions(original_data, X_pred, models):
    """
    원본 데이터와 모델의 예측을 시각화합니다.

    X_pred: 예측할 입력 값 (N, 1) 형태의 numpy 배열
    original_data: 원본 스펙트럼 데이터 배열. 각 스펙트럼은 (N,) 형태
    models: 학습된 회귀 모델의 딕셔너리
    """
    plt.figure(figsize=(14, 8))

    # 원본 데이터 시각화
    for i, data in enumerate(original_data):
        plt.plot(np.arange(len(data)), data, label=f'Original Spectrum {i+1}', alpha=0.6)  # 각 스펙트럼을 선형으로 그림

    # 모델 예측 결과 시각화
    for label, model_info in models.items():
        if label == 'poly_model':
            model, poly = model_info
            # X_pred를 다항 특징 변환 전에 2차원 형태로 변환
            X_pred_reshaped = X_pred.reshape(-1, 1)
            y_pred = model.predict(poly.transform(X_pred_reshaped))  # 다항 특징으로 변환 후 예측
        else:  # rf_model을 추가
            model = model_info
            # X_pred를 모델에 맞게 2차원으로 변환
            X_pred_reshaped = X_pred.reshape(-1, 1)
            y_pred = model.predict(X_pred_reshaped)  # 모델로 예측

        plt.plot(X_pred.flatten(), y_pred, label=f'Prediction ({label})', alpha=0.6, linestyle='--')  # 예측된 선을 그림

    # 그래프 제목과 축 레이블 설정
    plt.title('FT-IR Spectrum Prediction')
    plt.xlabel('Wavenumber [cm$^{-1}$]')
    plt.ylabel('Transmittance [%]')

    # 주요 피크 위치와 텍스트 레이블 추가 (예시)
    peak_positions = [800, 1000, 1600, 2900, 3400]  # 주요 파수 위치
    peak_texts = ['800 cm$^{-1}$', '1000 cm$^{-1}$', '1600 cm$^{-1}$', '2900 cm$^{-1}$', '3400 cm$^{-1}$']
    for pos, text in zip(peak_positions, peak_texts):
        closest_index = (np.abs(X_pred.flatten() - pos)).argmin()
        if closest_index >= len(original_data[0]):  # 인덱스가 데이터 길이를 넘지 않도록 조정
            closest_index = len(original_data[0]) - 1

        y_value = original_data[0][closest_index]
        plt.annotate(text, xy=(pos, y_value), xytext=(pos, y_value - 5),
                     textcoords='offset points', ha='center', fontsize=9)
    # 격자와 범례 표시
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()  # 레이아웃 조정
    plt.show()

def FTIR_process(FTIR_data, target_temp, device):
    MODEL_PATH = "pth/FTIR_model.pth"

    preprocessed_data = preprocess_FTIR_data(FTIR_data)
    # 입력 및 출력 데이터 설정
    temperature_data = np.array([250, 300, 350, 400], dtype=np.float32).reshape(-1, 1)
    output_data = preprocessed_data[0][:, 1, :]  # (4, 3476) 형태의 데이터

    # compare_models(np.asarray(output_data), target_temp[0], True)

    # PyTorch 텐서로 변환
    temperatures = torch.tensor(temperature_data).unsqueeze(1).to(device)  # (batch_size, 1, 1)
    outputs = torch.tensor(output_data).to(device)

    # 모델 초기화
    model = TemperatureToDataPredictorCNN(input_size=1).to(device)

    # 모델이 이미 저장되어 있으면 로드, 아니면 학습
    if os.path.exists(MODEL_PATH):
        print("Loading the existing model...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("Training the model...")
        train_and_evaluate(model, temperatures, outputs)
        torch.save(model.state_dict(), MODEL_PATH)  # 학습 후 모델 저장

    # 새로운 온도에서의 예측 및 시각화
    new_temperatures = torch.tensor([target_temp], dtype=torch.float32).unsqueeze(1).to(device)
    predict_ftir = predict_and_plot(model, preprocessed_data, new_temperatures)

    return predict_ftir, output_data

# def plt_tga(data, desired_temp, model_names, window_length=4, polyorder=3):
def plt_tga(data, desired_temp, model_names):
    """
    여러 배열 데이터를 주어진 레이블과 함께 그래프로 시각화하는 함수 (스무딩 적용).

    Args:
    data (numpy array): 수직으로 쌓인 배열 (shape: (n, m), n은 데이터셋 개수, m은 각 데이터셋의 길이)
    desired_temp (float or int): 원하는 온도, 그래프 제목이나 주석 등에 활용
    model_names (list of str): 각 데이터셋의 이름을 라벨로 사용
    window_length (int): 스무딩 윈도우의 길이 (홀수)
    polyorder (int): 다항식 차수
    """
    # x축에 사용할 인덱스 생성
    x_values = np.arange(data.shape[1])

    # 모델 레이블을 설정 (온도와 모델명 포함)
    group_labels = ['250°C', '300°C', f'{desired_temp}°C prediction'] + model_names

    # 그래프 그리기
    plt.figure(figsize=(10, 6))

    # 각 데이터셋에 대해 플롯을 그립니다
    for i in range(data.shape[0]):
        if i >= 2:
            # i가 2일 때부터는 투명도를 적용
            alpha_value = 0.5
        else:
            # i가 2 미만일 때는 기본 투명도 (1.0)
            alpha_value = 1.0

        plt.plot(x_values, np.abs(data[i]), label=group_labels[i], alpha=alpha_value)

    # 그래프의 제목, 축 라벨 및 범례 추가
    plt.title('')
    plt.xlabel('Temperature T (°C)')
    plt.ylabel('Weight (%)')
    plt.legend()

    # 그래프 보여주기
    plt.show()

def bar_graph(data, desired_temps, model_names, title, y_label='Area %', y_lim=70):
    """
    GC-MS 데이터를 시각화하는 함수
    """

    categories = ['Syringyl', 'Guaiacyl', 'Poly aromatics (C10~C21)', 'Other aromatics (C6~C20)',
                  'Alkanes', 'Cyclic', 'Fatty Acids', 'Alcohol', 'Glycerol derived', 'Other']
    colors = ['red', 'blue', 'yellow', 'green', 'purple', 'gray', 'pink', 'lightblue', 'orange', 'black']

    # 그룹 레이블을 설정 (온도와 모델명 포함)
    group_labels = ['250°C', '300°C', f'{desired_temps}°C prediction'] + model_names

    # 모델명이 길면 줄바꿈 적용 (optional)
    group_labels = [label.replace(' ', '\n') for label in group_labels]

    x = np.arange(data.shape[0])  # 행 인덱스 (0, 1, 2, 3)
    width = 0.15  # 막대 폭
    spacing = 1.0  # 그룹 간 간격

    # 막대그래프 생성
    plt.figure(figsize=(10, 6))

    # 각 열에 대해 막대그래프 그리기
    for i in range(data.shape[1]):
        plt.bar(x * (1 + spacing) + i * width, data[:, i] * 100, width, label=categories[i], color=colors[i])

    # 그래프 레이블 및 제목 설정
    plt.xlabel('')
    plt.ylabel(y_label)
    plt.ylim(0, y_lim)  # y축 범위 설정
    plt.title(title)

    # x축 레이블을 45도로 기울여서 표시
    plt.xticks(x * (1 + spacing) + (width * 5), group_labels, rotation=45, ha="right")  # x축 레이블을 기울이고 오른쪽 정렬

    plt.legend(title='Categories')
    plt.grid(True)

    # 그래프 출력 전에 tight_layout()을 추가하여 여백 조정
    plt.tight_layout()

    # 그래프 출력
    plt.show()

def bar_graph_paper(data, desired_temps, model_names, title, y_label='Area %', y_lim=70):
    """
    GC-MS 데이터를 시각화하는 함수
    """

    categories = ['Syringyl', 'Guaiacyl', 'Poly aromatics (C10~C21)', 'Other aromatics (C6~C20)',
                  'Alkanes', 'Cyclic', 'Fatty Acids', 'Alcohol', 'Glycerol derived', 'Other']
    colors = ['red', 'blue', 'yellow', 'green', 'purple', 'gray', 'pink', 'lightblue', 'orange', 'black']

    # 그룹 레이블을 설정 (온도와 모델명 포함)
    group_labels = ['Interpolation', 'Pytorch'] + model_names

    # 모델명이 길면 줄바꿈 적용 (optional)
    group_labels = [label.replace(' ', '\n') for label in group_labels]

    x = np.arange(data.shape[0])  # 행 인덱스 (0, 1, 2, 3)
    width = 0.15  # 막대 폭
    spacing = 1.0  # 그룹 간 간격

    # 막대그래프 생성
    plt.figure(figsize=(10, 6))

    # 각 열에 대해 막대그래프 그리기
    for i in range(data.shape[1]):
        plt.bar(x * (1 + spacing) + i * width, data[:, i] * 100, width, label=categories[i], color=colors[i])

    # 그래프 레이블 및 제목 설정
    plt.xlabel('')
    plt.ylabel(y_label)
    plt.ylim(0, y_lim)  # y축 범위 설정
    plt.title(title)

    # x축 레이블을 45도로 기울여서 표시
    plt.xticks(x * (1 + spacing) + (width * 5), group_labels, rotation=45, ha="right")  # x축 레이블을 기울이고 오른쪽 정렬

    plt.legend(title='Categories')
    plt.grid(True)

    # 그래프 출력 전에 tight_layout()을 추가하여 여백 조정
    plt.tight_layout()

    # 그래프 출력
    plt.show()


def compare_models_ftir(data, predict, target_temp, temp1, temp2):
    plt.figure(figsize=(12, 8))

    time = np.arange(data.shape[1])

    X = []
    y = []
    temperatures = [250, 300, 350, 400]
    for idx, t in enumerate(time):
        for i, temp in enumerate(temperatures):
            X.append([temp, t])
            y.append(data[i][idx])

    X_pred = np.array([[target_temp, t] for t in time])

    # 실제 온도 데이터 시각화
    plt.plot(time, np.abs(data[temp1]), label=f'{250 + 50 * temp1}°C')
    plt.plot(time, np.abs(data[temp2]), label=f'{250 + 50 * temp2}°C')

    ftir_model_results =  {'temperature': target_temp, 'models': {}}

    # predict 텐서 추가 (모델 예측값 시각화)
    plt.plot(time, predict.squeeze().numpy(), label=f'{target_temp}°C prediction', alpha=0.4)
    ftir_model_results['models']['PyTorch Model'] = predict[0].detach().numpy()


    # 선형 회귀
    y_pred = linear_regression(X, y, X_pred)
    plt.plot(time, np.abs(y_pred), label='Linear Regression', alpha=0.4)
    y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
    ftir_model_results['models']['Linear Regression'] = y_pred

    # 다항 회귀
    y_pred = polynomial_regression(X, y, X_pred, degree=3)
    plt.plot(time, np.abs(y_pred), label='Polynomial Regression (degree=3)', alpha=0.4)
    y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
    ftir_model_results['models']['Polynomial Regression'] = y_pred

    # 랜덤 포레스트
    y_pred = random_forest(X, y, X_pred)
    plt.plot(time, np.abs(y_pred), label='Random Forest', alpha=0.4)
    y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
    ftir_model_results['models']['Random Forest'] = y_pred

    # 서포트 벡터 회귀
    y_pred = support_vector_regression(X, y, X_pred)
    plt.plot(time, y_pred, label='SVR', alpha=0.4)
    y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
    ftir_model_results['models']['SVR'] = y_pred

    # K-최근접 이웃
    y_pred = k_nearest_neighbors(X, y, X_pred, n_neighbors=5)
    plt.plot(time, np.abs(y_pred), label='K-Nearest Neighbors', alpha=0.4)
    y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
    ftir_model_results['models']['K-Nearest Neighbors'] = y_pred

    plt.ylabel('%T')
    plt.xlabel('Wavenumber [cm-1]')
    plt.title('Prediction FT-IR')
    plt.legend()

    # x축과 y축 눈금을 사용자 지정 값으로 설정
    new_xticks = np.linspace(4000, 650, num=6)  # 예: 4000에서 650까지의 눈금을 6개로 설정
    new_yticks = np.linspace(110, 60, num=6)  # 예: 60에서 110까지의 눈금을 6개로 설정
    plt.xticks(ticks=np.linspace(0, len(time) - 1, num=6), labels=np.round(new_xticks, 0))
    plt.yticks(ticks=np.linspace(plt.ylim()[0], plt.ylim()[1], num=6), labels=np.round(new_yticks, 0))

    plt.ylim(0, 4)
    plt.gca().invert_yaxis()  # y축을 반전시킴
    plt.gca().invert_xaxis()  # x축을 반전시킴

    plt.show()

    return ftir_model_results

def combine_MoE(out_put) :
    result = []

    for i in range(10) :
        result.append(pd.DataFrame(out_put.numpy())[i].mean())

    return result

def evaluation_MoE(y_interp, predictions_np) :
    result = []

    for i in predictions_np :
        result.append(mean_squared_error(y_interp, i))

    return result

if __name__ == '__main__':
    condition_data, TGA_data, FTIR_data, GCMS_data = data_loader.load_data(data_loader.ROOT_DIR)

    tga_model_path = 'pth/tga.pth'
    ftir_model_path = 'pth/FTIR_model.pth'
    gcms_model_path = 'pth/composition_model.pth'

    cat = 'NoCat'

    # GPU 유무에 따라서 cuda or cpu 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 예측할 온도 설정
    desired_temps = np.array([[275.0]])

    '''
    process_data => Load CNN => load_model
    => generate => sample => results => rank
    
    * 이 아래에 있던 주석은 main 함수 아래로 이동 됐습니다. *
    '''


    TGA = False
    FTIR = False
    GCMS = False
    MoE = True

    if TGA :
        # 입력 값에 따라 1 ~ 16.xls 중 필요한 파일 선정 및 온도 설정
        processed_data, temp1, temp2 = process_TGA_data(TGA_data, cat, desired_temps[0])

        # 모델 정의
        model = ByproductPredictorCNN(1, 761).to(device)

        # 모델 가중치 로드
        load_model(model, tga_model_path, device)

        # 예측, 모델을 사용하여 주어진 온도에 대해 예측된 화합물 조성을 계산한 결과를 반환
        # data = generate_data_TGA(np.array(set(np.array(TGA_data[0][0], dtype=int))), model, desired_temps, device)
        temperatures = list(set(np.array(TGA_data[0][0], dtype=int)))

        # 모델 평가 ######## Taget 온도 바꾸려면 여기
        predicted_byproducts = evaluate_model(model, device, desired_temps)

        # Gaussian smoothing 적용
        predicted_byproducts_smoothed = smooth_data(predicted_byproducts, sigma=2)

        #generate_data_TGA()

        # 제너레이터를 리스트로 변환
        combine_tga_data = np.asarray([processed_data[i][2] for i in range(4)])

        # np.vstack에 리스트 전달
        result = generate_data_TGA(np.vstack(combine_tga_data), predicted_byproducts, model, desired_temps, cat, [temp1, temp2], device)

        # 모델이 평가한 예측
        evaluation_results, X1, X2 = evaluate_predictions_with_ratio(result, combine_tga_data)

        # 평가 결과를 기반으로 모델의 성능을 순위로 나열
        ranked_models = rank_models(evaluation_results)

        print("TGA 모델 순위:", ranked_models)

        # 모델을 이용해 예측 수행
        predicted_byproducts = evaluate_model(model, device, desired_temps)

        # 모델명과 예측값을 각각 저장
        model_names = [model_name for model_name, predictions in result[0]['models'].items() if
                       model_name != 'PyTorch Model']
        predictions_np = np.array([predictions for model_name, predictions in result[0]['models'].items() if model_name != 'PyTorch Model'])

        # 예측 결과 출력
        plt_tga(np.vstack((X1, X2, predicted_byproducts, predictions_np)), desired_temps[0][0], model_names)


    if FTIR :
        # 입력 값에 따라 1 ~ 16.xls 중 필요한 파일 선정 및 온도 설정
        temp1, temp2, time_series = process_FTIR_data(FTIR_data, cat, desired_temps[0][0])

        result, FTIR_datas = FTIR_process(FTIR_data, desired_temps[0], device)

        predict_results = compare_models_ftir(FTIR_datas, result, desired_temps[0][0], temp1, temp2)

        evaluation_results, X1, X2 = evaluate_predictions_with_ratio([predict_results], FTIR_datas)

        # 평가 결과를 기반으로 모델의 성능을 순위로 나열
        ranked_models = rank_models(evaluation_results)

        print("FT-IR 모델 순위:", ranked_models)

    if GCMS :
        # 추출 파일이 없는 경우 추출을 진행
        if not (os.path.exists('dataset/GC-MS_to_csv/16.xls')):
            GCMS_to_csv.process_and_export_gcms_data(GCMS_data)

            # 파일명에 따라 촉매, 전처리 온도 컬럼을 추가
            path = 'dataset/GC-MS_to_csv/'
            GCMS_add_Condition.process_csv_files_in_directory(path)

        # GC-MS pdf에서 추출하여 합친 파일이 있는 경우 그대로 읽어와서 할당
        # 없는 경우 합친 파일 생성 후 할당
        if not (os.path.exists('dataset/combined_GCMS.csv')):
            GCMS_combine.combine_csv_files()

        combined_data = normalize()

        # 모델 정의
        model = TemperatureToCompositionPredictor(input_size=1,output_size=10).to(device)

        # 모델 가중치 로드
        load_model(model, gcms_model_path, device)

        # 예측, 모델을 사용하여 주어진 온도에 대해 예측된 화합물 조성을 계산한 결과를 반환
        result = generate_data_GCMS(combined_data, model, desired_temps, device)

        # 모델이 평가한 예측
        evaluation_results, X1, X2 = evaluate_predictions_with_ratio(np.array(result), combined_data)

        # 평가 결과를 기반으로 모델의 성능을 순위로 나열
        ranked_models = rank_models(evaluation_results)

        print("GC-MS 모델 순위:", ranked_models)


        # 모델을 이용해 예측 수행
        predicted_byproducts = evaluate_model(model, device, desired_temps)

        # 모델명과 예측값을 각각 저장
        model_names = [model_name for model_name, predictions in result[0]['models'].items() if model_name != 'PyTorch Model']
        predictions_np = np.array([predictions for model_name, predictions in result[0]['models'].items() if model_name != 'PyTorch Model'])

        # 예측 결과 출력
        bar_graph(np.vstack((X1, X2, predicted_byproducts, predictions_np)), desired_temps[0][0], model_names, 'Prediction GC-MS')

    if MoE :
        TGA_model_path = 'pth/tga.pth'
        TGA_model = ByproductPredictorCNN(1, 761).to(device)
        TGA_model.load_state_dict(torch.load(TGA_model_path, weights_only=True))
        TGA_model.eval()

        FTIR_model_path = 'pth/FTIR_model.pth'
        FTIR_model = TemperatureToDataPredictorCNN(input_size=1).to(device)
        FTIR_model.load_state_dict(torch.load(FTIR_model_path, weights_only=True))
        FTIR_model.eval()

        GCMS_model_path = 'pth/composition_model.pth'
        GCMS_model = TemperatureToCompositionPredictor(input_size=1, output_size=10).to(device)
        GCMS_model.load_state_dict(torch.load(GCMS_model_path, weights_only=True))
        GCMS_model.eval()

        new_temperatures = np.arange(200, 401, 0.01, dtype=np.float32).reshape(-1, 1)
        new_temperatures = torch.tensor(new_temperatures).unsqueeze(1).to(device)

        TGA_data = TGA_model(new_temperatures)
        FTIR_data = FTIR_model(new_temperatures)
        GCMS_data = GCMS_model(new_temperatures)

        # 모델 초기화
        input_dim1 = 761
        input_dim2 = 3476
        latent_dim = 256
        output_dim = 10
        num_experts = 3

        MoE_path = 'pth/MoE_012.pth'
        model = MVAE_MoE(input_dim1, input_dim2, latent_dim, output_dim, num_experts).to(device)
        model.load_state_dict(torch.load(MoE_path, weights_only=True))
        model.eval()

        load_model(model, MoE_path, device)

        # 모델 예측 수행
        with torch.no_grad():  # 평가 시에는 gradient를 계산하지 않음
            recon_x1, recon_x2, output, mu, log_var = model(TGA_data, FTIR_data)

        MoE_results_ML = train_and_evaluate_models(TGA_data, FTIR_data, GCMS_data ,desired_temps[0][0])
        MoE_result = combine_MoE(output)

        combined_data = normalize()

        gcms1, gcms2 = combined_data[0], combined_data[1]

        # 보간할 지점 (0.5 조건)
        x = np.array([0, 1])  # 조건의 위치 (0과 1)
        x_interp = 0.5  # 보간 지점

        # 각 항목별로 보간 수행
        y_interp = []
        for i in range(len(gcms1)):
            # 보간 함수 생성 (선형 보간)
            f = interp1d(x, [gcms1[i], gcms2[i]], kind='linear')
            y_interp.append(f(x_interp))

        y_interp = np.array(y_interp)

        # 유사도 측정: 유클리드 거리
        dist_0 = euclidean(gcms1, y_interp)
        dist_1 = euclidean(gcms2, y_interp)

        # 유사도 측정: 코사인 유사도
        cos_sim_0 = cosine_similarity([gcms1], [y_interp])[0][0]
        cos_sim_1 = cosine_similarity([gcms2], [y_interp])[0][0]

        # 결과 출력
        print("0 조건과 보간된 데이터 간의 유클리드 거리:", dist_0)
        print("1 조건과 보간된 데이터 간의 유클리드 거리:", dist_1)
        print("0 조건과 보간된 데이터 간의 코사인 유사도:", cos_sim_0)
        print("1 조건과 보간된 데이터 간의 코사인 유사도:", cos_sim_1)

        # 터무니없는 수치인지 판단
        if cos_sim_0 > 0.9 and cos_sim_1 > 0.9:
            print("보간된 데이터는 0과 1 조건의 데이터와 높은 유사도를 보여 터무니없는 수치가 아닙니다.")
        else:
            print("보간된 데이터는 유사도가 낮아 터무니없는 수치일 수 있습니다.")

        # 모델명과 예측값을 각각 저장
        MoE_results_ML['models']['PyTorch Model'] = MoE_result

        model_names = [model_name for model_name in MoE_results_ML['models'].keys() if model_name != 'PyTorch Model']
        predictions_np = [[pred[1] for pred in predictions] if isinstance(predictions[0], tuple) else predictions for model_name, predictions in MoE_results_ML['models'].items() if model_name != 'PyTorch Model']

        bar_graph_paper(np.vstack((y_interp, MoE_result, predictions_np)), desired_temps[0][0], model_names,
                  'Prediction MoE')

        predictions_np.append(MoE_result)
        model_names.append('PyTorch Model')

        #evaluation_results, X1, X2 = evaluate_predictions_with_ratio_MoE(predictions_np, combined_data, model_names, desired_temps[0][0])

        evaluation_results = evaluation_MoE(y_interp, predictions_np)

        # 평가 결과를 기반으로 모델의 성능을 순위로 나열
        ranked_models = rank_models(evaluation_results)

        print("MoE 모델 순위:", ranked_models)


        # 예측 결과 출력
        print("Reconstructed TGA Data:", recon_x1)
        print("Reconstructed FTIR Data:", recon_x2)
        print("Predicted GCMS Data:", output)
        print()

    # 입력 값에 따라 1 ~ 16.xls 중 필요한 파일 선정 및 온도 설정
    # TGA_data, _, _ = process_TGA_data(TGA_data, cat, target_temp)
    # model = ByproductPredictorCNN(1, 761).to('cuda')  # ByproductPredictorCNN 사용
    # load_model(model, tga_model_path, device)
    # rst = generate_data(np.asarray(FTIR_data)[:,:,1,:], model, desired_temps, device)
    # # print("결과",rst)
    # TGA_data_samples = [TGA_data[0][2], TGA_data[1][2], TGA_data[2][2], TGA_data[3][2]]
    #
    # evaluation_results = evaluate_predictions_with_ratio(rst, TGA_data_samples)
    # ranked_models = rank_models(evaluation_results)
    # print("TGA 모델 순위:", ranked_models)

    # FTIR_data = preprocess_FTIR_data(FTIR_data)
    # model = TemperatureToDataPredictorCNN(input_size=1).to('cuda')
    # load_model(model, ftir_model_path, device)
    # rst = generate_data(np.asarray(FTIR_data)[0,:,1,:], model, desired_temps, device)
    # FTIR_data_samples = [FTIR_data[0][0][1], FTIR_data[0][1][1], FTIR_data[0][2][1], FTIR_data[0][3][1]]
    # evaluation_results = evaluate_predictions_with_ratio(rst, FTIR_data_samples)
    # ranked_models = rank_models(evaluation_results)
    # print("FTIR 모델 순위:", ranked_models)
    # print("Data generation and plotting completed.")