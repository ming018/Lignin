import os

import numpy as np
import torch
from fastdtw import fastdtw
from pandas import read_csv
from sklearn.metrics import mean_squared_error, mean_absolute_error

import GCMS_to_csv
import data_loader
from GCMS import GCMS_add_Condition, GCMS_combine
from TGA_dl import process_TGA_data, load_model, smooth_data
from FTIR_dl import preprocess_FTIR_data
from models.ByproductPredictorCNN import ByproductPredictorCNN
from models.TemperatureToCompositionPredictor import TemperatureToCompositionPredictor
from models.TemperatureToDataPredictorCNN import TemperatureToDataPredictorCNN
from models.ml import polynomial_regression, random_forest, support_vector_regression, k_nearest_neighbors, \
    linear_regression

from scipy.interpolate import griddata, PchipInterpolator, Akima1DInterpolator, Rbf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import pearsonr


# 온도 간의 비율 계산
def calculate_ratio(target_temp, X1_temp, X2_temp):
    return (target_temp - X1_temp) / (X2_temp - X1_temp)


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


def hybrid_score(predictions, X1, X2, ratio, alpha=0.5):
    # 정확성 평가 (MSE 사용)
    accuracy_mse = mean_squared_error(predictions, (X1 + X2) / 2)

    # 일관성 평가
    consistency = consistency_score(predictions, X1, X2, ratio)

    # 최종 점수 계산
    return alpha * (1 / (1 + accuracy_mse)) + (1 - alpha) * consistency


def evaluate_predictions_with_ratio(results, TGA_data, alpha=0.8):
    evaluation_results = []

    for result in results:
        target_temp = result['temperature']  # result에 저장된 목표 온도

        # 기준 온도와 데이터 설정
        if target_temp <= 300:
            X1_temp, X2_temp = 250, 300
            X1, X2 = TGA_data[0], TGA_data[1]  # 250도, 300도 데이터
        elif target_temp <= 350:
            X1_temp, X2_temp = 300, 350
            X1, X2 = TGA_data[1], TGA_data[2]  # 300도, 350도 데이터
        elif target_temp <= 400:
            X1_temp, X2_temp = 350, 400
            X1, X2 = TGA_data[2], TGA_data[3]  # 350도, 400도 데이터
        else:
            raise ValueError("Target temperature is out of range. Must be <= 400°C.")

        ratio = calculate_ratio(target_temp, X1_temp, X2_temp)

        models = result['models']
        model_evaluations = {'temperature': target_temp, 'evaluations': {}}

        for model_name, predictions in models.items():
            score = hybrid_score(predictions, X1, X2, ratio, alpha)
            model_evaluations['evaluations'][model_name] = score

        evaluation_results.append(model_evaluations)

    return evaluation_results


# 정규화 함수
def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)
    return [(v - min_val) / (max_val - min_val) if max_val != min_val else 0 for v in values]


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
# 모델별 결과 플롯 함수 (실제 데이터 X를 추가하여 함께 플롯)
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
def generate_data(data, model, desired_temps, device):
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

        # Linear Spline
        y_pred = linear_spline_interpolation(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['Linear Spline'] = y_pred

        # PCHIP 보간
        y_pred = pchip_interpolation(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['PCHIP Interpolation'] = y_pred

        # RBF 보간
        y_pred = rbf_interpolation(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['RBF Interpolation'] = y_pred

        # Akima 보간
        y_pred = akima_interpolation(X, y, X_pred)
        y_pred = np.clip(y_pred, None, 0.2)  # 0.2로 값 제한
        temp_results['models']['Akima Interpolation'] = y_pred

        if model is not None and device is not None:
            torch_pred = evaluate_model(model, device, [desired_temp])
            torch_pred = np.clip(torch_pred.flatten(), None, 0.2)  # 0.2로 값 제한
            temp_results['models']['PyTorch Model'] = torch_pred

        results.append(temp_results)

    # plot_results(results, data, "TGA")
    return results


def normalization_check_graph(data, y_label='Area %', y_lim=70, title='Bar Graph: normalization_check_graph', group_labels=None):
    """
    normalization 데이터를 막대 그래프로 표현해서 시각적인체크 용도
    """

    categories = ['Syringyl', 'Guaiacyl', 'Poly aromatics (C10~C21)', 'Other aromatics (C6~C20)',
                  'Alkanes', 'Cyclic', 'Fatty Acids', 'Alcohol', 'Other', 'Other']
    colors = ['red', 'blue', 'yellow', 'green', 'purple', 'gray', 'pink', 'lightblue', 'black', 'orange']

    if group_labels is None:
        group_labels = ['250°C', '300°C', '350°C', '400°C']

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
    plt.xticks(x * (1 + spacing) + (width * 5), group_labels)  # 그룹 레이블 설정
    plt.legend(title='Categories')
    plt.grid(True)

    # 그래프 출력
    plt.show()


"""
GCMS interpolation한 값이 합이 1이 안되기 때문에
cross-entrophy전에 e넣으면 된다?
"""



if __name__ == '__main__':
    condition_data, TGA_data, FTIR_data, GCMS_data = data_loader.load_data(data_loader.ROOT_DIR)

    tga_model_path = 'pth/tga.pth'
    ftir_model_path = 'pth/FTIR_model.pth'
    gcms_model_path = 'pth/composition_model.pth'

    cat = 'NoCat'
    target_temp = 275  # not for use

    # GPU 유무에 따라서 cuda or cpu 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    desired_temps = np.arange(250, 401, 10, dtype=np.float32).reshape(-1, 1)

    '''
    process_data => Load CNN => load_model
    => generate => sample => results => rank
    '''
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

    data = read_csv('dataset/combined_GCMS.csv')
    data = data[data['Catalyst'] == cat]
    temps = [250, 300, 350, 400]


    # 이하, normalization
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

    combined_data = np.array(combined_data)

    # 일반화 체크 용
    normalization_check_graph(combined_data)

    model = TemperatureToCompositionPredictor(input_size=1,output_size=10).to(device)
    load_model(model, gcms_model_path, device)
    rst = generate_data(combined_data, model, desired_temps, device)
    evaluation_results = evaluate_predictions_with_ratio(rst, combined_data)
    ranked_models = rank_models(evaluation_results)
    print("FTIR 모델 순위:", ranked_models)
