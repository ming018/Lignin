import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 루트 디렉토리 설정
root_dir = 'dataset/after_csv/'

# 모든 하위 디렉토리를 순회하여 CSV 파일을 읽어 병합하는 함수
def merge_csv_files(directory_path, output_file=None):
    """
    주어진 경로에 있는 모든 CSV 파일을 병합하여 DataFrame으로 반환하는 함수.

    Parameters:
    directory_path (str): CSV 파일들이 저장된 경로
    output_file (str): 병합된 결과를 저장할 파일 경로 (선택적)

    Returns:
    pd.DataFrame: 병합된 DataFrame
    """
    data_frames = []

    # 주어진 경로에 있는 모든 파일을 순회하여 병합
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):  # 파일 확장자가 csv인 경우만 처리
            file_path = os.path.join(directory_path, filename)
            # CSV 파일을 읽어서 데이터프레임으로 저장
            df = pd.read_csv(file_path)
            # 데이터프레임을 리스트에 추가
            data_frames.append(df)

    # 여러 데이터프레임을 하나로 병합 (행 기준으로 이어붙이기)
    merged_df = pd.concat(data_frames, ignore_index=True)

    # 병합된 데이터를 파일로 저장할 경우
    if output_file:
        merged_df.to_csv(output_file, index=False)

    return merged_df


# 랜덤 포레스트 회귀 모델 함수
def TGA_RF(Cat, Temp, only_predict=False):
    if not(only_predict):
        # 데이터 로드 및 병합
        data = merge_csv_files(root_dir)

        # 데이터가 올바르게 로드되었는지 확인
        if data is None or data.empty:
            print("Error: Data could not be loaded or is empty.")
            return

        # Feature (X)와 Target (y) 설정
        X = data[['Time', 'Temperature', 'Catalyst', 'temp']]
        y1 = data[['Weight']]  # 출력 변수 1
        y2 = data[['Deriv. Weight']]  # 출력 변수 2

        # Step 3: 범주형 변수 'Catalyst'를 원-핫 인코딩
        X = pd.get_dummies(X, columns=['Catalyst'])

        # 사용된 피처 출력
        print("훈련에 사용된 피처들:")
        print(X.columns)

        # Step 4: 데이터 스케일링 (표준화)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 5: 랜덤 포레스트 회귀 모델 정의 (병렬 처리)
        rf_model1 = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
        rf_model2 = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

        # Step 6: 모델 학습
        rf_model1.fit(X_scaled, y1)
        rf_model2.fit(X_scaled, y2)

        # Step 7: 학습된 모델과 스케일러 저장
        joblib.dump(rf_model1, 'models/rf_model_tga1.pkl')
        joblib.dump(scaler, 'models/scaler_tga1.pkl')
        joblib.dump(rf_model2, 'models/rf_model_tga2.pkl')
        joblib.dump(scaler, 'models/scaler_tga2.pkl')

        print("랜덤 포레스트 모델과 스케일러를 저장했습니다.")

    # 저장된 모델과 스케일러 불러오기 (only_predict=True)
    rf_model1 = joblib.load('models/rf_model_tga1.pkl')
    scaler1 = joblib.load('models/scaler_tga1.pkl')

    rf_model2 = joblib.load('models/rf_model_tga2.pkl')
    scaler2 = joblib.load('models/scaler_tga2.pkl')

    # 하위 디렉터리까지 순회하며 모든 CSV 파일을 찾음
    csv_files = glob.glob('dataset/after_csv/*.csv', recursive=True)

    # 가열 온도의 최소값과 최대값을 찾기 위한 변수 초기화
    min_temp = float('inf')
    max_temp = float('-inf')

    # 각 CSV 파일에서 가열 온도 범위 추출
    for file in csv_files:
        data = pd.read_csv(file)
        c_values = data['Temperature']  # 가열 온도 컬럼이 'C'로 되어 있다고 가정
        file_min_temp = c_values.min()
        file_max_temp = c_values.max()

        # 전체 파일 중 최소값과 최대값 찾기
        if file_min_temp < min_temp:
            min_temp = file_min_temp
        if file_max_temp > max_temp:
            max_temp = file_max_temp

    # 최소 온도가 39도 미만이면 39도로 설정
    if min_temp < 39:
        min_temp = 39

    # 예측할 가열 온도 범위를 설정
    c_values = np.linspace(min_temp, max_temp, 22080)


    # 예측을 위한 데이터 생성
    X_test = pd.DataFrame({'C': c_values, 'Temperature': [Temp] * len(c_values)})

    # 원-핫 인코딩 처리 (훈련된 모델과 동일한 피처 구조 유지)
    X_test['Catalyst_' + Cat] = 1  # 예측하고자 하는 Catalyst에 대해 1로 설정

    # 나머지 Catalyst 열을 0으로 설정
    for other_catalyst in ['Catalyst_PT_C', 'Catalyst_RN', 'Catalyst_RU_C']:
        if 'Catalyst_' + Cat != other_catalyst:
            X_test[other_catalyst] = 0

    # 훈련 시 사용된 피처 순서와 맞춤
    expected_features = ['C', 'Temperature', 'Catalyst_NOCAT', 'Catalyst_PT_C', 'Catalyst_RN', 'Catalyst_RU_C']
    X_test = X_test.reindex(columns=expected_features, fill_value=0)

    # 스케일링 적용
    X_test_scaled = scaler1.transform(X_test)

    # 랜덤 포레스트 모델로 예측
    y_pred = rf_model1.predict(X_test_scaled)

    # 예측 결과를 DataFrame으로 저장 (TGA는 %와 % / C 두 가지 출력이 있으므로 각각 저장)
    predicted_data = pd.DataFrame({'C': c_values, '%': y_pred[:, 0], '% / C': y_pred[:, 1]})

    # 예측 결과를 CSV 파일로 저장
    output_file_path = f'/content/drive/MyDrive/lignin/TGA/pkl(Random forest)/predict/{Cat}_{Temp}_predicted_tga.csv'
    predicted_data.to_csv(output_file_path, index=False)

    print(f"{Temp}℃에서 {Cat}의 랜덤 포레스트 예측 데이터를 {output_file_path}에 저장했습니다.")