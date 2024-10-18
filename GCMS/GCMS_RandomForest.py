import os
import pandas as pd
import numpy as np  # NumPy를 직접 import
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

#### 수정중 #####

# TGA 데이터 불러오기 (모든 CSV 파일을 병합)
tga_data = pd.read_csv('../dataset/combined_file.csv')
gcms_data = pd.read_csv('../dataset/combined_GCMS.csv')

# TGA와 GC-MS 데이터를 Catalyst와 Temperature를 기준으로 병합
if not tga_data.empty and not gcms_data.empty:
    merged_data = pd.merge(tga_data, gcms_data, on=['Catalyst', 'Temperature'], how='inner')

    # 필요한 열 선택 (Catalyst와 Temperature 포함)
    tga_features = merged_data[['Temperature', 'Weight', 'Deriv.Weight', 'Catalyst', 'temp']]
    gcms_target = merged_data['Value']  # 병합된 후 GC-MS 값 사용

    # Catalyst에 대한 원-핫 인코딩 (Catalyst 종류: 'NOCAT', 'PT_C', 'RN', 'RU_C')
    encoder = OneHotEncoder(sparse_output=False, categories=[['NOCAT', 'PT_C', 'RN', 'RU_C']])
    catalyst_encoded = encoder.fit_transform(merged_data[['Catalyst']])

    # 스케일링
    scaler = StandardScaler()
    tga_numerical_scaled = scaler.fit_transform(merged_data[['C', '%', '% / C', 'Temperature']])

    # 스케일링된 수치 데이터와 원-핫 인코딩된 Catalyst 데이터를 결합 (NumPy 사용)
    tga_scaled = pd.DataFrame(
        data = np.hstack([tga_numerical_scaled, catalyst_encoded]),
        columns = ['C', '%', '% / C', 'Temperature'] + encoder.get_feature_names_out(['Catalyst']).tolist()
    )

    # 랜덤 포레스트 모델 초기화 및 전체 데이터로 학습
    rf_model_tga = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model_tga.fit(tga_scaled, gcms_target)  # 병합된 데이터를 사용하여 학습

    # 성능 평가
    y_pred_tga = rf_model_tga.predict(tga_scaled)  # 학습 데이터에 대해 예측
    mse_tga = mean_squared_error(gcms_target, y_pred_tga)
    print(f"TGA 모델 MSE (전체 데이터): {mse_tga}")

    # TGA 모델 저장
    joblib.dump(rf_model_tga, './')
    print("TGA 모델이 'rf_model_GC-MS_tga.pkl'로 저장되었습니다.")
else:
    print("해당 조건을 만족하는 TGA 또는 GC-MS 데이터가 없습니다.")
