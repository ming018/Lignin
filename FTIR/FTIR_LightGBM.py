import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import joblib
import numpy as np

def FTIR_GBM(only_predict = False) :
    if only_predict:
        # 병합된 데이터 로드
        data = pd.read_csv('/content/drive/MyDrive/lignin/FT-IRdata.csv')

        # 특성과 타겟 설정 (입력: 온도, 파장(cm), 첨가물 / 출력: 흡수율)
        X = data[['Temperature', 'cm', 'Catalyst']]
        y = data['%']

        # 범주형 변수 'Catalyst'를 원-핫 인코딩
        X = pd.get_dummies(X, columns=['Catalyst'])

        # 사용된 피처 출력
        print("훈련에 사용된 피처들:")
        print(X.columns)

        # 데이터 스케일링 (표준화, LightGBM에서는 필수는 아님)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # LightGBM 회귀 모델 정의
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        # 모델 학습
        lgb_model.fit(X_scaled, y)

        # 학습된 모델과 스케일러 저장
        joblib.dump(lgb_model, '/content/drive/MyDrive/lignin/pkl(LightGBM)/lgb_model.pkl')  # 모델 저장
        joblib.dump(scaler, '/content/drive/MyDrive/lignin/pkl(LightGBM)/scaler.pkl')  # 스케일러 저장

        print("LightGBM 모델과 스케일러를 저장했습니다.")


    # 저장된 LightGBM 모델과 스케일러 불러오기
    lgb_model = joblib.load('/content/drive/MyDrive/lignin/pkl(LightGBM)/lgb_model.pkl')  # LightGBM 모델 파일 경로
    scaler = joblib.load('/content/drive/MyDrive/lignin/pkl(LightGBM)/scaler.pkl')        # 스케일러 파일 경로

    # 예측할 조건 설정 (예: PT_C에서 375℃)
    temperature = 375
    catalyst = 'RU_C'

    # 파장 범위를 650 ~ 4000 cm^-1로 설정, 3477개의 파장 값으로 나눔
    cm_values = np.linspace(650, 4000, 3477)

    # 예측을 위한 데이터 생성
    X_test = pd.DataFrame({'Temperature': [temperature] * len(cm_values), 'cm': cm_values})

    # 원-핫 인코딩 처리 (훈련된 모델과 동일한 피처 구조 유지)
    X_test['Catalyst_' + catalyst] = 1  # PT_C 열만 1로 설정 (원-핫 인코딩)

    # 나머지 Catalyst 열을 0으로 채움
    for other_catalyst in ['Catalyst_NOCAT', 'Catalyst_RN', 'Catalyst_RU_C']:
        X_test[other_catalyst] = 0

    # 훈련 시 사용된 피처의 순서를 정확하게 맞춤
    expected_features = ['Temperature', 'cm', 'Catalyst_NOCAT', 'Catalyst_PT_C', 'Catalyst_RN', 'Catalyst_RU_C']
    X_test = X_test.reindex(columns=expected_features, fill_value=0)

    # 스케일링 적용
    X_test_scaled = scaler.transform(X_test)

    # LightGBM 모델로 예측
    y_pred = lgb_model.predict(X_test_scaled)

    # 예측 결과를 DataFrame으로 저장
    predicted_data = pd.DataFrame({'cm': cm_values, '%': y_pred})

    # 예측 결과를 CSV 파일로 저장
    output_file_path = f'/content/drive/MyDrive/lignin/pkl(LightGBM)/predict/{catalyst}{temperature}_predicted_lgb.csv'
    predicted_data.to_csv(output_file_path, index=False)

    print(f"{temperature}℃에서 {catalyst}의 LightGBM 예측 데이터를 {output_file_path}에 저장했습니다.")