import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib  # 모델 저장을 위한 라이브러리


##### 수정중 입니다.


# 데이터 경로 정의
data_paths = {
    "1": "dataset/GC-MS_to_csv/1.xls",
    "5": "dataset/GC-MS_to_csv/5.xls",
    "9": "dataset/GC-MS_to_csv/9.xls",
    "13": "dataset/GC-MS_to_csv/13.xls",
}

# 특징 추출 함수
def extract_features(data):
    features = data[["C", "% / C_abs"]]
    diff = features.diff().fillna(0)
    pct_change = features.pct_change().fillna(0)
    moving_avg = features.rolling(window=3).mean().fillna(0)
    cum_sum = features.cumsum()
    return pd.concat([features, diff, pct_change, moving_avg, cum_sum], axis=1)

# 모든 파일에서 특징을 추출하고 메타데이터(온도, 촉매)를 추가합니다.
all_features = []
for key, path in data_paths.items():
    data = pd.read_csv(path)
    temperature = data["Second_Row_Temperature"].iloc[0]
    catalyst = data["Second_Row_Catalyst"].iloc[0]
    features = extract_features(data)
    features["Temperature"] = temperature
    features["Catalyst"] = catalyst
    all_features.append(features)

# 모든 특징 데이터 결합
combined_features = pd.concat(all_features, ignore_index=True)
combined_features["Catalyst"] = combined_features["Catalyst"].astype('category').cat.codes
# 온도와 촉매를 One-Hot 인코딩
combined_features = pd.get_dummies(combined_features, columns=["Temperature", "Catalyst"])

# 타겟 값 준비
target_values = {
    "1-1": [14.37, 4.01, 0.91, 20.06, 1.99, 4.78, 10.03, 0.44, 34.58, 8.83, 100.0],
    "5-1": [25.71, 17.28, 1.17, 23.22, 2.16, 2.38, 10.85, 0.4, 8.07, 8.76, 100.0],
    "9-1": [25.78, 24.75, 2.59, 21.71, 1.08, 2.16, 2.7, 0.28, 5.06, 13.89, 100.0],
    "13-1": [1.98, 4.11, 6.02, 64.22, 0.58, 5.38, 4.67, 0.65, 0.46, 11.93, 100.0],
}
Y = (
    [target_values["1-1"]] * len(all_features[0])
    + [target_values["5-1"]] * len(all_features[1])
    + [target_values["9-1"]] * len(all_features[2])
    + [target_values["13-1"]] * len(all_features[3])
)
Y = np.array(Y)

# 데이터 정규화 및 분리
scaler = StandardScaler()
X = combined_features.drop(columns=["Temperature", "Catalyst"])
X_scaled = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# XGBoost 모델 학습
xgb_model = MultiOutputRegressor(
    XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42, verbosity=0)
)
xgb_model.fit(X_train, Y_train)

# 모델 저장
model_path = "/content/drive/MyDrive/lignin/TGA/total/xgboost_model.joblib"
joblib.dump(xgb_model, model_path)
print(f"모델이 저장되었습니다: {model_path}")

# 저장된 모델 불러오기 (테스트)
loaded_model = joblib.load(model_path)

# 예측 및 MSE 계산
Y_pred_train = loaded_model.predict(X_train)
mse_train = mean_squared_error(Y_train, Y_pred_train)
print(f"\n학습 데이터의 MSE: {mse_train}")

# 예측 결과를 데이터프레임에 추가하여 출력
predicted_df = pd.DataFrame(Y_pred_train, columns=[f'Target_{i+1}' for i in range(Y_train.shape[1])])
full_df = pd.concat([combined_features.reset_index(drop=True), predicted_df], axis=1)
print("\n[전체 학습 데이터와 예측 결과]")
print(full_df.head(10))

# 피처 중요도 추출 및 시각화
importances = loaded_model.estimators_[0].feature_importances_
feature_names = X.columns
plt.figure(figsize=(12, 6))
plt.barh(feature_names, importances, align='center')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance from XGBoost Model')
plt.show()

