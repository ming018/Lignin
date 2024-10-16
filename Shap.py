import xgboost as xgb
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import shap  # SHAP library for feature importance

def SHAP_main(X, y):
    # 1. 데이터 리스트를 numpy 배열로 변환
    X = np.array(X)
    y = np.array(y)

    # 2. 만약 X가 1차원 배열이라면 2차원 배열로 변환
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # 3. 훈련/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. XGBoost 회귀 모델 학습 (XGBRegressor)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # 5. 예측 및 평가 (R2 스코어 계산)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2:.2f}")

    # 6. SHAP 값 계산 (TreeExplainer 사용)
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test)

    # 7. SHAP Summary Plot을 통한 Feature Importance 시각화
    shap.summary_plot(shap_values, X_test)

    # 8. SHAP Force Plot을 통해 개별 예측에 대한 기여도 시각화 (첫 번째 데이터 포인트 예시)
    shap.initjs()  # Jupyter notebook 환경에서 초기화
    shap.force_plot(explainer.expected_value, shap_values[0], X_test[0])

    # 9. 피처 중요도 계산 및 출력
    feature_importance = np.mean(np.abs(shap_values), axis=0)

    # X가 DataFrame인지 확인 후 처리
    if isinstance(X, pd.DataFrame):
        features = X.columns
    else:
        features = [f"Feature {i}" for i in range(X.shape[1])]

    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Feature Importance 출력
    print(importance_df)


def shapshap(df_A, df_B, df_C) :
    # Check the column names
    # print(df_C.columns)
    # print(df_A.columns)
    # Create a DataFrame with A, B, and C columns
    df = pd.DataFrame({
        'A': list(df_A),
        'B': list(df_B),
        'C': df_C,
    })

    # Split the data into features (X) and target (y)
    X = df[['A', 'B']]  # Input features (A and B)
    y = df['C']  # Target variable (C)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestRegressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate feature importances using the trained model
    feature_importances = model.feature_importances_

    # Visualize the feature importance
    plt.figure(figsize=(8, 6))
    plt.bar(X.columns, feature_importances, color='skyblue')
    plt.title('Feature Importance for A and B with respect to C shaply value ')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()

    # SHAP analysis for more detailed feature importance
    explainer = shap.TreeExplainer(model)  # Create a SHAP explainer for the RandomForest model
    shap_values = explainer.shap_values(X_test)  # Calculate SHAP values for the test set

    # SHAP summary plot
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)

    # SHAP dependence plot (optional: specific to each feature)
    shap.dependence_plot('A', shap_values, X_test, feature_names=X.columns)

    # Evaluate model performance on the test set
    y_pred = model.predict(X_test)
    mse = np.mean((y_test - y_pred) ** 2)
    print(f'Mean Squared Error (MSE) on Test Set: {mse:.4f}')


# 예시 사용: 데이터 입력
# X, y = your_dataset_here
# SHAP_main(X, y)


# 예시: 샘플 데이터셋을 사용해 실행할 수 있습니다.
# X, y = some_dataset
# SHAP_main(X, y)
