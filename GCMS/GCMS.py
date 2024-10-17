import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 수정중

predict_CSV = "/content/drive/MyDrive/lignin/TGA/total/1-1.csv"

# 성분 목록 정의 (순서대로 출력)
components = [
    "Syringyl", "Guaiacyl", "Poly aromatics", "Other aromatics",
    "Alkanes(Paraffins)", "Cyclic", "Fatty Acids", "alcohol",
    "Glycerol derived", "Other", "Total"
]

# 사용할 입력 피처 목록 (명시적으로 지정)
input_features = [
    'Second_Row_Temperature', 'Second_Row_Catalyst',
    'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
    '% / C1', '% / C2', '% / C3', '% / C4', '% / C5', '% / C6', '% / C7', '% / C8',
    '% / C9', '% / C10', '% / C11',
    '% / C_abs1', '% / C_abs2', '% / C_abs3', '% / C_abs4', '% / C_abs5',
    '% / C_abs6', '% / C_abs7', '% / C_abs8', '% / C_abs9', '% / C_abs10', '% / C_abs11'
]

# 모델과 인코더 경로 설정
model_filename = "multi_output_regressor.pkl"
encoder_filename = "onehot_encoder.pkl"  # 인코더 저장 파일

# 모델과 인코더 불러오기
loaded_model = joblib.load(model_filename)
encoder = joblib.load(encoder_filename)

print("모델과 인코더가 성공적으로 불러와졌습니다.")

# 예측을 위한 함수 정의
def predict_for_row(row):
    # 온도와 촉매 추출
    temperature = row['Second_Row_Temperature']
    catalyst = row['Second_Row_Catalyst']

    # 촉매 원-핫 인코딩
    catalyst_encoded = encoder.transform([[catalyst]])

    # 나머지 수치형 피처 추출 (온도와 촉매 제외)
    feature_values = row.drop(['Second_Row_Temperature', 'Second_Row_Catalyst']).values.reshape(1, -1)

    # 입력 데이터 준비 (온도 + 원-핫 인코딩된 촉매 + 나머지 수치형 피처)
    input_prepared = np.hstack(([[temperature]], catalyst_encoded, feature_values))

    # 예측 수행
    predicted_values = loaded_model.predict(input_prepared)[0]

    # 예측 결과 출력
    print(f"\n입력된 온도: {temperature}도")
    print(f"입력된 촉매: {catalyst}")
    print("\n예측된 성분 값:")

    for component, value in zip(components, predicted_values):
        print(f"{component}: {value:.4f}")

    return predicted_values

# 예측 결과를 시각화하는 함수 정의
def plot_predicted_values(components, predicted_values, temperature, catalyst):
    plt.figure(figsize=(10, 6))
    plt.barh(components, predicted_values, color='skyblue')
    plt.xlabel('Predicted Value')
    plt.title(f'Predicted GC-MS Component Values\nTemperature: {temperature}°C, Catalyst: {catalyst}')
    plt.gca().invert_yaxis()  # 성분을 위에서 아래로 정렬
    plt.show()

# CSV 파일에서 입력 데이터 읽기
file_path = predict_CSV  # 업로드된 파일 경로
data = pd.read_csv(file_path)

# 필요한 입력 피처만 추출
data = data[input_features]

# CSV 파일의 각 행에 대해 예측 수행 및 시각화
for index, row in data.iterrows():
    predicted_values = predict_for_row(row)
    plot_predicted_values(components, predicted_values, row['Second_Row_Temperature'], row['Second_Row_Catalyst'])


