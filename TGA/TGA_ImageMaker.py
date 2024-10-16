import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기 (파일 경로를 지정)
file_path = '/content/drive/MyDrive/lignin/TGA/pkl(AdaBoost)/predict/_270C_NOCAT.csv'

# CSV 파일 읽기
df = pd.read_csv(file_path)

# 온도와 무게 관련 데이터 추출
temperature = df['C']  # 온도 열
#deriv_weight = df['% / C']  # 미분값 열
weight = df['%']  # Weight (%) 열

# 미분값 절대값 처리
#deriv_weight_abs = np.abs(deriv_weight)

# 그래프 그리기
fig, ax1 = plt.subplots()

# 첫 번째 Y축 설정 (Weight %)
ax1.set_xlim(40, 800)
ax1.set_ylim(0, 120)
ax1.plot(temperature, weight, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# X축 및 첫 번째 Y축 눈금 제거
ax1.set_yticks([])
ax1.set_xticks([])

# 두 번째 Y축 설정 (Deriv. Weight)
ax2 = ax1.twinx()
ax2.set_ylim(0, 1)

# Y축 및 X축 눈금 제거
ax2.set_yticks([])
ax2.set_xticks([])


# 그래프를 화면에 표시
plt.show()