import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from scipy.interpolate import interp1d

# 0 조건과 1 조건의 데이터
data_0 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # 0 조건 데이터
data_1 = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])  # 1 조건 데이터

# 보간할 지점 (0.5 조건)
x = np.array([0, 1])  # 조건의 위치 (0과 1)
x_interp = 0.5  # 보간 지점

# 각 항목별로 보간 수행
y_interp = []
for i in range(len(data_0)):
    # 보간 함수 생성 (선형 보간)
    f = interp1d(x, [data_0[i], data_1[i]], kind='linear')
    y_interp.append(f(x_interp))

y_interp = np.array(y_interp)

# 유사도 측정: 유클리드 거리
dist_0 = euclidean(data_0, y_interp)
dist_1 = euclidean(data_1, y_interp)

# 유사도 측정: 코사인 유사도
cos_sim_0 = cosine_similarity([data_0], [y_interp])[0][0]
cos_sim_1 = cosine_similarity([data_1], [y_interp])[0][0]

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
