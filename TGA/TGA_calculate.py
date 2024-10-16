import numpy as np

# 온도를 이용해서 %계산, ex) 260은 250과 300 사이에서 20%증가한 값이기 때문에,
def calculate_percent(start_temp, end_temp, target_temp) :
    percentage = (abs(target_temp - start_temp) / abs(end_temp - start_temp)) * 100
    return percentage


# 아래 두 함수는 pearson 상관계수
def calculate_pearson_correlation(x, y):

    minimize = min(len(x), len(y))
    x = x[:minimize - 1]
    y = y[:minimize - 1]

    # 평균 계산
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    # 분자 계산: 공분산
    numerator = np.sum((x - mean_x) * (y - mean_y))

    # 분모 계산: 각 데이터 세트의 표준 편차의 곱
    std_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2))

    denominator = std_x * std_y

    # 상관 계수 계산
    if denominator == 0:
        return np.nan  # 표준 편차가 0이면 상관 계수를 계산할 수 없음
    return numerator / denominator


def cal_corrcoef(data1, data2, data3):
    corr_data1_data3 = calculate_pearson_correlation(data1, data3)
    corr_data2_data3 = calculate_pearson_correlation(data2, data3)

    return corr_data1_data3, corr_data2_data3

def linear_interpolation(A_list, B_list, t_values):
    A_array = np.array(A_list)
    B_array = np.array(B_list)
    return (A_array + (B_array - A_array) * t_values)
