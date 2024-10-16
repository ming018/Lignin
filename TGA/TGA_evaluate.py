import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from fastdtw import fastdtw

# 코사인 유사도를 계산하는 함수 (리스트 입력)
def calculate_cosine_similarity(list1, list2):

    minimize = min(len(list1), len(list2))

    vec1 = np.array(list1[:minimize])
    vec2 = np.array(list2[:minimize])
    return 1 - cosine(vec1, vec2)

# 피어슨 상관계수를 계산하는 함수 (리스트 입력)
def calculate_pearson_correlation(list1, list2):

    minimize = min(len(list1), len(list2))

    vec1 = np.array(list1[:minimize])
    vec2 = np.array(list2[:minimize])
    correlation, _ = pearsonr(vec1, vec2)
    return correlation

def cal_corrcoef(data1, data2, data3):
    corr_data1_data3 = calculate_pearson_correlation(data1, data3)
    corr_data2_data3 = calculate_pearson_correlation(data2, data3)

    return corr_data1_data3, corr_data2_data3

def dtw_distance(data1, data2, data3):
    distance3_1, _ = fastdtw(list(data1), data3)
    distance3_2, _ = fastdtw(list(data2), data3)
    return distance3_1, distance3_2

def TGA_evaluate(target_temp, catalyst, data1, data2, predict_data, start_temp, limit_temp) :

    start_idx = (target_temp // 50) - 5
    limit_idx = (target_temp // 50) - 4

    # 코사인 유사도 계산
    cosine_similarity_A_B = calculate_cosine_similarity(predict_data, data1)
    cosine_similarity_A_C = calculate_cosine_similarity(predict_data, data2)

    # 피어슨 상관계수 계산
    pearson_corr_A_B = calculate_pearson_correlation(predict_data, data1)
    pearson_corr_A_C = calculate_pearson_correlation(predict_data, data2)

    # DTW 거리 계산
    distanceC_A, distanceC_B = dtw_distance(data1, data2, predict_data)

    # TGA_compare_interpolation.compare_main()

    print(f"{catalyst}촉매의 {target_temp}C 유사도 계산")
    print(f"{start_temp}와 예측 데이터의 코사인 유사도 : {cosine_similarity_A_B}")
    print(f"{limit_temp}와 예측 데이터의 코사인 유사도 : {cosine_similarity_A_C}\n")

    print(f"{start_temp}와 예측 데이터의 피어슨 상관계수 : {pearson_corr_A_B}")
    print(f"{limit_temp}와 예측 데이터의 피어슨 상관계수 : {pearson_corr_A_C}\n")

    print(f"{start_temp}와 예측 데이터의 DTW거리 : {distanceC_A}")
    print(f"{limit_temp}와 예측 데이터의 DTW거리 : {distanceC_B}\n")
