import csv
import pandas as pd

# 시간 온도 무게% 무게변화량 파일이름
def save_custom_format_csv(data1, data2, data3, data4, filename):
    """
    data1, data2, data3을 받아 고정된 경로와 헤더로 CSV 파일을 저장하는 함수.

    Parameters:
    data1 (list): 첫 번째 데이터 리스트 (예: Temperature)
    data2 (list): 두 번째 데이터 리스트 (예: Deriv. Weight)
    data3 (list): 세 번째 데이터 리스트 (예: 추가 데이터)
    filename (str): 저장할 파일 이름 (확장자 없이 전달)

    예시:
    save_custom_format_csv([1, 2, 3], [0.1, 0.2, 0.3], [10, 20, 30], 'example_file')
    """
    # 고정된 저장 경로 설정
    save_path = 'dataset/train/Interpolated_TGA/'

    # 고정된 헤더 설정
    header = ['Time', 'Temperature', 'Weight', 'Deriv. Weight',]

    # data1, data2, data3을 합쳐서 2차원 리스트로 변환
    data_list = list(zip(data1, data2, data3, data4))

    # 파일명에 경로와 확장자 추가
    file_with_extension = f"{save_path}{filename}.csv"

    # 파일 저장
    with open(file_with_extension, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 헤더 저장
        writer.writerow(header)
        # 리스트 데이터를 행 단위로 저장
        writer.writerows(data_list)

    print(f"파일 '{file_with_extension}'이(가) 저장되었습니다.")
