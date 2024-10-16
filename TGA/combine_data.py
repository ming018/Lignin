import os
import pandas as pd


"""
diriectory_path내에 있는 csv파일들을 하나의 csv파일로 만들어
output_file의 위치에 저장
"""



# CSV 파일이 있는 경로 설정
directory_path = '../dataset/after_csv/'

# 특정 경로에 결합된 csv파일을 저장
output_file = '../dataset/combined_file.csv'

# 결과를 저장할 DataFrame 초기화
combined_data = pd.DataFrame()

# 파일명에 따른 첫 번째 값 추가 함수 ('NoCat', 'PtC', 'RuC', 'RN')
def get_first_value_from_filename(filename):
    # 확장자 제거 및 predict_로 시작하는 파일 처리
    base_filename = filename.split('.')[0]
    if base_filename.startswith('predict'):
        if 'No' in base_filename:
            return 'NoCat'
        elif 'PtC' in base_filename:
            return 'PtC'
        elif 'RuC' in base_filename:
            return 'RuC'
        elif 'RN' in base_filename:
            return 'RN'
        else:
            return 'unknown'
    else:
        # 기존 숫자 기반 파일명 처리 (1, 5, 9, 13 -> NoCat / 2, 6, 10, 14 -> PtC 등)
        file_number = int(base_filename)
        if file_number in [1, 5, 9, 13]:
            return 'NoCat'
        elif file_number in [2, 6, 10, 14]:
            return 'PtC'
        elif file_number in [3, 7, 11, 15]:
            return 'RuC'
        elif file_number in [4, 8, 12, 16]:
            return 'RN'
        else:
            return 'unknown'

# 파일명에 따른 두 번째 값 추가 함수 (온도 값 추출)
def get_second_value_from_filename(filename):
    # 확장자 제거 및 predict_로 시작하는 파일 처리
    base_filename = filename.split('.')[0]
    if base_filename.startswith('predict'):
        temp_value = ''.join(filter(str.isdigit, base_filename))
        if temp_value:
            return int(temp_value)
        else:
            return 'unknown'
    else:
        # 기존 숫자 기반 파일명 처리 (1-4 -> 250, 5-8 -> 300 등)
        file_number = int(base_filename)
        if file_number in [1, 2, 3, 4]:
            return 250
        elif file_number in [5, 6, 7, 8]:
            return 300
        elif file_number in [9, 10, 11, 12]:
            return 350
        elif file_number in [13, 14, 15, 16]:
            return 400
        else:
            return 'unknown'

# 디렉토리 내 모든 CSV 파일 처리
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory_path, filename)

        # CSV 파일 읽기
        df = pd.read_csv(file_path)

        # 파일명에 따라 첫 번째 값과 두 번째 값 추가
        df['Catalyst'] = get_first_value_from_filename(filename)
        df['temp'] = get_second_value_from_filename(filename)

        # 기존 데이터에 추가
        combined_data = pd.concat([combined_data, df], ignore_index=True)

# 하나의 파일로 저장

combined_data.to_csv(output_file, index=False)

print(f'모든 CSV 파일이 결합되어 {output_file}에 저장되었습니다.')
