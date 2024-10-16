import os
import pandas as pd


# folder_path의 위치에 있는
# csv파일의 파일명에 따라 특정 단어를 추가할 함수
# 전처리 온도, 촉매를 추가


# CSV 파일들이 위치한 경로 설정
folder_path = '../dataset/after_csv/'


def add_word_based_on_filename(filename, df):
    if df is None:
        return None  # df가 None인 경우 그대로 반환

    base_filename = filename.split('.')[0]

    # Catalyst 이름을 우선적으로 확인
    if 'No' in filename:
        df['Catalyst'] = 'NoCat'
    elif 'PtC' in filename:
        df['Catalyst'] = 'PtC'
    elif 'RN' in filename:
        df['Catalyst'] = 'RN'
    elif 'RuC' in filename:
        df['Catalyst'] = 'RuC'
    else:
        df['Catalyst'] = 'Unknown_Catalyst'

    if base_filename.isdigit() :
        file_number = int(base_filename)

        if file_number in [1, 5, 9, 13]:
            df['Catalyst'] = 'NoCat'
        elif file_number in [2, 6, 10, 14]:
            df['Catalyst'] = 'PtC'
        elif file_number in [3, 7, 11, 15]:
            df['Catalyst'] = 'RuC'
        elif file_number in [4, 8, 12, 16]:
            df['Catalyst'] = 'RN'
        else:
            df['Catalyst'] = 'Unknown_Catalyst'

    return df


# 파일명에서 두 번째 값(숫자)을 가져와 추가하는 함수
def get_second_value_from_filename(filename, df):
    if df is None:
        return None  # df가 None인 경우 그대로 반환

    # 확장자 제거 및 predict_로 시작하는 파일 처리
    base_filename = filename.split('.')[0]
    if base_filename.startswith('predict'):
        temp_value = ''.join(filter(str.isdigit, base_filename))
        if temp_value:
            df['temp'] = int(temp_value)
        else:
            df['temp'] = 'unknown'
    else:
        # 기존 숫자 기반 파일명 처리 (1-4 -> 250, 5-8 -> 300 등)
        try:
            file_number = int(base_filename)
            if file_number in [1, 2, 3, 4]:
                df['temp'] = 250
            elif file_number in [5, 6, 7, 8]:
                df['temp'] = 300
            elif file_number in [9, 10, 11, 12]:
                df['temp'] = 350
            elif file_number in [13, 14, 15, 16]:
                df['temp'] = 400
            else:
                df['temp'] = 'unknown'
        except ValueError:
            df['temp'] = 'unknown'

    return df


# 폴더 내 모든 CSV 파일을 처리
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        # 파일명에 따라 Catalyst 값 추가
        df = add_word_based_on_filename(filename, df)
        if df is None:  # df가 None인 경우 다음 파일로 넘어감
            print(f"Skipping file {filename} due to invalid DataFrame.")
            continue

        # 두 번째 값(숫자)을 파일명에서 추출하여 temp 열 추가
        df = get_second_value_from_filename(filename, df)
        if df is None:  # df가 None인 경우 다음 파일로 넘어감
            print(f"Skipping file {filename} due to invalid DataFrame.")
            continue

        # 변경된 파일을 저장 (덮어쓰기)
        df.to_csv(file_path, index=False)

print("CSV 파일 처리 완료!")
