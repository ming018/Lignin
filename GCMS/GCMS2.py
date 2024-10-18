import os
import pandas as pd

##### 수정중 입니다.

def process_excel_files(directory_path, specific_numbers, value_to_add, row, col):
    """
    주어진 디렉토리 내에서 파일 제목이 숫자인 .xls 파일을 검색하고,
    특정 숫자 리스트에 포함된 파일에 대해 지정된 위치에 값을 추가하는 함수.

    Parameters:
    directory_path (str): 엑셀 파일들이 저장된 디렉토리 경로
    specific_numbers (list): 파일 이름이 해당 숫자들에 부합하는 경우 처리할 숫자 리스트
    value_to_add (str): 추가할 값
    row (int): 값을 추가할 행 번호 (0부터 시작)
    col (int): 값을 추가할 열 번호 (0부터 시작)
    """

    # 특정 위치에 값을 추가하는 내부 함수
    def add_values_to_excel(file_path, value1, row, col):
        # Excel 파일을 불러옴 (첫 번째 시트를 기본으로 사용)
        df = pd.read_excel(file_path)

        # 행 또는 열이 부족한 경우 확장
        max_row, max_col = df.shape

        if row >= max_row:
            # 부족한 행을 채워넣음
            for _ in range(row + 1 - max_row):
                df.loc[len(df)] = [None] * max_col

        if col >= max_col:
            # 부족한 열을 채워넣음
            df = pd.concat([df, pd.DataFrame([[None] * (col + 1 - max_col)], columns=range(max_col, col + 1))], axis=1)

        # 특정 위치에 값 추가 (index는 0부터 시작)
        df.iloc[row, col] = value1

        # 변경된 DataFrame을 다시 파일로 저장
        df.to_excel(file_path, index=False)

    # 디렉토리 내의 모든 파일들에 대해 처리
    for filename in os.listdir(directory_path):
        # 파일 이름이 숫자이고 .xls로 끝나는지 확인
        if filename.endswith(".xls") and filename[:-4].isdigit():
            # 파일 이름에서 숫자 부분만 추출
            file_number = int(filename.split('.')[0])

            # 파일 번호가 특정 숫자 리스트에 있는지 확인
            if file_number in specific_numbers:
                file_path = os.path.join(directory_path, filename)

                # 값을 엑셀 파일에 추가
                add_values_to_excel(file_path, value_to_add, row, col)

    print("작업이 완료되었습니다.")


# 함수 호출 예시
directory_path = "../dataset/GC-MS_to_csv/"  # 파일이 있는 디렉토리 경로
specific_numbers = [1]  # 처리할 숫자 리스트
value_to_add = "특정값"  # 추가할 값
row = 2  # 추가할 행 번호 (예: 3번째 줄에 추가)
col = 5  # 추가할 열 번호 (예: 6번째 열)

# 함수 실행
process_excel_files(directory_path, specific_numbers, value_to_add, row, col)
