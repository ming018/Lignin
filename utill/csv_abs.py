import os
import pandas as pd

# 원본 xls 파일들이 있는 폴더 경로 설정
input_directory = ('/Users/parkmingyu/Desktop/24_09_02/dataset/TGA_')  # 원본 파일이 있는 디렉토리 경로
# 변환된 csv 파일을 저장할 폴더 경로 설정
output_directory = '../dataset/after_csv/'  # 변환된 파일을 저장할 디렉토리 경로

# 출력 디렉토리가 없으면 생성
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 폴더 내 모든 파일에 대해 반복 작업 수행
for filename in os.listdir(input_directory):
    if filename.endswith('.xls'):  # 확장자가 .xls인 파일만 처리
        file_path = os.path.join(input_directory, filename)

        # xls 파일 읽기
        df = pd.read_excel(file_path)

        # Deriv. Weight 열의 값을 절대값으로 변환 (해당 열이 존재하는 경우)
        if 'Deriv. Weight' in df.columns:
            df['Deriv. Weight'] = df['Deriv. Weight']

        # 변환된 파일을 지정한 출력 경로에 저장 (.csv 확장자로 저장)
        output_file = os.path.join(output_directory, filename.replace('.xls', '.csv'))  # .xls를 .csv로 변경하여 저장
        df.to_csv(output_file, index=False)

        print(f"{filename} 파일이 성공적으로 처리되어 {output_file}로 저장되었습니다.")
