import os
import pandas as pd


def process_csv_files_in_directory(directory_path):
    # 각 그룹의 숫자 리스트
    group1 = [1, 5, 9, 13]
    group2 = [2, 6, 10, 14]
    group3 = [3, 7, 11, 15]
    group4 = [4, 8, 12, 16]

    # 디렉터리 안의 모든 파일을 확인
    for filename in os.listdir(directory_path):
        # CSV 파일만 처리
        if filename.endswith(".csv"):
            # 파일 이름에서 숫자를 추출 (숫자.csv의 형태)

            # 파일 이름에서 .csv를 제거하고 숫자 부분 추출
            number = int(filename.split('.')[0])

            # CSV 파일 읽기
            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # 'Catalyst'와 'temp' 컬럼이 이미 있는지 확인
            # if 'Catalyst' in df.columns and 'temp' in df.columns:
            #     continue

            # GC-MS.csv 파일명에 따라서 촉매와 전처리 온도를 추가
            if number in group1:
                df['Catalyst'] = 'NoCat'
                df['temp'] = 250 + 50 * group1.index(number)

            elif number in group2:
                df['Catalyst'] = 'PtC'
                df['temp'] = 250 + 50 * group2.index(number)

            elif number in group3:
                df['Catalyst'] = 'RuC'
                df['temp'] = 250 + 50 * group3.index(number)


            elif number in group4:
                df['Catalyst'] = 'RN'
                df['temp'] = 250 + 50 * group4.index(number)


            else :
                print(f"* {filename} 파일은 적용이 되지 않았습니다.")

            # CSV 파일 덮어쓰기
            df.to_csv(file_path, index=False)
            print(f"{filename} 파일이 업데이트되었습니다.")

