import os
import pandas as pd


def combine_csv_files(directory = 'dataset/GC-MS_to_csv/', output_path = 'dataset/combined_GCMS.csv' ):
    """
    주어진 디렉토리 내의 모든 CSV 파일을 하나의 데이터프레임으로 결합하고,
    지정된 경로에 저장하는 함수.

    Args:
    - directory (str): CSV 파일들이 저장된 디렉토리 경로
    - output_path (str): 결합된 CSV 파일을 저장할 경로

    Returns:
    - pd.DataFrame: 모든 CSV 파일을 결합한 데이터프레임
    """
    # 모든 CSV 파일을 담을 리스트
    csv_list = []

    # 디렉토리 내 모든 파일을 확인
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # 각 CSV 파일을 데이터프레임으로 읽어서 리스트에 추가
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            csv_list.append(df)

    # 모든 CSV 파일을 하나의 데이터프레임으로 합치기
    if csv_list:  # 리스트가 비어있지 않으면
        combined_csv = pd.concat(csv_list, ignore_index=True)

        # 합쳐진 데이터프레임을 지정된 경로에 CSV로 저장
        combined_csv.to_csv(output_path, index=False)
        print(f"결합된 CSV 파일이 {output_path}에 저장되었습니다.")


