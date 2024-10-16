import pandas as pd
import os

def save_data_to_excel(data, file_name):
    """
    데이터 배열과 파일 이름을 받아 엑셀 파일로 저장하는 함수.

    Args:
    data (list): 값 리스트 (한 행에 저장될 값들).
    file_name (str): 저장할 엑셀 파일 이름.
    """
    # 고정된 제목 리스트
    titles = ["Syringyl", "Guaiacyl", "Poly aromatics", "Other aromatics", "Alkanes(Paraffins)", "Cyclic", "Fatty Acids", "alcohol", "Glycerol derived", "Other", "Total"]

    # 제목과 데이터를 한 행에 저장
    df = pd.DataFrame([titles, data])

    save_path = "dataset/GC-MS_to_xls"
    # 경로가 없으면 폴더 생성
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 저장 경로와 파일 이름을 합침
    full_path = os.path.join(save_path, file_name)

    # 파일 저장
    df.to_excel(full_path, index=False, header=False, engine='openpyxl')
    print(f'{full_path} 저장 완료!')


