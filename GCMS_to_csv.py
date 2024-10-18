import pandas as pd
import os

def process_and_export_gcms_data(GCMS_data) :
    # GCMS 파일에서 Bold된 글자들을 CSV 파일로 저장
    # dataset/GC-MS_to_xls/*.csv

    mass_values = []

    for value in GCMS_data.split():
        try:
            mass_values.append(float(value))
        except ValueError:
            continue

    formatted_mass_data = []

    for section_index in range(4):
        for column_index in range(4):
            mass_row = []
            for offset in range(0, 33, 4):
                mass_row.append(mass_values[int(((section_index * len(mass_values)) / 4) + column_index + offset)])

            mass_row.append(100 - sum(mass_row))  # 100에서 질량값들의 합을 뺀 값을 추가
            mass_row.append(sum(mass_row))  # 질량값들의 합을 추가

            formatted_mass_data.append(mass_row)

    for row_index in range(len(formatted_mass_data)):
        save_data_to_csv(formatted_mass_data[row_index], f"{row_index + 1}.csv")


def save_data_to_csv(data, file_name):
    """
    데이터 배열과 파일 이름을 받아 CSV 파일로 저장하는 함수 (세로로 저장).

    Args:
    data (list): 값 리스트 (한 열에 저장될 값들).
    file_name (str): 저장할 CSV 파일 이름.
    """
    # 고정된 제목 리스트
    titles = ["Syringyl", "Guaiacyl", "Poly aromatics", "Other aromatics", "Alkanes(Paraffins)",
              "Cyclic", "Fatty Acids", "alcohol", "Glycerol derived", "Other", "Total"]

    # 제목과 데이터를 세로로 변환 (하나의 열에 두 리스트를 쌓음)
    df = pd.DataFrame({
        "Category": titles,
        "Value": data
    })

    save_path = "dataset/GC-MS_to_csv"
    # 경로가 없으면 폴더 생성
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 저장 경로와 파일 이름을 합침
    full_path = os.path.join(save_path, file_name)

    # 파일 저장 (CSV 형식으로)
    df.to_csv(full_path, index=False)
    print(f'{full_path} 저장 완료!')
