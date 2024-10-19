import os
import pandas as pd
import PyPDF2
import re
import os
import numpy as np

ROOT_DIR = 'dataset/'

def extract_numbers(file_name):
    """
    파일 이름에서 숫자를 추출하는 함수.
    예: '10-1.csv' -> [10, 1]
    """
    return list(map(int, re.findall(r'\d+', file_name)))


def save_data_as_npy(data, npy_file_path):
    # 데이터를 npy 파일로 저장
    np.save(npy_file_path, data)

def load_data_from_npy(npy_file_path):
    # npy 파일로부터 데이터를 로드
    return np.load(npy_file_path, allow_pickle=True)

def load_TGA(directory_path, cut_off):
    """
    TGA 데이터를 로드하여 각 파일을 npy 파일로 저장하거나 npy 파일이 있을 경우 이를 로드합니다.

    :param directory_path: XLS 및 XLSX 파일들이 있는 디렉토리 경로
    :param cut_off: 컷오프 값을 기준으로 파일 필터링
    :return: 각 파일의 데이터를 포함하는 리스트
    """
    # 디렉토리 내의 모든 XLS 및 XLSX 파일 목록 가져오기
    xls_files = [f for f in os.listdir(directory_path) if f.endswith('.xls') or f.endswith('.xlsx')]

    # 컷오프 값을 기준으로 파일들 필터링
    valid_files = [f for f in xls_files if int(f.split('.')[0]) < cut_off]

    valid_files.sort(key=extract_numbers)

    # 각 파일의 데이터를 저장할 리스트
    data = []

    # 각 파일을 처리하고 npy 파일로 저장/로드
    for file in valid_files:
        file_name_without_ext = os.path.splitext(file)[0]  # 확장자 없는 파일명
        npy_file_path = os.path.join(directory_path, f"{file_name_without_ext}.npy")  # npy 파일 경로

        if os.path.exists(npy_file_path):
            # npy 파일이 존재하면 이를 로드하여 추가
            print(f"Loading data from {npy_file_path}")
            file_data = load_data_from_npy(npy_file_path)
        else:
            # npy 파일이 없으면 엑셀 파일을 읽고 저장
            file_path = os.path.join(directory_path, file)
            df = pd.read_excel(file_path, sheet_name=1, skiprows=3, header=None)  # 두 번째 시트 읽기, 첫 3행 건너뛰기
            # df에서 2, 3, 4 번째 열만 추출
            df_filtered = df[[1, 3, 4]]
            df_filtered = df_filtered[(df_filtered[1] >= 40) & (df_filtered[1] < 800)]

            file_data = df_filtered
            print(f"Saving data to {npy_file_path}")
            save_data_as_npy(file_data, npy_file_path)

        # 데이터를 리스트에 추가
        data.append(file_data.T)

    # 파일명을 기준으로 데이터를 정렬 (숫자 부분을 기준으로)
    # data.sort(key=lambda x: int(x[0].split('.')[0]))

    return data

def load_FTIR(directory_path, cut_off, npy_path=None):
    """
    지정된 디렉토리 내의 모든 CSV 파일에서 20번째 행부터 숫자가 있는 동안의 데이터를 추출하여 npy로 저장하거나, npy 파일이 있을 경우 이를 로드합니다.

    :param cut_off: 파일 이름의 숫자 부분이 이 값보다 작은 파일들만 처리
    :param directory_path: CSV 파일들이 있는 디렉토리 경로
    :param npy_path: 저장된 npy 파일 경로 (없을 경우 새로 로드하고 저장)
    :return: 각 파일의 20번째 행부터 숫자가 있는 동안의 데이터를 포함하는 리스트
    """
    # npy 파일이 있다면 로드
    if npy_path and os.path.exists(npy_path):
        print(f"Loading data from {npy_path}")
        return load_data_from_npy(npy_path)

    # 디렉토리 내의 모든 CSV 파일 목록 가져오기
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    # cut_off 값보다 작은 파일들만 처리
    valid_files = [f for f in csv_files if extract_numbers(f)[0] < cut_off]

    # 파일 이름의 숫자 부분을 기준으로 정렬
    valid_files.sort(key=extract_numbers)

    # 각 파일의 20번째 행부터 숫자가 있는 동안의 데이터를 저장할 리스트
    data = []

    # 각 CSV 파일을 처리하여 데이터를 추출
    # 각 CSV 파일을 처리하여 데이터를 추출
    for file in valid_files:
        file_name_without_ext = os.path.splitext(file)[0]  # 확장자 없는 파일명
        npy_file_path = os.path.join(directory_path, f"{file_name_without_ext}.npy")  # npy 파일 경로

        # npy 파일이 존재하면 이를 로드하여 추가
        if os.path.exists(npy_file_path):
            print(f"Loading data from {npy_file_path}")
            file_data = load_data_from_npy(npy_file_path)
        else:
            # npy 파일이 없으면 CSV 파일을 읽어서 저장
            file_path = os.path.join(directory_path, file)
            df = pd.read_csv(file_path, header=None)  # header=None으로 첫 행을 헤더로 사용하지 않음

            if len(df) >= 20:
                # 20번째 행부터 유효한 숫자가 있는 동안의 데이터를 가져오기
                data_from_20th_row = df.iloc[19:]
                valid_data = data_from_20th_row[pd.to_numeric(data_from_20th_row[0], errors='coerce').notnull() &
                                                pd.to_numeric(data_from_20th_row[1], errors='coerce').notnull()]
                file_data = valid_data
            else:
                file_data = pd.DataFrame()  # 파일에 20번째 행이 없는 경우 빈 DataFrame으로 표시

            # npy 파일로 저장
            print(f"Saving data to {npy_file_path}")
            save_data_as_npy(file_data,npy_file_path)

        # 데이터를 리스트에 추가
        data.append(file_data.T.astype(np.float32))

    return data


def read_condition_file(file_path):
    """
    지정된 XLSX 파일에서 두 번째 행부터 마지막 행까지의 데이터를 추출합니다.

    :param file_path: XLSX 파일 경로
    :return: 두 번째 행부터 마지막 행까지의 데이터를 포함하는 DataFrame
    """
    # 파일을 읽어서 첫 행을 건너뛰고 두 번째 행부터 데이터를 가져오기
    df = pd.read_excel(file_path)  # 첫 행 건너뛰기
    return df


def load_GCMS(file_path, start_page=6):
    # PDF 파일 열기
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ""
        # 지정된 페이지 이후의 모든 페이지의 텍스트 추출
        for page_num in range(start_page, reader.numPages):
            page = reader.getPage(page_num)
            text += page.extract_text()

    # 텍스트를 줄 단위로 나누기
    list_txt = text.split('\n')
    txt_dict = {}

    # 정규 표현식 패턴
    pattern = re.compile(r'(Syringyl|Guaiacyl|Poly aromatics|Other aromatics|Alkanes\s*\(Paraffins\)|Cyclic|Fatty Acids|alcohol|Glycerol derived)((?:\s*-?\d+(?:\.\d+)?){5})')

    # 특정 패턴을 찾기
    for line in list_txt:
        match = pattern.match(line)
        if match:
            key = match.group(1) + match.group(2)
            txt_dict[key] = match.group(0)  # 전체 매칭된 텍스트를 저장

    # 딕셔너리의 값을 한 줄씩 추가
    txt = "\n".join(txt_dict.values())

    return txt

def load_data(root_dir, data_type=None):
    # 공통적으로 사용하는 condition_data는 항상 로드
    condition_data = read_condition_file(os.path.join(root_dir, "train.xlsx"))

    # 선택한 data_type에 따라 필요한 데이터만 로드
    if data_type == 'TGA':
        TGA_data = load_TGA(os.path.join(root_dir, "TGA"), cut_off=condition_data['number'].max() + 1)
        return condition_data, TGA_data
    elif data_type == 'FTIR':
        FTIR_data = load_FTIR(os.path.join(root_dir, "FT-IR"), cut_off=condition_data['number'].max() + 1)
        return condition_data, FTIR_data
    elif data_type == 'GCMS':
        GCMS_data = load_GCMS(os.path.join(root_dir, "GC-MS.pdf"))
        return condition_data, GCMS_data
    else:
        # 모든 데이터를 반환해야 할 경우에만 필요한 데이터 로드
        TGA_data = load_TGA(os.path.join(root_dir, "TGA"), cut_off=condition_data['number'].max() + 1)
        FTIR_data = load_FTIR(os.path.join(root_dir, "FT-IR"), cut_off=condition_data['number'].max() + 1)
        GCMS_data = load_GCMS(os.path.join(root_dir, "GC-MS.pdf"))
        return condition_data, TGA_data, FTIR_data, GCMS_data


if __name__ == '__main__' :
    load_data(ROOT_DIR, data_type='TGA')