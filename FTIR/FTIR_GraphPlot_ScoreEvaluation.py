import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from fastdtw import fastdtw

def do() :
    # CSV 파일 경로 설정 (5개 파일의 경로를 리스트에 저장)
    file_paths = [
        r'/content/drive/MyDrive/lignin/FT-IR/pkl(AdaBoost)/predict/NOCAT270_predicted_ada.csv',
        r'/content/drive/MyDrive/lignin/FT-IR/pkl(AdaBoost)/predict/NOCAT275_predicted_ada.csv',
    ]

    # 데이터와 파일 이름을 저장할 리스트 초기화
    data_list = []
    file_names = []

    # 각 파일에 대해 데이터 읽기 및 리스트에 저장
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        data_list.append(data)
        file_name = os.path.basename(file_path)
        file_names.append(file_name)

    # 그래프 그리기
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Wavenumber [cm⁻¹]')
    # x축(파수)을 내림차순으로 변경 (FT-IR 특성상)
    ax1.set_xlim(4000, 650)  # x축을 내림차순으로 지정
    ax1.set_ylabel('% Transmittance')
    ax1.set_ylim(40, 110)

    # 색상 리스트 (6개의 색상 지정)
    colors = ['blue', 'red', 'green', 'magenta', 'cyan', 'orange']


    # DTW 계산 함수
    def calculate_dtw(cm_a, per_a, cm_b, per_b, start_wavenumber, end_wavenumber):
        idx_a = (cm_a >= start_wavenumber) & (cm_a <= end_wavenumber)
        idx_b = (cm_b >= start_wavenumber) & (cm_b <= end_wavenumber)

        cm_a_range = cm_a[idx_a]
        per_a_range = per_a[idx_a]

        cm_b_range = cm_b[idx_b]
        per_b_range = per_b[idx_b]

        if len(cm_a_range) == 0 or len(cm_b_range) == 0:
            print(f"No data in the range {start_wavenumber}-{end_wavenumber} cm⁻¹ for comparison.")
            return None

        cm_common = np.linspace(start_wavenumber, end_wavenumber, num=100)

        per_a_interp = np.interp(cm_common, cm_a_range, per_a_range)
        per_b_interp = np.interp(cm_common, cm_b_range, per_b_range)

        distance, path = fastdtw(per_a_interp, per_b_interp, dist=lambda x, y: abs(x - y))

        return distance


    # 각 데이터에 대해 그래프에 플롯 (굵기 설정: linewidth=2)
    for i, data in enumerate(data_list):
        cm = data['cm']
        per = data['%']
        ax1.plot(cm, per, color=colors[i % len(colors)], label=f'{file_names[i]}', linewidth=2)

    # 범례 설정
    ax1.legend(loc='lower left', fontsize=9, frameon=True, fancybox=True, shadow=True, borderpad=1.5)

    plt.title('FT-IR Spectrum')

    # y축 범위 설정 (FT-IR 표준 범위 40%에서 110%)
    plt.ylim(40, 110)

    # x축 눈금 범위 및 간격 설정
    plt.xticks([4000, 3750, 3500, 3250, 3000, 2750, 2500, 2250, 2000, 1750, 1500, 1250, 1000, 750, 650])


    # 기능기 관련 세로선 및 라벨 표시 함수
    def add_vertical_lines(x_start, x_end, color, label):
        plt.axvline(x=x_start, color=color, linestyle='--', linewidth=2)
        plt.axvline(x=x_end, color=color, linestyle='--', linewidth=2)
        plt.text((x_start + x_end) / 2, 115, label, horizontalalignment='center', fontsize=10, color=color)


    # 각 기능기에 대한 세로선 및 라벨 추가
    add_vertical_lines(3200, 3600, 'blue', 'O-H')
    add_vertical_lines(1500, 1650, 'red', 'C=C')
    add_vertical_lines(1400, 1600, 'purple', 'Aromatic\nC-C')
    add_vertical_lines(1200, 1270, 'orange', 'Aromatic\nC-O')



    # 구간 설정 (FT-IR에서 중요한 구간)
    regions = {
        'O-H': (3200, 3600),
        'C=C': (1500, 1650),
        'Aromatic C-C': (1400, 1600),
        'Aromatic C-O': (1200, 1270),
    }

    # 9-1.csv 파일을 기준으로 설정
    reference_data = data_list[-1]  # 마지막 파일이 9-1.csv라고 가정
    cm_ref = reference_data['cm']
    per_ref = reference_data['%']

    # 구간별로 DTW 점수 계산 및 출력
    for region_name, (start_wavenumber, end_wavenumber) in regions.items():
        print(f"Region: {region_name} ({start_wavenumber}-{end_wavenumber} cm⁻¹)")

        for i, data in enumerate(data_list[:-1]):  # 9-1.csv 파일은 기준이므로 제외
            cm = data['cm']
            per = data['%']

            # DTW 계산
            distance = calculate_dtw(cm, per, cm_ref, per_ref, start_wavenumber, end_wavenumber)

            if distance is not None:
                print(f"  DTW distance between {file_names[i]} and 9-1.csv: {distance}")
            else:
                print(f"  No data for {file_names[i]} in the range {start_wavenumber}-{end_wavenumber} cm⁻¹")

    # 평균 DTW 점수 계산 및 출력
    print("\nCalculating average DTW score...\n")
    for i, data in enumerate(data_list[:-1]):  # 9-1.csv 파일은 기준이므로 제외
        total_distance = 0
        valid_regions = 0

        for region_name, (start_wavenumber, end_wavenumber) in regions.items():
            cm = data['cm']
            per = data['%']

            # DTW 계산
            distance = calculate_dtw(cm, per, cm_ref, per_ref, start_wavenumber, end_wavenumber)

            if distance is not None:
                total_distance += distance
                valid_regions += 1

        if valid_regions > 0:
            average_distance = total_distance / valid_regions
            print(f"Average DTW distance between {file_names[i]} and 9-1.csv: {average_distance}")
        else:
            print(f"No valid regions for {file_names[i]} to calculate average DTW distance.")

    print("\nDTW 계산 완료.")
    # 그래프 보여주기
    plt.show()
