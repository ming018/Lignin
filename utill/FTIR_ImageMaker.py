import os
import pandas as pd
import matplotlib.pyplot as plt

def makeImage(save = False) :
    # CSV 파일 경로 설정
    file_path1 = '/content/drive/MyDrive/lignin/FT-IR/data/NOCAT/NOCAT300/5-1.csv'

    # CSV 파일에서 데이터 읽기
    data1 = pd.read_csv(file_path1)

    cm1 = data1['cm']
    per1 = data1['%']

    # 그래프 그리기
    fig, ax1 = plt.subplots()
    ax1.set_xlim(4000, 650)
    ax1.set_ylim(40, 110)

    # 선의 굵기 설정 (linewidth=2.5로 예시)
    ax1.plot(cm1, per1, color='blue', linewidth=2)
    ax1.set_yticks([])
    ax1.set_xticks([])
    # y축 범위 설정 (FT-IR 표준 범위 40%에서 110%)
    plt.ylim(40, 110)

    # 그래프 보여주기
    plt.show()

    if save:
        # 그래프를 이미지로 저장

        # CSV 파일들이 저장된 디렉터리 경로 설정
        directory_path = '/content/drive/MyDrive/lignin/FT-IR/data/NOCAT/NOCAT300/'
        # 저장할 이미지 파일 경로 설정
        output_dir = '/content/drive/MyDrive/lignin/FT-IR/graphs/'

        # 디렉터리가 존재하지 않으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 고유한 파일 이름을 위한 순번 변수
        file_counter = 1

        # 지정한 디렉터리에서 모든 CSV 파일을 읽음
        for filename in os.listdir(directory_path):
            if filename.endswith(".csv"):
                # 파일 경로 설정
                file_path = os.path.join(directory_path, filename)

                # CSV 파일에서 데이터 읽기
                data1 = pd.read_csv(file_path)

                # 온도 및 퍼센트 데이터 추출
                cm1 = data1['cm']
                per1 = data1['%']

                # 그래프 그리기
                fig, ax1 = plt.subplots()
                ax1.set_xlim(4000, 650)
                ax1.set_ylim(40, 110)

                # 선의 굵기 설정 (linewidth=2)
                ax1.plot(cm1, per1, color='blue', linewidth=2)

                # X축 및 Y축 눈금 제거
                ax1.set_yticks([])
                ax1.set_xticks([])

                # y축 범위 설정 (FT-IR 표준 범위 40%에서 110%)
                plt.ylim(40, 110)

                # 고유한 파일 이름 생성 (순번 추가)
                unique_name = f"{filename[:-4]}_{file_counter}.png"
                output_file = os.path.join(output_dir, unique_name)

                plt.savefig(output_file)

                # 그래프 닫기 (메모리 누수 방지)
                plt.close()

                # 순번 증가
                file_counter += 1

        print("모든 그래프가 고유한 이름으로 저장되었습니다.")

