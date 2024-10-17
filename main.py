import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from TGA.train import TGA_RandomForest
from FTIR import FTIR_Interpolate_combine, FTIR_RandomForest, FTIR_LightGBM, FTIR_AdaBoost
from TGA_dl import process_TGA_data, load_model, prepare_dataloader, train_model, evaluate_model, smooth_data
from models.ByproductPredictorCNN import ByproductDataset, ByproductPredictorCNN
from postprocessing import gaussian_smooth_data
from utill import FTIR_ImageMaker
import data_loader
import GCMS_to_xls
from TGA import TGA_interpolate, TGA_compare_interpolations, group
import TGA.TGA_evaluate
from preprocessing import reduce_by_temperature, interpolate_temperature, reduce_to_one_degree_interval

# 0 Time
# 1 Temperature
# 2 Weight
# 3 Weight%
# 4 Deriv. Weight
# 5 Detail_Time
# 6 Detail_Temperature
# 7 Round_G
# 8 Round_G == A:A
# 9 Amount of change
# 10 Detail_Weight
# 11 Round_K
# 12 Round_G == C

def TGA_augmentation(TGA_data, cat, target_temp, model_path='tga.pth', train_new_model=True):
    # 여러 보간 방법들을 비교 TF
    compare_interpolate_methods = False

    # 보간 진행 T/F
    TGA_interpolation = False

    # 특정 데이터 평가 T/F
    interpolated_data_evaluate = False

    # 입력 값에 따라 1 ~ 16.xls 중 필요한 파일 선정 및 온도 설정
    data, temp1, temp2 = process_TGA_data(TGA_data, cat, target_temp)
    # 모델 정의
    model = ByproductPredictorCNN(1, 761)

    # 이미 학습된 모델이 있는 경우 로드, 없으면 학습
    if not load_model(model, model_path) and train_new_model:
        # 데이터로부터 DataLoader 준비
        dataloader = prepare_dataloader(data)

        # 모델 학습
        train_model(model, dataloader, model_path)

    # 모델 평가 ######## Taget 온도 바꾸려면 여기
    predicted_byproducts = evaluate_model(model)

    # Gaussian smoothing 적용
    predicted_byproducts_smoothed = smooth_data(predicted_byproducts, sigma=2)

    return predicted_byproducts_smoothed



if __name__ == '__main__' :
    # condition_data, TGA_data, FTIR_data, GCMS_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='TGA')
    condition_data, TGA_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='TGA')


    # cat = input('촉매 입력 No Pt/C Ru/C Raney Ni ')
    cat = 'No'

    # target_temp = int(input('온도 입력 250 ~ 400'))
    target_temp = 275

    TGA_bool = True
    FTIR_bool = False
    GCMS_bool = False

    if TGA_bool :
        augmented_data = TGA_augmentation(TGA_data, cat, target_temp)
    elif FTIR_bool :

        # 모든 함수에서 경로 변경이 필요

        for j in range(1, 16) :
            for i in range(1, 3) :
              # 데이터 보간 및 병합(머신러닝 용)
              FTIR_Interpolate_combine.Interpolate_combine(FTIR_data[i][1], FTIR_data[i + 1][1], f"{j}-{i}", f"{j}-{i + 1}")


        # # 학습 및 모델 저장 및 예측 결과 저장
        # FTIR_RandomForest.FTIR_RF(only_predict=False)
        # FTIR_LightGBM.FTIR_GBM(only_predict=False)
        # FTIR_AdaBoost.FTIR_ABoost(only_predict=False)
        #
        # # ImageMaker를 이용하여 그래프 생성 및 이미지 저장
        # FTIR_ImageMaker.makeImage(save=False)

    elif GCMS_bool :

        # GCMS 파일에서 Bold된 글자들을 엑셀파일로 저장
        # dataset/GC-MS_to_xls/*.xls

        mass = []

        for i in GCMS_data.split():
            try:
                mass.append(float(i))
            except:
                continue

        test = []

        for i in range(4):
            for j in range(4):
                adding = []
                for k in range(0, 33, 4):
                    adding.append(mass[int(((i * len(mass)) / 4) + j + k)])

                adding.append(100 - sum(adding))

                adding.append(sum(adding))

                test.append(adding)

        for i in range(len(test)):
            GCMS_to_xls.save_data_to_excel(test[i], f"{i + 1}.xls")
