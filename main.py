import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import read_csv
from torch import nn, optim
from torch.utils.data import DataLoader
import os

from TGA.train import TGA_RandomForest
from FTIR import FTIR_Interpolate_combine, FTIR_RandomForest, FTIR_LightGBM, FTIR_AdaBoost
from TGA_dl import process_TGA_data, load_model, prepare_dataloader, train_model, evaluate_model, smooth_data
from FTIR_dl import preprocess_FTIR_data, train_and_evaluate, predict_and_plot
from GCMS_dl import train_and_evaluate as GCMS_train_and_evaluate
from models.ByproductPredictorCNN import ByproductDataset, ByproductPredictorCNN
from models.TemperatureToCompositionPredictor import TemperatureToCompositionPredictor
from models.TemperatureToDataPredictorCNN import TemperatureToDataPredictorCNN
from postprocessing import gaussian_smooth_data
from utill import FTIR_ImageMaker
import data_loader
import GCMS_to_csv
from TGA import TGA_interpolate, TGA_compare_interpolations, group
import TGA.TGA_evaluate
from preprocessing import reduce_by_temperature, interpolate_temperature, reduce_to_one_degree_interval

from GCMS import GCMS_add_Condition, GCMS_combine, GCMS_RandomForest

from preprocessing import reduce_by_temperature, interpolate_temperature, reduce_to_one_degree_interval, \
    group_and_average_data, group_preprocessed_data, clip_data_to_100, process_data_with_log


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

def GCMS_augmentation(data,cat):
    data = data[data['Catalyst'] == cat]
    data_250 = data[data['temp'] == 250]['Value'].values[:-1]
    data_300 = data[data['temp'] == 300]['Value'].values[:-1]
    data_350 = data[data['temp'] == 350]['Value'].values[:-1]
    data_400 = data[data['temp'] == 400]['Value'].values[:-1]

    combined_data = np.vstack((data_250, data_300, data_350, data_400))

    temperature_data = np.array([250, 300, 350, 400], dtype=np.float32).reshape(-1, 1)
    temperature_data = torch.tensor(temperature_data).unsqueeze(1).to('cuda')
    composition_data = torch.tensor(combined_data / 100, dtype=torch.float32).to('cuda')

    model = TemperatureToCompositionPredictor(input_size=1, output_size=10).to('cuda')
    predicted_composition = GCMS_train_and_evaluate(model, temperature_data, composition_data)
    return predicted_composition

def FTIR_augmentation(FTIR_data):
    preprocessed_data = preprocess_FTIR_data(FTIR_data)
    # 입력 및 출력 데이터 설정
    temperature_data = np.array([250, 300, 350, 400], dtype=np.float32).reshape(-1, 1)
    output_data = preprocessed_data[0][:, 1, :]  # (4, 3476) 형태의 데이터

    # PyTorch 텐서로 변환
    temperatures = torch.tensor(temperature_data).unsqueeze(1).to('cuda')  # (batch_size, 1, 1)
    outputs = torch.tensor(output_data).to('cuda')

    # 모델 초기화 및 학습
    model = TemperatureToDataPredictorCNN(input_size=1).to('cuda')
    train_and_evaluate(model, temperatures, outputs)

    # 새로운 온도에서의 예측 및 시각화
    new_temperatures = torch.tensor([[275.0], [325.0], [375.0]], dtype=torch.float32).unsqueeze(1).to('cuda')
    predict_and_plot(model, preprocessed_data, new_temperatures)
    return new_temperatures

def TGA_augmentation(TGA_data, cat, target_temp, model_path='tga.pth', train_new_model=True):

    # GPU 유무에 따라서 cuda or cpu 설정
    computer_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 입력 값에 따라 1 ~ 16.xls 중 필요한 파일 선정 및 온도 설정
    data, temp1, temp2 = process_TGA_data(TGA_data, cat, target_temp)
    # 모델 정의
    model = ByproductPredictorCNN(1, 761)

    # 이미 학습된 모델이 있는 경우 로드, 없으면 학습
    if not load_model(model, model_path, computer_device) and train_new_model:
        # 데이터로부터 DataLoader 준비
        dataloader = prepare_dataloader(data, computer_device)

        # 모델 학습
        train_model(model, dataloader, model_path, computer_device)

    # 모델 평가 ######## Taget 온도 바꾸려면 여기
    predicted_byproducts = evaluate_model(model, computer_device)

    # Gaussian smoothing 적용
    predicted_byproducts_smoothed = smooth_data(predicted_byproducts, sigma=2)

    return predicted_byproducts_smoothed



if __name__ == '__main__' :
    # condition_data, TGA_data, FTIR_data, GCMS_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='TGA')
    # condition_data, TGA_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='TGA')
    # condition_data, FTIR_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='FTIR')
    condition_data, GCMS_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='GCMS')

    # cat = input('촉매 입력 No PtC RuC RN')
    cat = 'No'

    # target_temp = int(input('온도 입력 250 ~ 400'))
    target_temp = 275

    TGA_bool = False
    FTIR_bool = False
    GCMS_bool = True

    if TGA_bool :
        augmented_TGA_data = TGA_augmentation(TGA_data, cat, target_temp)
    elif FTIR_bool :
        augmented_FTIR_data = FTIR_augmentation(FTIR_data)
    elif GCMS_bool :

        # 추출 파일이 없는 경우 추출을 진행
        if not(os.path.exists('dataset/GC-MS_to_csv/16.xls')) :
            GCMS_to_csv.process_and_export_gcms_data(GCMS_data)

            # 파일명에 따라 촉매, 전처리 온도 컬럼을 추가
            path = 'dataset/GC-MS_to_csv/'
            GCMS_add_Condition.process_csv_files_in_directory(path)

        # GC-MS pdf에서 추출하여 합친 파일이 있는 경우 그대로 읽어와서 할당
        # 없는 경우 합친 파일 생성 후 할당
        if not(os.path.exists('dataset/combined_GCMS.csv')):
            GCMS_combine.combine_csv_files()


        data = read_csv('dataset/combined_GCMS.csv')
        prediction = GCMS_augmentation(data,"NoCat")

       # GCMS_RandomForest.process_and_train_tga_gcms()