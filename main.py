import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from pandas import read_csv
from torch import nn, optim
from torch.utils.data import DataLoader
import os

from GCMS.GCMS_drawModel_graph import plot_grouped_bar_chart
from TGA.train import TGA_RandomForest
from FTIR import FTIR_Interpolate_combine, FTIR_RandomForest, FTIR_LightGBM, FTIR_AdaBoost
from TGA_dl import process_TGA_data, load_model, prepare_dataloader, train_model, evaluate_model, smooth_data
from FTIR_dl import preprocess_FTIR_data, train_and_evaluate, predict_and_plot
from GCMS_dl import train_and_evaluate as GCMS_train_and_evaluate
from models.ByproductPredictorCNN import ByproductDataset, ByproductPredictorCNN
from models.MoE import MVAE_MoE
from models.TemperatureToCompositionPredictor import TemperatureToCompositionPredictor
from models.TemperatureToDataPredictorCNN import TemperatureToDataPredictorCNN
from models.ml import compare_models
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

def GCMS_augmentation(data, cat, target_temp, device):
    data = data[data['Catalyst'] == cat]
    data_250 = data[data['temp'] == 250]['Value'].values[:-1]
    data_300 = data[data['temp'] == 300]['Value'].values[:-1]
    data_350 = data[data['temp'] == 350]['Value'].values[:-1]
    data_400 = data[data['temp'] == 400]['Value'].values[:-1]

    combined_data = np.vstack((data_250, data_300, data_350, data_400))

    temperature_data = np.array([250, 300, 350, 400], dtype=np.float32).reshape(-1, 1)
    temperature_data = torch.tensor(temperature_data).unsqueeze(1).to(device)
    composition_data = torch.tensor(combined_data / 100, dtype=torch.float32).to(device)
    compare_models(composition_data.detach().cpu().numpy(), target_temp[0])
    model = TemperatureToCompositionPredictor(input_size=1, output_size=10).to(device)
    predicted_composition = GCMS_train_and_evaluate(model, temperature_data, composition_data, target_temp, device)
    return predicted_composition

def FTIR_augmentation(FTIR_data, target_temp, device):
    MODEL_PATH = "pth/FTIR_model.pth"

    preprocessed_data = preprocess_FTIR_data(FTIR_data)
    # 입력 및 출력 데이터 설정
    temperature_data = np.array([250, 300, 350, 400], dtype=np.float32).reshape(-1, 1)
    output_data = preprocessed_data[0][:, 1, :]  # (4, 3476) 형태의 데이터

    test = []

    for i in FTIR_data:
        test.append(i[1])

    compare_models(np.asarray(output_data), target_temp[0], True)

    # PyTorch 텐서로 변환
    temperatures = torch.tensor(temperature_data).unsqueeze(1).to(device)  # (batch_size, 1, 1)
    outputs = torch.tensor(output_data).to(device)

    # 모델 초기화
    model = TemperatureToDataPredictorCNN(input_size=1).to(device)

    # 모델이 이미 저장되어 있으면 로드, 아니면 학습
    if os.path.exists(MODEL_PATH):
        print("Loading the existing model...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("Training the model...")
        train_and_evaluate(model, temperatures, outputs)
        torch.save(model.state_dict(), MODEL_PATH)  # 학습 후 모델 저장

    # 새로운 온도에서의 예측 및 시각화
    new_temperatures = torch.tensor([target_temp], dtype=torch.float32).unsqueeze(1).to(device)
    predict_ftir = predict_and_plot(model, preprocessed_data, new_temperatures)

    return new_temperatures

def TGA_augmentation(TGA_data, cat, target_temp, device, model_path='pth/tga.pth', train_new_model=True):


    # 입력 값에 따라 1 ~ 16.xls 중 필요한 파일 선정 및 온도 설정
    data, temp1, temp2 = process_TGA_data(TGA_data, cat, target_temp[0])
    # 모델 정의

    compare_models(np.asarray(data)[:,2,:], 1, target_temp[0])

    model = ByproductPredictorCNN(1, 761)

    # 이미 학습된 모델이 있는 경우 로드, 없으면 학습
    if not load_model(model, model_path, device) and train_new_model:
        # 데이터로부터 DataLoader 준비
        dataloader = prepare_dataloader(data, device)

        # 모델 학습
        train_model(model, dataloader, model_path, device)

    # 모델 평가 ######## Taget 온도 바꾸려면 여기
    predicted_byproducts = evaluate_model(model, device)

    # Gaussian smoothing 적용
    predicted_byproducts_smoothed = smooth_data(predicted_byproducts, sigma=2)

    return predicted_byproducts_smoothed



if __name__ == '__main__' :

    flag = False

    '''
    True  : Augmentation
    False : FTIR + TGA to GCMS 
    '''

    # GPU 유무에 따라서 cuda or cpu 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    if flag:
        condition_data, TGA_data, FTIR_data, GCMS_data = data_loader.load_data(data_loader.ROOT_DIR)
        # condition_data, TGA_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='TGA')
        # condition_data, FTIR_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='FTIR')
        # condition_data, GCMS_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='GCMS')

        # cat = input('촉매 입력 NoCat PtC RuC RN')
        cat = 'NoCat'

        # target_temp = int(input('온도 입력 250 ~ 400'))
        target_temp = [275]

        TGA_bool = False
        FTIR_bool = False
        GCMS_bool = False

        if TGA_bool :
            augmented_TGA_data = TGA_augmentation(TGA_data, cat, target_temp, device)

        if FTIR_bool :
            augmented_FTIR_data = FTIR_augmentation(FTIR_data, target_temp, device)

        if GCMS_bool :

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
            prediction = GCMS_augmentation(data, cat, target_temp, device)
            # GCMS_RandomForest.process_and_train_tga_gcms()

            plot_grouped_bar_chart(prediction, target_temp[0])

    else:
        TGA_model_path = 'pth/tga.pth'
        TGA_model = ByproductPredictorCNN(1, 761).to(device)
        TGA_model.load_state_dict(torch.load(TGA_model_path, weights_only=True))
        TGA_model.eval()

        FTIR_model_path = 'pth/FTIR_model.pth'
        FTIR_model = TemperatureToDataPredictorCNN(input_size=1).to(device)
        FTIR_model.load_state_dict(torch.load(FTIR_model_path, weights_only=True))
        FTIR_model.eval()

        GCMS_model_path = 'pth/composition_model.pth'
        GCMS_model = TemperatureToCompositionPredictor(input_size=1, output_size=10).to(device)
        GCMS_model.load_state_dict(torch.load(GCMS_model_path, weights_only=True))
        GCMS_model.eval()

        new_temperatures = np.arange(200, 401, 0.01, dtype=np.float32).reshape(-1, 1)
        new_temperatures = torch.tensor(new_temperatures).unsqueeze(1).to(device)

        TGA_data = TGA_model(new_temperatures)
        FTIR_data = FTIR_model(new_temperatures)
        GCMS_data = GCMS_model(new_temperatures)

        print("A")

        # 손실 함수 정의
        def loss_function(recon_x1, x1, recon_x2, x2, output, target, mu, log_var):
            recon_loss1 = nn.MSELoss()(recon_x1, x1)
            recon_loss2 = nn.MSELoss()(recon_x2, x2)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            pred_loss = nn.MSELoss()(output, target)
            return recon_loss1 + recon_loss2 + kl_loss + pred_loss


        train = False

        # 모델 초기화
        input_dim1 = 761
        input_dim2 = 3476
        latent_dim = 256
        output_dim = 10
        num_experts = 3

        model = MVAE_MoE(input_dim1, input_dim2, latent_dim, output_dim, num_experts).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.1)

        # print(pd.DataFrame(GCMS_data.detach().cpu().numpy())[0].mean())

        if train :

            # TGA_data = TGA_data.unsqueeze(1).to('cuda')
            # FTIR_data = FTIR_data.unsqueeze(1).to('cuda')
            # GCMS_data = GCMS_data.unsqueeze(1).to('cuda')

            epochs = 100
            for epoch in range(epochs):
                model.train()

                # 역전파 (backward) 및 최적화
                optimizer.zero_grad()
                recon_x1, recon_x2, output, mu, log_var = model(TGA_data, FTIR_data)
                loss = loss_function(recon_x1, TGA_data, recon_x2, FTIR_data, output, GCMS_data, mu, log_var)
                loss.backward(retain_graph=True)
                optimizer.step()
                scheduler.step()

                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

            # 모델 저장
            model_path = 'pth/MoE_012.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        else :

            MoE_path = 'pth/MoE_012.pth'
            model.load_state_dict(torch.load(MoE_path, weights_only=True))
            model.eval()

            # TGA_data = TGA_data.unsqueeze(1).to(device)
            # FTIR_data = FTIR_data.unsqueeze(1).to(device)
            # GCMS_data = GCMS_data.unsqueeze(1).to(device)

            # TGA_data와 FTIR_data는 이전에 처리된 입력 데이터여야 함
            TGA_data = TGA_data.to(device)
            FTIR_data = FTIR_data.to(device)

            # 모델 예측 수행
            with torch.no_grad():  # 평가 시에는 gradient를 계산하지 않음
                recon_x1, recon_x2, output, mu, log_var = model(TGA_data, FTIR_data)

            # 예측 결과 출력
            print("Reconstructed TGA Data:", recon_x1)
            print("Reconstructed FTIR Data:", recon_x2)
            print("Predicted GCMS Data:", output)
            print()

