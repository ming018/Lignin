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

def GCMS_augmentation(data,cat,device):

    data = data[data['Catalyst'] == cat]
    data_250 = data[data['temp'] == 250]['Value'].values[:-1]
    data_300 = data[data['temp'] == 300]['Value'].values[:-1]
    data_350 = data[data['temp'] == 350]['Value'].values[:-1]
    data_400 = data[data['temp'] == 400]['Value'].values[:-1]

    combined_data = np.vstack((data_250, data_300, data_350, data_400))

    temperature_data = np.array([250, 300, 350, 400], dtype=np.float32).reshape(-1, 1)
    temperature_data = torch.tensor(temperature_data).unsqueeze(1).to(device)
    composition_data = torch.tensor(combined_data / 100, dtype=torch.float32).to(device)
    compare_models(composition_data.detach().cpu().numpy())
    model = TemperatureToCompositionPredictor(input_size=1, output_size=10).to(device)
    predicted_composition = GCMS_train_and_evaluate(model, temperature_data, composition_data)
    return predicted_composition

def FTIR_augmentation(FTIR_data, device):
    MODEL_PATH = "FTIR_model.pth"

    preprocessed_data = preprocess_FTIR_data(FTIR_data)
    # 입력 및 출력 데이터 설정
    temperature_data = np.array([250, 300, 350, 400], dtype=np.float32).reshape(-1, 1)
    output_data = preprocessed_data[0][:, 1, :]  # (4, 3476) 형태의 데이터

    compare_models(np.asarray(output_data))

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
    new_temperatures = torch.tensor([[275.0], [325.0], [375.0]], dtype=torch.float32).unsqueeze(1).to(device)
    predict_and_plot(model, preprocessed_data, new_temperatures)
    return new_temperatures

def TGA_augmentation(TGA_data, cat, target_temp, device, model_path='tga.pth', train_new_model=True):

    # 입력 값에 따라 1 ~ 16.xls 중 필요한 파일 선정 및 온도 설정
    data, temp1, temp2 = process_TGA_data(TGA_data, cat, target_temp)
    # 모델 정의

    compare_models(np.asarray(data)[:,2,:])

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

    flag = True
    '''
    True  : Augmentation
    False : FTIR + TGA to GCMS 
    '''

    # GPU 유무에 따라서 cuda or cpu 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if flag:
        # condition_data, TGA_data, FTIR_data, GCMS_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='TGA')
        # condition_data, TGA_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='TGA')
        # condition_data, FTIR_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='FTIR')
        condition_data, GCMS_data = data_loader.load_data(data_loader.ROOT_DIR, data_type='GCMS')

        # cat = input('촉매 입력 NoCat PtC RuC RN')
        cat = 'NoCat'

        # target_temp = int(input('온도 입력 250 ~ 400'))
        target_temp = 275

        TGA_bool = False
        FTIR_bool = False
        GCMS_bool = True

        if TGA_bool :
            augmented_TGA_data = TGA_augmentation(TGA_data, cat, target_temp, device)

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
            prediction = GCMS_augmentation(data, cat, device)
            # GCMS_RandomForest.process_and_train_tga_gcms()
    else:
        TGA_model_path = 'tga.pth'
        TGA_model = ByproductPredictorCNN(1, 761).to(device)
        TGA_model.load_state_dict(torch.load(TGA_model_path, weights_only=True))
        TGA_model.eval()

        FTIR_model_path = 'FTIR_model.pth'
        FTIR_model = TemperatureToDataPredictorCNN(input_size=1).to(device)
        FTIR_model.load_state_dict(torch.load(FTIR_model_path, weights_only=True))
        FTIR_model.eval()

        GCMS_model_path = 'composition_model.pth'
        GCMS_model = TemperatureToCompositionPredictor(input_size=1, output_size=10).to(device)
        GCMS_model.load_state_dict(torch.load(GCMS_model_path, weights_only=True))
        GCMS_model.eval()

        new_temperatures = np.arange(200, 401, 0.01, dtype=np.float32).reshape(-1, 1)
        new_temperatures = torch.tensor(new_temperatures).unsqueeze(1).to(device)

        TGA_data = TGA_model(new_temperatures)
        FTIR_data = FTIR_model(new_temperatures)
        GCMS_data = GCMS_model(new_temperatures)

        print("A")


        # 첫 번째 입력에 대한 인코더 정의
        class Encoder1(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super(Encoder1, self).__init__()
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, latent_dim * 2)  # mu와 log_var를 위해 2배 크기
                )

            def forward(self, x):
                h = self.fc(x)
                mu, log_var = h.chunk(2, dim=-1)
                return mu, log_var


        # 두 번째 입력에 대한 인코더 정의
        class Encoder2(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super(Encoder2, self).__init__()
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, latent_dim * 2)
                )

            def forward(self, x):
                h = self.fc(x)
                mu, log_var = h.chunk(2, dim=-1)
                return mu, log_var


        # 디코더 정의
        class Decoder(nn.Module):
            def __init__(self, latent_dim, output_dim):
                super(Decoder, self).__init__()
                self.fc = nn.Sequential(
                    nn.Linear(latent_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, output_dim)
                )

            def forward(self, z):
                return self.fc(z)


        # 전문가 네트워크 정의
        class Expert(nn.Module):
            def __init__(self, latent_dim, output_dim):
                super(Expert, self).__init__()
                self.fc = nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, output_dim)
                )

            def forward(self, z):
                return self.fc(z)


        # 게이트 네트워크 정의
        class GatingNetwork(nn.Module):
            def __init__(self, latent_dim, num_experts):
                super(GatingNetwork, self).__init__()
                self.fc = nn.Sequential(
                    nn.Linear(latent_dim, num_experts)
                )

            def forward(self, z):
                return self.fc(z)


        # MVAE + MoE 모델 정의
        class MVAE_MoE(nn.Module):
            def __init__(self, input_dim1, input_dim2, latent_dim, output_dim, num_experts):
                super(MVAE_MoE, self).__init__()
                self.encoder1 = Encoder1(input_dim1, latent_dim)
                self.encoder2 = Encoder2(input_dim2, latent_dim)
                self.decoder1 = Decoder(latent_dim, input_dim1)
                self.decoder2 = Decoder(latent_dim, input_dim2)
                self.experts = nn.ModuleList([Expert(latent_dim, output_dim) for _ in range(num_experts)])
                self.gating_network = GatingNetwork(latent_dim, num_experts)

            def reparameterize(self, mu, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std

            def forward(self, x1, x2):
                # 인코딩
                mu1, log_var1 = self.encoder1(x1)
                mu2, log_var2 = self.encoder2(x2)

                # 평균 결합
                mu = (mu1 + mu2) / 2
                log_var = (log_var1 + log_var2) / 2

                # 잠재 벡터 샘플링
                z = self.reparameterize(mu, log_var)

                # 재구성
                recon_x1 = self.decoder1(z)
                recon_x2 = self.decoder2(z)

                # 전문가 네트워크 예측
                expert_outputs = torch.stack([expert(z) for expert in self.experts], dim=1)
                gating_weights = torch.softmax(self.gating_network(z), dim=1)
                output = torch.sum(gating_weights.unsqueeze(2) * expert_outputs, dim=1)

                return recon_x1, recon_x2, output, mu, log_var


        # 손실 함수 정의
        def loss_function(recon_x1, x1, recon_x2, x2, output, target, mu, log_var):
            recon_loss1 = nn.MSELoss()(recon_x1, x1)
            recon_loss2 = nn.MSELoss()(recon_x2, x2)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            pred_loss = nn.MSELoss()(output, target)
            return recon_loss1 + recon_loss2 + kl_loss + pred_loss


        # 모델 초기화
        input_dim1 = 761
        input_dim2 = 3476
        latent_dim = 256
        output_dim = 10
        num_experts = 3

        model = MVAE_MoE(input_dim1, input_dim2, latent_dim, output_dim, num_experts).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # TGA_data = TGA_data.unsqueeze(1).to('cuda')
        # FTIR_data = FTIR_data.unsqueeze(1).to('cuda')
        # GCMS_data = GCMS_data.unsqueeze(1).to('cuda')

        epochs = 10000
        for epoch in range(epochs):
            model.train()

            # 역전파 (backward) 및 최적화
            optimizer.zero_grad()
            recon_x1, recon_x2, output, mu, log_var = model(TGA_data, FTIR_data)
            loss = loss_function(recon_x1, TGA_data, recon_x2, FTIR_data, output, GCMS_data, mu, log_var)
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step(loss)

            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

