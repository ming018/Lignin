import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from TGA.train import TGA_RandomForest
from FTIR import FTIR_Interpolate_combine, FTIR_RandomForest, FTIR_LightGBM, FTIR_AdaBoost
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
        # Interpolated_TGA 데이터 보간 및 보간된 데이터 평가
        # 저장 위치, dataset/train/Interpolated_TGA/
        # 시간, 기기온도, 무게%, 무게 변화율만 추출

        # 여러 보간 방법들을 비교 TF
        compare_interpolate_methods = False

        # 보간 진행 T/F
        TGA_interpolation = False

        # 특정 데이터 평가 T/F
        interpolated_data_evaluate = False

        # 입력 값에 따라 1 ~ 16.xls 중 필요한 파일 선정 및 온도 설정
        data_for_return, temp1, temp2 = group.process_group_for_TGA(TGA_data, cat, target_temp)
        data = [reduce_by_temperature(data) for data in data_for_return]
        data = [interpolate_temperature(d, 40, 800) for d in data]
        data = [reduce_to_one_degree_interval(d) for d in data]

        # plt.plot(data[0][0],data[0][2])
        # plt.plot(data[1][0],data[1][2])
        plt.plot(data[2][0],data[2][2])
        plt.plot(data[3][0],data[3][2])
        # plt.show()

        temperature_data = torch.tensor([[250],[300],[350],[400]],dtype=torch.float).to('cuda')
        byproduct_data = torch.tensor(data,dtype=torch.float).to('cuda')

        # Dataset 및 DataLoader 생성
        batch_size = 16
        dataset = ByproductDataset(temperature_data, byproduct_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = ByproductPredictorCNN(1, 761)
        model.to('cuda')
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        epochs = 10000  # 예시로 10 에포크로 설정
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for i, (temperatures, byproducts) in enumerate(dataloader):
                # 데이터 배치 처리
                temperatures = temperatures.unsqueeze(1)  # (batch_size, 1, input_size)로 변환 (CNN 입력 요구 형태)

                # 순전파(forward) 계산
                outputs = model(temperatures)
                loss = criterion(outputs, byproducts[:,2,:])

                # 역전파(backward) 및 최적화
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 손실 누적
                running_loss += loss.item()
            # 에포크마다 평균 손실 출력
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

        model.eval()
        with torch.no_grad():
            new_temperatures =  torch.tensor([[260],[320],[370],[420]],dtype=torch.float).to('cuda')
            new_temperatures = new_temperatures.unsqueeze(1)
            predicted_byproducts = model(new_temperatures)
            # print("Predicted byproducts for new temperatures:")
            # print(predicted_byproducts)

        predicted_byproducts = predicted_byproducts.detach().cpu().numpy()

        # predicted_byproducts 각 행에 Gaussian smoothing 적용
        sigma = 2  # 스무딩 강도 조절
        predicted_byproducts_smoothed_gaussian = np.array(
            [gaussian_smooth_data(byproduct, sigma) for byproduct in predicted_byproducts])

        # for x in predicted_byproducts:
        #     plt.plot(data[0][0],x)
        plt.plot(data[0][0],predicted_byproducts_smoothed_gaussian[2])
        plt.show()

        plt.plot(data[0][0], predicted_byproducts_smoothed_gaussian[2])
        plt.show()


        start_temp = 250 + 50 * temp1
        limit_temp = 250 + 50 * temp2

        if compare_interpolate_methods :
            TGA_compare_interpolations.compare_main(data_for_return[temp1], data_for_return[temp1])

        if TGA_interpolation :
            TGA_interpolate.interpolate(data_for_return, temp1, temp2)

        if interpolated_data_evaluate :
            df = pd.read_csv(f"dataset/Interpolated_TGA/predict_{cat}_{target_temp}.csv")
            predict_data = df['Deriv. Weight']

            TGA.TGA_evaluate.TGA_evaluate(target_temp, cat, data_for_return[temp1], data_for_return, predict_data, start_temp, limit_temp)


        # 학습 함수 적용, 수정 필요
        TGA_RandomForest.TGA_RF(only_predict=True)



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
