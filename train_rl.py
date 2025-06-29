import os
import csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO, A2C, DQN
from models.ByproductPredictorCNN import ByproductPredictorCNN
from models.TemperatureToDataPredictorCNN import TemperatureToDataPredictorCNN
from models.TemperatureToCompositionPredictor import TemperatureToCompositionPredictor
from TemperatureEnv import TemperatureEnv

# 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reward_modes = ["biofuel", "high_value", "functional"]
algorithms = {"PPO": PPO, "A2C": A2C, "DQN": DQN}
total_timesteps = 10000
log_dir = "logs_rl_compare_all"
os.makedirs(log_dir, exist_ok=True)

# 모델 로드
TGA_model = ByproductPredictorCNN(1, 761).to(device)
TGA_model.load_state_dict(torch.load('pth/tga.pth', map_location=device))
TGA_model.eval()

FTIR_model = TemperatureToDataPredictorCNN(input_size=1).to(device)
FTIR_model.load_state_dict(torch.load('pth/FTIR_model.pth', map_location=device))
FTIR_model.eval()

GCMS_model = TemperatureToCompositionPredictor(input_size=1, output_size=10).to(device)
GCMS_model.load_state_dict(torch.load('pth/composition_model.pth', map_location=device))
GCMS_model.eval()

all_results = []

# 주 목적별로 학습 및 로그 수집
for reward_mode in reward_modes:
    for algo_name, AlgoClass in algorithms.items():
        print(f"▶ 시작: {reward_mode} - {algo_name}")
        env = TemperatureEnv(TGA_model, FTIR_model, GCMS_model, device, reward_mode=reward_mode)
        model = AlgoClass("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=total_timesteps)
        print(f"✔ 학습 완료: {reward_mode} - {algo_name}")

        obs, _ = env.reset()
        log_path = os.path.join(log_dir, f"log_{reward_mode}_{algo_name}.csv")
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['step', 'temperature', 'reward'] + [f'GCMS_{i}' for i in range(10)]
            writer.writerow(header)

        best_reward = -float('inf')
        best_log = None

        for step in range(50):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            row = [step, obs[0], reward] + list(np.round(info['gcms'], 4))

            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

            if reward > best_reward:
                best_reward = reward
                best_log = row

            if terminated or truncated:
                break

        model.save(os.path.join(log_dir, f"{algo_name}_{reward_mode}"))
        if best_log is not None and len(best_log) == 13:
            all_results.append((reward_mode, algo_name, best_log[1], best_log[2], *best_log[3:]))
        else:
            print(f"[경고] {algo_name}-{reward_mode} 로그 누락 또는 손상")

        print(f"✅ 완료: {reward_mode} - {algo_name}")

# 시각화 결과 저장
summary_df = pd.DataFrame(all_results, columns=["Task", "Algorithm", "Best Temperature", "Best Reward"] + [f'GCMS_{i}' for i in range(10)])
summary_path = os.path.join(log_dir, "summary.csv")
summary_df.to_csv(summary_path, index=False)

# 텍스트 출력
print("📊 GCMS 최적 조성 요약:")
print(summary_df.to_string(index=False))

# GCMS 항목별 영향도 시각화
summary_long = summary_df.melt(id_vars=["Task", "Algorithm", "Best Temperature", "Best Reward"],
                                value_vars=[f'GCMS_{i}' for i in range(10)],
                                var_name="GCMS_Component", value_name="Composition")

plt.figure(figsize=(12, 6))
sns.barplot(data=summary_long, x="GCMS_Component", y="Composition", hue="Task")
plt.title("각 목적별 최적 온도에서의 GCMS 조성")
plt.ylabel("Composition (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()