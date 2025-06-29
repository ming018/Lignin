# TemperatureEnv.py
from gymnasium import Env, spaces
import numpy as np
import torch

class TemperatureEnv(Env):
    def __init__(self, tga_model, ftir_model, gcms_model, device, reward_mode="biofuel"):
        super(TemperatureEnv, self).__init__()
        self.tga_model = tga_model
        self.ftir_model = ftir_model
        self.gcms_model = gcms_model
        self.device = device

        self.current_temp = 500.0
        self.min_temp = 200.0
        self.max_temp = 600.0
        self.step_size = 5.0# Temperature Step Size

        self.reward_mode = reward_mode  # ✅ 목적 선택 가능

        # 🎯 목적별 온도 구간 설정
        if reward_mode == "biofuel":
            self.min_temp = 400.0
            self.max_temp = 600.0
            self.init_temp = 500.0
        elif reward_mode == "high_value":
            self.min_temp = 350.0
            self.max_temp = 500.0
            self.init_temp = 425.0
        elif reward_mode == "functional":
            self.min_temp = 350.0
            self.max_temp = 450.0
            self.init_temp = 400.0
        else:
            raise ValueError(f"Invalid reward_mode: {reward_mode}")

        self.current_temp = self.init_temp

        self.observation_space = spaces.Box(low=self.min_temp, high=self.max_temp, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # -5도, 0도, +5도

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_temp = 400.0
        return np.array([self.current_temp], dtype=np.float32), {}

    def step(self, action):
        delta_temp = (action - 1) * self.step_size
        self.current_temp = np.clip(self.current_temp + delta_temp, self.min_temp, self.max_temp)

        t = torch.tensor([[self.current_temp]], dtype=torch.float32).unsqueeze(1).to(self.device)

        with torch.no_grad():
            gcms_output = self.gcms_model(t).squeeze().cpu().numpy()

        reward = self._calculate_reward(gcms_output)
        obs = np.array([self.current_temp], dtype=np.float32)
        terminated = False
        truncated = False
        info = {'gcms': gcms_output}

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, gcms_output):
        if self.reward_mode == "biofuel":
            weights = np.array([0.9, 0.5, 1.0, 1.0, 1.0, 0.2, 0.4, 0.1, -0.5, -0.8])
        elif self.reward_mode == "high_value":
            weights = np.array([1.0, 1.0, 0.2, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.1])
        elif self.reward_mode == "functional":
            weights = np.array([1.0, 1.0, 0.2, 1.0, -0.8, 0.1, 1.0, 0.5, 0.4, 0.2])
        else:
            raise ValueError(f"Invalid reward_mode: {self.reward_mode}")
        return float(np.dot(gcms_output, weights))
