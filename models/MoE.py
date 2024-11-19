import torch
from torch import nn


class Encoder1(nn.Module):
    # 첫 번째 입력에 대한 인코더 정의
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