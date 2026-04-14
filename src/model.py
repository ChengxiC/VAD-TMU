import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from src.memory import Memory_Unit
from src.translayer import Transformer
from src.utils import norm
from torch.nn import functional as F

# 四元组损失
class QuadrupletLoss(nn.Module):
    def __init__(self, margin1=1.0, margin2=0.5):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1  # Margin for triplet loss
        self.margin2 = margin2  # Additional margin for the second negative pair
        self.triplet_loss = nn.TripletMarginLoss(margin=margin1)

    def forward(self, anchor, positive, negative1, negative2):
        loss_triplet = self.triplet_loss(anchor, positive, negative1)
        distance_negative2 = F.pairwise_distance(anchor, negative2, p=2)
        distance_negative1 = F.pairwise_distance(anchor, negative1, p=2)
        loss_quadruplet = torch.mean(F.relu(distance_negative1 - distance_negative2 + self.margin2))
        return loss_triplet + loss_quadruplet


# 时序特征的提取,使用一维卷积层（Conv1d）来处理输入的时间序列数据
class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x


class ADCLS_head(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim), nn.Sigmoid())

    def forward(self, x):
        return self.mlp(x)


# 主模型
class TMwVAD(Module):
    def __init__(self, input_size, flag, a_nums, n_nums, x_nums):
        super().__init__()
        self.flag = flag  # train or test
        self.a_nums = a_nums
        self.n_nums = n_nums

        self.layernorm = nn.LayerNorm(512)

        self.embedding = Temporal(input_size, 512)
        self.triplet = nn.TripletMarginLoss(margin=1)
        self.quadruplet_loss = QuadrupletLoss(margin1=1.0, margin2=0.5)
        self.cls_head = ADCLS_head(1024, 1)
        self.Amemory = Memory_Unit(nums=a_nums, dim=512)
        self.Nmemory = Memory_Unit(nums=n_nums, dim=512)
        self.Xmemory = Memory_Unit(nums=x_nums, dim=512)  # 模糊记忆模块

        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout=0.5)
        self.encoder_mu = nn.Sequential(nn.Linear(512, 512))
        self.encoder_var = nn.Sequential(nn.Linear(512, 512))
        self.relu = nn.ReLU()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def latent_loss(self, mu, var):
        kl_loss = torch.mean(-0.5 * torch.sum(1 + var - mu ** 2 - var.exp(), dim=1))
        return kl_loss

    def forward(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()  # (b n t d) => (batch_size * num_sequences, sequence_length, feature_dim)
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        x = self.embedding(x)
        x = self.selfatt(x)

        if self.flag == "Train":

            N_x = x[:b * n // 2]  # Normal part
            A_x = x[b * n // 2:]  # Abnormal part

            A_att, A_aug = self.Amemory(A_x)  ### bt,btd,   anomaly video --->>>>> Anomaly memeory  at least 1 [1,0,0,...,1]
            N_Aatt, N_Aaug = self.Nmemory(A_x)  ### bt,btd,   anomaly video --->>>>> Normal memeory   at least 0 [0,1,1,...,1]

            A_Natt, A_Naug = self.Amemory(N_x)  ### bt,btd,   normal video --->>>>> Anomaly memeory   all 0 [0,0,0,0,0,...,0]
            N_att, N_aug = self.Nmemory(N_x)  ### bt,btd,   normal video --->>>>> Normal memeory    all 1 [1,1,1,1,1,...,1]

            # Xmemory
            X_Aatt, X_Aaug = self.Xmemory(A_x)
            X_Natt, X_Naug = self.Xmemory(N_x)

            # 异常样本中的 困难负样本
            _, A_index = torch.topk(A_att, t // 16 + 1, dim=-1)  # topk(t//16+1)：选择每个时间步前6.25%的注意力区域
            negative_ax = torch.gather(A_x, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)

            # 正常样本中的 可靠锚点  N_att: (b,t);  N_index: (b,k) k = t//16 + 1
            _, N_index = torch.topk(N_att, t // 16 + 1, dim=-1)
            anchor_nx = torch.gather(N_x, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)

            # 异常样本中的 相似正样本
            _, P_index = torch.topk(N_Aatt, t // 16 + 1, dim=-1)
            positivte_nx = torch.gather(A_x, 1, P_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)

            # 构造额外的负样本
            _, low_attention_indices = torch.topk(-X_Aatt, k=t // 16 + 1, dim=-1)  # 选择关注度最低的区域
            extra_negative = torch.gather(X_Aaug, 1, low_attention_indices.unsqueeze(2).expand([-1, -1, A_x.size(-1)]))
            extra_negative = extra_negative.mean(1).reshape(b // 2, n, -1).mean(1)

            # 计算四元组损失
            quadruplet_loss = self.quadruplet_loss(norm(anchor_nx), norm(positivte_nx), norm(negative_ax), norm(extra_negative))

            N_aug_mu = self.encoder_mu(N_aug)
            N_aug_var = self.encoder_var(N_aug)
            N_aug_new = self._reparameterize(N_aug_mu, N_aug_var)
            A_aug_new = self.encoder_mu(A_aug)

            anchor_nx_new = torch.gather(N_aug_new, 1, N_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)
            negative_ax_new = torch.gather(A_aug_new, 1, A_index.unsqueeze(2).expand([-1, -1, x.size(-1)])).mean(1).reshape(b // 2, n, -1).mean(1)

            kl_loss = self.latent_loss(N_aug_mu, N_aug_var)  # KL散度损失

            A_Naug = self.encoder_mu(A_Naug)
            N_Aaug = self.encoder_mu(N_Aaug)

            normal_fusion = N_aug_new + A_Naug + X_Naug
            abnormal_fusion = A_aug_new + N_Aaug + X_Aaug

            x = torch.cat((x, (torch.cat([normal_fusion, abnormal_fusion], dim=0))), dim=-1)

            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)  # 20250422

            # 当||anchor|| - ||negative|| < 100时不惩罚
            distance = torch.relu(100 - torch.norm(negative_ax_new, p=2, dim=-1) + torch.norm(anchor_nx_new, p=2, dim=-1)).mean()

            return {
                "frame": pre_att,
                'quad_margin': quadruplet_loss,
                'kl_loss': kl_loss,
                'distance': distance,
                'A_att': A_att.reshape((b // 2, n, -1)).mean(1),
                "N_att": N_att.reshape((b // 2, n, -1)).mean(1),
                "A_Natt": A_Natt.reshape((b // 2, n, -1)).mean(1),
                "N_Aatt": N_Aatt.reshape((b // 2, n, -1)).mean(1),
            }

        else:
            _, A_aug = self.Amemory(x)
            _, N_aug = self.Nmemory(x)
            _, X_aug = self.Xmemory(x)

            A_aug = self.encoder_mu(A_aug)
            N_aug = self.encoder_mu(N_aug)
            X_aug = self.encoder_mu(X_aug)

            fused_feat = N_aug + A_aug + X_aug

            x = torch.cat([x, fused_feat], dim=-1)

            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)  # 20250422

            feat = x.mean(dim=1)  # 便于 t-SNE 可视化

            return {"frame": pre_att, "feat": feat}   # 202505
            # return {"frame": pre_att}
