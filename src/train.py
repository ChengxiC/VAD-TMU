import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F


def center_loss(instances_logits, bag_labels, device):
    losses = []
    for i, lbl in enumerate(bag_labels):
        if lbl == 0:  # normal bag
            v = instances_logits[i].view(-1)
            losses.append(v.var().unsqueeze(0))
    if len(losses) == 0:
        return torch.tensor(0., device=device)
    return torch.cat(losses).mean()


def norm(data):
    l2 = torch.norm(data, p=2, dim=-1, keepdim=True)
    return torch.div(data, l2)


class AD_Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, result, _label):
        loss = {}
        _label = _label.float()
        quad = result["quad_margin"]
        att = result['frame']
        A_att = result["A_att"]
        N_att = result["N_att"]
        A_Natt = result["A_Natt"]
        N_Aatt = result["N_Aatt"]
        kl_loss = result["kl_loss"]
        distance = result["distance"]

        b = _label.size(0)//2
        t = att.size(1)      
        anomaly = torch.topk(att, t//16 + 1, dim=-1)[0].mean(-1)
        anomaly_loss = self.bce(anomaly, _label)

        panomaly = torch.topk(1 - N_Aatt, t//16 + 1, dim=-1)[0].mean(-1)
        panomaly_loss = self.bce(panomaly, torch.ones((b)).cuda())

        A_att = torch.topk(A_att, t//16 + 1, dim=-1)[0].mean(-1)
        A_loss = self.bce(A_att, torch.ones((b)).cuda())

        N_loss = self.bce(N_att, torch.ones_like((N_att)).cuda())
        A_Nloss = self.bce(A_Natt, torch.zeros_like((A_Natt)).cuda())

        cost = anomaly_loss + 0.1 * (A_loss + panomaly_loss + N_loss + A_Nloss) + 0.1 * quad + 0.001 * kl_loss + 0.0001 * distance

        loss['total_loss'] = cost
        loss['att_loss'] = anomaly_loss
        loss['N_Aatt'] = panomaly_loss
        loss['A_loss'] = A_loss
        loss['N_loss'] = N_loss
        loss['A_Nloss'] = A_Nloss
        loss["quadruplet"] = quad
        loss['kl_loss'] = kl_loss
        return cost, loss


def train_one_epoch(ith_epoch, model, normal_loader, abnormal_loader, optimizer, criterion, device):

    model.train()

    model.flag = "Train"
    model.epoch = ith_epoch
    abnormal_iter = iter(abnormal_loader)
    t_cost = 0.0

    for idx, (ninput, nlabel) in enumerate(tqdm(normal_loader, desc=f"Training Epoch {ith_epoch}")):
        try:
            ainput, alabel = next(abnormal_iter)
        except StopIteration:
            abnormal_iter = iter(abnormal_loader)
            ainput, alabel = next(abnormal_iter)
        # print('nlabel:', nlabel)
        # print('alabel:', alabel)
        # 将正常和异常样本在 batch 维度上拼接
        _data = torch.cat([ninput, ainput], 0)
        _label = torch.cat([nlabel, alabel], 0)

        _data = _data.to(device)
        _label = _label.to(device)

        predict = model(_data)
        cost, loss = criterion(predict, _label)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        t_cost += cost.item()
    return t_cost/(idx+1)






