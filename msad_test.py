# Chengxi Chu, Universiti Malaya
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
from matplotlib import pyplot as plt
from src.model import TMwVAD
from src.make_config import make_config
from torch.utils.data import DataLoader, Dataset
from src.dataloader import MSAD
from src.utils import setup_seed
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")


def test(model, test_loader, device, gt_path, ckpt_file=None):
    with torch.no_grad():

        model.eval()
        model.flag = "Test"
        if ckpt_file is not None:
            model.load_state_dict(torch.load(ckpt_file))

        load_iter = iter(test_loader)
        frame_gt = np.load(gt_path)
        frame_predict = None

        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros(0).to(device)

        for i in range(len(test_loader.dataset)):

            _data, _label, _ = next(load_iter)

            _data = _data.to(device)
            _label = _label.to(device)

            res = model(_data)
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()

                cls_pre.append(1 if a_predict.max() > 0.5 else 0)
                fpre_ = np.repeat(a_predict, 16)
                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])
                temp_predict = torch.zeros(0).to(device)

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)

        # abnormal frames: 58192, you can use MSAD_cnt_a_frames.py to get the value.
        # to calc AUC on anomaly video data
        ano_gt = frame_gt[:58192]
        ano_pred = frame_predict[:58192]
        fpr_ano, tpr_ano, _ = roc_curve(ano_gt, ano_pred)
        ano_auc = auc(fpr_ano, tpr_ano)

        # to calc false alarm rate, that is the rate of false prediction in normal video data
        nor_gt = frame_gt[58192:]
        nor_pred = frame_predict[58192:]
        false_alarm_rate = np.sum(nor_pred > 0.5) / len(nor_gt)  # false alarm rate

        return auc_score, ano_auc, false_alarm_rate



