import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
from src.model import TMwVAD
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from src.dataloader import SDN
from src.utils import setup_seed
from sklearn.manifold import TSNE
from src.make_config import make_config
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

        # anomaly videos(含padding): 34032
        # normal videos(含padding): 17696
        # total: 51728
        # to calc AUC on anomaly video data
        ano_gt = frame_gt[:34032]
        ano_pred = frame_predict[:34032]
        fpr_ano, tpr_ano, _ = roc_curve(ano_gt, ano_pred)
        ano_auc = auc(fpr_ano, tpr_ano)  # AUC value in anomaly visualization

        # to calc false alarm rate, that is the rate of false prediction in normal video data
        nor_gt = frame_gt[34032:]
        nor_pred = frame_predict[34032:]
        false_alarm_rate = np.sum(nor_pred > 0.5) / len(nor_gt)  # false alarm rate

        return auc_score, ano_auc, false_alarm_rate



if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TMwVAD(input_size=1024, flag='test', a_nums=60, n_nums=60, x_nums=20).to(device)
    config = make_config('./config_yaml/config_sdn.yaml')
    setup_seed(config.Model.seed)
    ckpt_file = './ckpt/sdn_best.pth'
    loader_test = DataLoader(
        dataset=SDN(mode='Test', num_segments=config.Test.num_segments),
        batch_size=config.Test.batch_size,
        shuffle=False,
        num_workers=config.Test.num_workers,
    )

