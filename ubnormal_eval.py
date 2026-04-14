from src.dataloader import *
from src.make_config import make_config
from src.model import TMwVAD
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from torch.utils.data import DataLoader, Dataset
from src.utils import *


def valid(model, test_loader, model_file=None):
    with torch.no_grad():
        model.eval()
        model.flag = "Test"
        if model_file is not None:
            model.load_state_dict(torch.load(model_file, map_location='cuda:0'))

        load_iter = iter(test_loader)
        frame_gt = np.load("frame_gt/gt_ubnormal.npy")
        frame_predict = None
        cls_label = []
        cls_pre = []
        temp_predict = torch.zeros(0).cuda()
        count = 0
        for i in range(len(test_loader.dataset)):

            _data, _label, _name = next(load_iter)
            _name = _name[0]
            _data = _data.cuda()
            _label = _label.cuda()

            res = model(_data)
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)
            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()
                pl = len(a_predict) * 16

                count = count + pl
                cls_pre.append(1 if a_predict.max() > 0.5 else 0)
                fpre_ = np.repeat(a_predict, 16)
                if frame_predict is None:
                    frame_predict = fpre_
                else:
                    frame_predict = np.concatenate([frame_predict, fpre_])
                temp_predict = torch.zeros(0).cuda()

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)
        print(f'auc score: {auc_score}')

        # anomaly visualization(including padding): 70896
        # normal visualization(including padding): 24224
        # total: 95120
        ano_gt = frame_gt[:70896]
        ano_pred = frame_predict[:70896]
        fpr_ano, tpr_ano, _ = roc_curve(ano_gt, ano_pred)
        ano_auc = auc(fpr_ano, tpr_ano)
        print(f'anomaly score: {ano_auc}')


if __name__ == "__main__":
    torch.cuda.empty_cache()
    config = make_config('./config_yaml/config_ubnormal.yaml')
    setup_seed(config.Model.seed)
    device = torch.device('cuda' if config.Model.use_gpu else 'cpu')
    len_feature = 1024
    model = TMwVAD(len_feature, flag="Train", a_nums=config.Model.a_nums, n_nums=config.Model.n_nums, x_nums=config.Model.x_nums).to(device)

    loader_test = DataLoader(
        dataset=UB(mode='Test', num_segments=config.Test.num_segments),
        batch_size=config.Test.batch_size,
        shuffle=False,
        num_workers=config.Test.num_workers,
    )

    valid(model, loader_test, model_file=os.path.join(config.Model.ckpt_path, "ubnormal_best.pth"))

