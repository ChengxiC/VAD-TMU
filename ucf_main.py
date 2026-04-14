import torch
import time
from src.logger import get_logger
from torchinfo import summary
from matplotlib import pyplot as plt
from src.train import *
from ucf_test import test
from src.model import TMwVAD
from src.dataloader import *
from src.utils import setup_seed
from src.make_config import make_config


def generate_data(config):
    loader_n_train = DataLoader(
        dataset=UCFCrime(mode='Train', num_segments=config.Train.num_segments, is_normal=True),
        batch_size=config.Train.batch_size,
        shuffle=True,
        num_workers=config.Train.num_workers,
        drop_last=True,
    )
    loader_a_train = DataLoader(
        dataset=UCFCrime(mode='Train', num_segments=config.Train.num_segments, is_normal=False),
        batch_size=config.Train.batch_size,
        shuffle=True,
        num_workers=config.Train.num_workers,
        drop_last=True,
    )

    loader_test = DataLoader(
        dataset=UCFCrime(mode='Test', num_segments=config.Test.num_segments),
        batch_size=config.Test.batch_size,
        shuffle=False,
        num_workers=config.Test.num_workers,
    )

    return loader_n_train, loader_a_train, loader_test

if __name__ == "__main__":
    torch.cuda.empty_cache()
    config = make_config('./config_yaml/config_ucf.yaml')
    setup_seed(config.Model.seed)
    device = torch.device('cuda' if config.Model.use_gpu else 'cpu')
    epochs = config.Train.epochs
    len_feature = 1024

    model = TMwVAD(len_feature, flag="Train", a_nums=config.Model.a_nums, n_nums=config.Model.n_nums, x_nums=config.Model.x_nums).to(device)

    logger = get_logger(filename=config.Model.logger_path)
    if not os.path.exists(config.Model.ckpt_path):
        os.makedirs(config.Model.ckpt_path)

    gt_path = config.Model.gt_path
    logger.info(f'model:\n{summary(model)}')

    logger.info(f'a_nums = {config.Model.a_nums}, n_nums = {config.Model.n_nums}, x_nums = {config.Model.x_nums}')
    best_auc = 0.0
    _ano_auc = 0.0
    _far = 0.0
    best_predicts = None
    criterion = AD_Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.Model.lr, betas=(0.9, 0.999), weight_decay=0.00005)
    normal_train_loader, abnormal_train_loader, test_loader = generate_data(config)
    best_ckpt = os.path.join(config.Model.ckpt_path, "ucf_best.pth")

    tot_train_time = 0.0
    tot_test_time = 0.0
    for epoch in range(epochs):
        t0 = time.time()
        loss = train_one_epoch(epoch+1, model, normal_train_loader, abnormal_train_loader, optimizer, criterion, device)
        t1 = time.time()
        epoch_train_time = t1 - t0
        tot_train_time += epoch_train_time

        t2 = time.time()
        auc, ano_auc, far = test(model, test_loader, device, gt_path)
        t3 = time.time()
        epoch_test_time = t3 - t2
        tot_test_time += epoch_test_time

        if best_auc < auc:
            best_auc = auc
            _ano_auc = ano_auc
            _far = far
            torch.save(model.state_dict(), best_ckpt)
        logger.info(f'epoch: {epoch+1}|{epochs}; loss: {loss:.5f}; total auc: {auc:.5f}; anomaly auc: {_ano_auc:.5f}; _false alarm rate: {far:.5f}')

    avg_test_time = tot_test_time / epochs
    logger.info(f'*******************************best auc: {best_auc:.5f}; anomaly auc: {_ano_auc: .5f}; _false alarm rate: {_far:.5f}***********************************')
    logger.info(f"Total Training Time: {tot_train_time:.2f}s over {epochs} epochs")
    logger.info(f"Test Time  : {avg_test_time:.2f}s")

