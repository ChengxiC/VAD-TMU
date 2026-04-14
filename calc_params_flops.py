from src.logger import get_logger
from torchinfo import summary
from thop import profile
from src.train import *
from src.dataloader import *
from src.model import TMwVAD
from src.utils import setup_seed
from src.make_config import make_config


def generate_data(config):
    loader_n_train = DataLoader(
        dataset=UCFCrime(mode='Train', num_segments=config.Train.num_segments, is_normal=True),
        batch_size=8,
        shuffle=True,
        num_workers=config.Train.num_workers,
        drop_last=True,
    )
    loader_a_train = DataLoader(
        dataset=UCFCrime(mode='Train', num_segments=config.Train.num_segments, is_normal=False),
        batch_size=8,
        shuffle=True,
        num_workers=config.Train.num_workers,
        drop_last=True,
    )

    loader_test = DataLoader(
        dataset=UB(mode='Test', num_segments=config.Test.num_segments),
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
    config.len_feature = 1024

    model = TMwVAD(config.len_feature, flag="Train", a_nums=config.Model.a_nums, n_nums=config.Model.n_nums, x_nums=config.Model.x_nums).to(device)

    logger = get_logger(filename=config.Model.logger_path)
    if not os.path.exists(config.Model.ckpt_path):
        os.makedirs(config.Model.ckpt_path)
    gt_path = config.Model.gt_path

    print(f'model:\n{summary(model)}')

    # 用真实 的 Batch，不用虚拟的.
    train_loader_n, train_loader_a, test_loader = generate_data(config)
    dummy_batch, _ = next(iter(train_loader_n))
    dummy_input = dummy_batch.to(device)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    # thop 报的是 MACs (一次乘加算一次)，若要当作 FLOPs 则乘 2
    flops = macs * 2
    print(f"Params: {params/1e6:.3f} M; FLOPs: {flops/1e9:.3f} G")



