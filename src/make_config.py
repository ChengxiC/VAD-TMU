from dataclasses import dataclass
import yaml


@dataclass
class Train:
    batch_size: int
    num_workers: int
    epochs: int
    num_segments: int


@dataclass
class Test:
    batch_size: int
    num_workers: int
    num_segments: int


@dataclass
class Model:
    type: str
    ckpt_path: str
    logger_path: str
    gt_path: str
    anno_path: str
    lr: float
    seed: int
    use_gpu: bool
    a_nums: int
    n_nums: int
    x_nums: int


@dataclass
class Config:
    Train: Train
    Test: Test
    Model: Model


def make_config(config_path):
    with open(config_path, 'rb') as file:
        config_data = yaml.safe_load(file)

    return Config(
        Train=Train(**config_data['Train']),
        Test=Test(**config_data['Test']),
        Model=Model(**config_data['Model'])
    )



