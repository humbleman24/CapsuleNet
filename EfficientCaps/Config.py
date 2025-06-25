from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Mnist_Train_Loader_Cfg:
    mnist_path: str = "/root/info/Capsule/data/MNIST/raw/"
    train: bool = True
    batch_size: int = 128
    num_workers: int = 4
    data_mean: Optional[List[float]] = None
    data_std: Optional[List[float]] = None


@dataclass
class Mnist_Test_Loader_Cfg:
    mnist_path: str = "/root/info/Capsule/data/MNIST/raw/"
    train: bool = False
    batch_size: int = 128
    num_workers: int = 4
    data_mean: Optional[List[float]] = None
    data_std: Optional[List[float]] = None


@dataclass
class Mnist_Training_Cfg:
    start_epoch: int = 0
    num_epochs: int = 50
    lr: float = 1e-3
    output_dir: str = "path/to/outputs/"
    save_model_epochs: int = 20
    resume: Optional[str] = None
    beta1: float = 0.9
    beta2: float = 0.999
    step_size: int = 10
    gamma: float = 0.1
