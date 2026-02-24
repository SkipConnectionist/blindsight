import platform
import torch

is_macos = platform.system() == 'Darwin'


def get_platform_default_device() -> str:
    if platform.system() == 'Darwin' and torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
