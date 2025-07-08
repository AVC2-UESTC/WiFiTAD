import torch


def get_device() -> torch.device:
    """
    Pick cuda if available, else MPS (Apple Metal), else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
