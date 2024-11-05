import torch


def get_device(gpu_id: int = -1) -> str:
    """
    Get device string with support for auto-selection and specific GPU.
    
    Args:
        gpu_id (int): GPU index (-1 for auto-selection, >=0 for specific GPU)
    
    Returns:
        str: Device string ('cuda', 'cuda:n', 'mps', or 'cpu')
    """
    if torch.cuda.is_available() and gpu_id >= 0:
        return f"cuda:{gpu_id}"
    elif torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    print(get_device())
