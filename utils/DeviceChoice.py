import torch

# Method to get the device (cuda if available, else cpu)
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

