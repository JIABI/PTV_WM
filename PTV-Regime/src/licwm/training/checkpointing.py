import os
import torch

def save_checkpoint(path: str, model, optimizer, epoch: int, metrics: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics}, path)
