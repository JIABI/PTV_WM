import torch

def build_optimizer(cfg, model):
    return torch.optim.Adam(model.parameters(), lr=cfg.trainer.lr, weight_decay=cfg.trainer.weight_decay)
