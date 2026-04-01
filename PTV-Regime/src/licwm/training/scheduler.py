import torch

def build_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.trainer.epochs // 3), gamma=cfg.trainer.lr_gamma)
