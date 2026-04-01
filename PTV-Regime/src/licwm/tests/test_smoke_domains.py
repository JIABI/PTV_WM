from omegaconf import OmegaConf
from licwm.training.engine import run_training

def test_smoke_train(tmp_path):
    cfg = OmegaConf.load('configs/config.yaml')
    cfg.output_dir = str(tmp_path / 'run')
    cfg.trainer.epochs = 1
    cfg.trainer.batch_size = 2
    cfg.trainer.history_len = 6
    cfg.trainer.pred_len = 3
    run_training(cfg)
