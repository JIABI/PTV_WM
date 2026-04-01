from omegaconf import OmegaConf
from licwm.audits.capacity_sweep import run_capacity_sweep

def test_capacity_keys(tmp_path):
    cfg = OmegaConf.load('configs/config.yaml')
    cfg.output_dir = str(tmp_path / 'run')
    cfg.trainer.epochs = 1
    cfg.trainer.batch_size = 2
    cfg.trainer.history_len = 6
    cfg.trainer.pred_len = 3
    out = run_capacity_sweep(cfg)
    assert set(out.keys()) == {"small", "medium", "large"}
