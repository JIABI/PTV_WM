import os
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import hydra
from omegaconf import DictConfig
from licwm.audits.antisteg import run_antisteg_audit
from licwm.audits.capacity_sweep import run_capacity_sweep
from licwm.audits.no_event import run_no_event_audit
from licwm.audits.bounded_residual import run_bounded_residual_audit
from licwm.audits.leakage import run_leakage_audit
from licwm.audits.ood_stress import run_ood_stress
from licwm.audits.oracle_runs import run_oracle_audit

os.chdir(PROJECT_ROOT)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    task = cfg.task.name
    if task == "antisteg_audit":
        run_antisteg_audit(cfg)
    elif task == "capacity_audit":
        if getattr(cfg.ablation, 'name', 'full') == 'bounded_residual':
            run_bounded_residual_audit(cfg)
        elif getattr(cfg.ablation, 'name', 'full') in {'gt_climate_oracle', 'law_param_oracle'}:
            run_oracle_audit(cfg)
        else:
            run_capacity_sweep(cfg)
    elif task == "no_event":
        run_no_event_audit(cfg)
    elif task == "ood_stress":
        run_ood_stress(cfg)
    elif task == "leakage":
        run_leakage_audit(cfg)
    else:
        run_antisteg_audit(cfg)

if __name__ == "__main__":
    main()
