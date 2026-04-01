
import copy
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from hydra.core.hydra_config import HydraConfig

from fatewm.runners.train_loop import train
from fatewm.algos import make_algo
from fatewm.core.risk_router import RiskRouter, RouterConfig
from fatewm.core.attention_gate import AttentionGate
from fatewm.envs import make_env
from fatewm.core.runlog import setup_train_log
from fatewm.core.utils import clone_model


def infer_specs(cfg):
    env = make_env(cfg.env)
    specs = (env.obs_shape, env.action_dim, env.is_discrete, getattr(env, "action_bounds", None))
    return specs


def build_components(cfg: DictConfig, device: torch.device):
    obs_shape, action_dim, is_discrete, action_bounds = infer_specs(cfg)
    latent_dim = int(cfg.model.latent_dim)
    hidden_dim = int(cfg.model.hidden_dim)

    algo = make_algo(cfg.algo, obs_shape, action_dim, is_discrete, latent_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(algo.parameters(), lr=float(cfg.train.lr))
    components = {
        "algo": algo,
        "optimizer": optimizer,
        "device": device,
        "action_dim": action_dim,
        "is_discrete": is_discrete,
        "action_bounds": action_bounds,
    }

    if cfg.method.name in ("rrrm", "fatewm"):
        time_emb_freq = int(getattr(cfg.method, "time_emb_freq", 8))
        rcfg = RouterConfig(
            z_feat_dim=4,
            time_emb_dim=2 * time_emb_freq,
            hidden=int(getattr(cfg.method, "router_hidden", 128)),
            n_experts=4,
            entropy_beta=float(getattr(cfg.method, "entropy_beta", 0.02)),
            budget_B=float(getattr(cfg.method, "B", 8.0)),
            dual_lr=float(getattr(cfg.method, "dual_lr", 1e-3)),
            selection_temperature=float(getattr(cfg.method, "selection_temperature", 0.5)),
            selection_kind=str(getattr(cfg.method, "selection_kind", "sparsemax")),
        )
        router = RiskRouter(rcfg).to(device)
        components["fate_estimator"] = router
        components["teacher_algo"] = clone_model(algo).to(device)
        components["optimizer"] = torch.optim.Adam(
            list(algo.parameters()) + list(router.parameters()),
            lr=float(cfg.train.lr),
        )
    elif cfg.method.name == "attention_only":
        gate = AttentionGate(in_dim=3, out_dim=len(cfg.timescales.deltas), hidden=64).to(device)
        components["att_gate"] = gate
        components["optimizer"] = torch.optim.Adam(
            list(algo.parameters()) + list(gate.parameters()),
            lr=float(cfg.train.lr),
        )
    return components


@hydra.main(version_base=None, config_path="../../../configs", config_name="default")
def main(cfg: DictConfig):
    try:
        out_dir = HydraConfig.get().runtime.output_dir
        setup_train_log(out_dir, "train.log")
    except Exception:
        pass
    print(OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device)
    train(cfg, build_components)


if __name__ == "__main__":
    main()
