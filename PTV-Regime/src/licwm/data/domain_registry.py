"""Domain dataset factory."""
from .base import TinySyntheticFallbackDataset
from .lic_boids.dataset import LICBoidsDataset
from .crowd.eth_ucy import ETHUCYDataset
from .crowd.sdd import SDDDataset
from .interaction.dataset import InteractionDataset
from .lic_uav.dataset import LICUAVDataset

def build_domain_dataset(cfg, split: str):
    common = dict(seq_len=cfg.trainer.history_len, pred_len=cfg.trainer.pred_len)
    if cfg.domain.name == "lic_boids":
        return LICBoidsDataset(split=split, mode=cfg.domain.mode, n_samples=cfg.domain.n_samples, n_agents=cfg.domain.num_agents, event_dim=cfg.domain.event_dim, **common)
    if cfg.domain.name == "eth_ucy":
        return ETHUCYDataset(split=split, data_root=cfg.domain.data_root, use_velocity_proxy=cfg.domain.use_velocity_proxy, **common)
    if cfg.domain.name == "sdd":
        return SDDDataset(split=split, data_root=cfg.domain.data_root, use_velocity_proxy=cfg.domain.use_velocity_proxy, **common)
    if cfg.domain.name == "interaction":
        return InteractionDataset(split=split, data_root=cfg.domain.data_root, **common)
    if cfg.domain.name == "lic_uav":
        return LICUAVDataset(split=split, scenario=cfg.domain.scenario, n_samples=cfg.domain.n_samples, n_agents=cfg.domain.num_agents, **common)
    return TinySyntheticFallbackDataset(num_samples=64, n_agents=cfg.domain.num_agents, obs_dim=cfg.domain.obs_dim, act_dim=cfg.domain.action_dim, event_dim=cfg.domain.event_dim, **common)
