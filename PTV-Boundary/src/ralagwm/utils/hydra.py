from omegaconf import OmegaConf

def cfg_to_dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True)
