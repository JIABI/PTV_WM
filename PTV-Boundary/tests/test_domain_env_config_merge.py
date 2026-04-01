import torch

from ralagwm.models.ralag_wm import RALAGWM
from ralagwm.training.paper_pipeline import build_domain_cfg, load_training_presets
from ralagwm.utils.io import load_yaml


def _load_base_cfg():
    return {
        'env': load_yaml('configs/env/dummy.yaml'),
        'model': load_yaml('configs/model/ralag_wm.yaml'),
        'deploy': {'kind': 'linear'},
        'audit': {'num_audits': 3, 'trim_ratio': 0.1},
        'chart': {'chart_budget': 8, 'pool_budget': 16},
        'trainer': {'batch_size': 8},
        'runtime': {'max_steps': 1},
        'device': 'cpu',
        'seed': 7,
    }


def test_dmc_proprio_domain_loads_named_env_defaults_instead_of_dummy_values():
    base_cfg = _load_base_cfg()
    presets = load_training_presets()
    domain = {
        'name': 'dmc_proprio',
        'label': 'DMC-Proprio-Cartpole',
        'domain_name': 'cartpole',
        'task_name': 'swingup',
        'group': 'nonimage_continuous',
    }

    cfg = build_domain_cfg(base_cfg, domain, presets)

    assert cfg['env']['name'] == 'dmc_proprio'
    assert cfg['env']['obs_type'] == 'proprio'
    assert cfg['env']['action_type'] == 'continuous'
    assert int(cfg['env']['obs_dim']) == 5
    assert int(cfg['env']['action_dim']) == 1
    assert bool(cfg['env']['from_pixels']) is False

    model = RALAGWM(obs_type='proprio', obs_dim=int(cfg['env']['obs_dim']), action_dim=int(cfg['env']['action_dim']))
    first = model.encoder.net[0]
    assert isinstance(first, torch.nn.Linear)
    assert first.in_features == 5


def test_atari_domain_loads_named_env_defaults_instead_of_dummy_values():
    base_cfg = _load_base_cfg()
    presets = load_training_presets()
    domain = {
        'name': 'atari100k',
        'label': 'Atari-Pong',
        'env_id': 'ALE/Pong-v5',
        'group': 'visual_discrete',
    }

    cfg = build_domain_cfg(base_cfg, domain, presets)

    assert cfg['env']['name'] == 'atari100k'
    assert cfg['env']['obs_type'] == 'image'
    assert cfg['env']['action_type'] == 'discrete'
    assert int(cfg['env']['obs_dim']) == 84
    assert int(cfg['env']['frame_stack']) == 4
    assert int(cfg['env']['action_dim']) == 6
