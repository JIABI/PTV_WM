from pathlib import Path

from atlas_one_step.cli import smoke_test


def test_smoke_pipeline(tmp_path: Path):
    cfg = tmp_path / 'smoke.yaml'
    cfg.write_text(
        """
seed: 3
output_root: TMP_OUT

dataset:
  name: synthetic
  root: data
  channels: 3
  image_size: 16
  num_samples: 64
  batch_size: 8

corruption:
  name: diffusion_like
  t_min: 0.05
  t_max: 0.95

model:
  name: tiny_unet
  base_channels: 16
  time_dim: 16

phi_map:
  type: shallow
  hidden_channels: 8

loss:
  pred_weight: 1.0
  sem_weight: 0.5
  stab_weight: 1.0e-4

train:
  lr: 1.0e-3
  max_steps: 6
  eval_every: 3
  grad_clip: 1.0

sweep:
  family: line_x0_u
  num_points: 2
  max_steps_per_target: 3
  seeds: [1]

selection:
  semantic_gap_weight: 0.5
  complexity_weight: 0.05
  candidate_family: scheduled
  schedule_basis_order: 2

eval:
  collapse_threshold: 0.35
  tail_percentiles: [95, 99]
""".replace('TMP_OUT', str(tmp_path / 'outputs')),
        encoding='utf-8',
    )
    smoke_test(str(cfg))
    assert (tmp_path / 'outputs' / 'atlas' / 'atlas.parquet').exists()
