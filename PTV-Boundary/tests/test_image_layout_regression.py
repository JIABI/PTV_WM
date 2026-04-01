import torch

from ralagwm.audit.heads import build_audit_head_for_obs
from ralagwm.models.encoders import ImageEncoder


def test_image_encoder_preserves_batch_bchw():
    enc = ImageEncoder(image_size=84, image_channels=3, hidden_dim=32)
    x = torch.randn(8, 3, 84, 84)
    y = enc(x)
    assert y.shape == (8, 32)


def test_image_encoder_accepts_single_stacked_thwc():
    enc = ImageEncoder(image_size=84, image_channels=12, hidden_dim=32)
    x = torch.randn(4, 84, 84, 3)
    y = enc(x)
    assert y.shape == (1, 32)


def test_audit_head_preserves_batch_bchw():
    head = build_audit_head_for_obs(obs_shape=(3, 84, 84), num_actions=5)
    x = torch.randn(8, 3, 84, 84)
    y = head(x)
    assert y.shape == (8, 5)
