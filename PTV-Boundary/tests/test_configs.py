from hydra import initialize, compose

def test_compose_base():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="base")
        assert cfg.env.name is not None
