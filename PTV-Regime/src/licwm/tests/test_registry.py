from licwm.registry import Registry

def test_registry_register_build():
    r = Registry(); r.register('k', lambda x: x + 1)
    assert r.build('k', 2) == 3
