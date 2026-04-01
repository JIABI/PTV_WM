import numpy as np


def test_numpy_bool8_alias_present_for_legacy_gym():
    assert hasattr(np, "bool8")
    assert np.bool8 is np.bool_
