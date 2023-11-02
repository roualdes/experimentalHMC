import pytest

import experimentalhmc as ehmc

import numpy as np

def test_inv_std_normal():

    assert np.isclose(ehmc.inv_std_normal(0.25), -0.6744897501960817)
    assert np.isclose(ehmc.inv_std_normal(0.001), -3.090232306167814)
    assert np.isclose(ehmc.inv_std_normal(1e-20), -9.262340089798408)
