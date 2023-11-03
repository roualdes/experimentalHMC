import pytest

import experimentalhmc as ehmc

import numpy as np

def test_normal_invcdf():

    assert np.isclose(ehmc.normal_invcdf(1e-20), -9.262340089798408)
    assert np.isclose(ehmc.normal_invcdf(0.001), -3.090232306167814)
    assert np.isclose(ehmc.normal_invcdf(0.25), -0.6744897501960817)
    assert np.isclose(ehmc.normal_invcdf(0.5), 0.0)
    assert np.isclose(ehmc.normal_invcdf(0.75), 0.6744897501960817)
    assert np.isclose(ehmc.normal_invcdf(0.999), 3.090232306167814)
    assert np.isclose(ehmc.normal_invcdf(1 - 1e-16), 8.209536151601386)

    with pytest.raises(Exception):
        ehmc.normal_invcdf(1)

    with pytest.raises(Exception):
        ehmc.normal_invcdf(0)


def test_normal_invcdf_broadcast():

    p = np.arange(0.1, 1, step = 0.05)
    z = np.array([-1.2815515655446008, -1.0364333894937894,
                  -0.8416212335729142, -0.6744897501960817, -0.5244005127080407,
                  -0.3853204664075676, -0.2533471031357998, -0.1256613468550740,
                  0.0, 0.1256613468550741, 0.2533471031357998, 0.3853204664075676,
                  0.5244005127080407, 0.6744897501960817, 0.8416212335729144,
                  1.0364333894937894, 1.2815515655446008, 1.644853626951471])
    assert np.allclose(ehmc.normal_invcdf(p), z)
