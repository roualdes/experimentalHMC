import ctypes

from pathlib import Path

import numpy.ctypeslib as npc
import bridgestan as bs
import numpy as np
import pytest

import experimentalhmc as ehmc

STAN_FOLDER = Path(__file__).parent.parent / "test_models"

bs.set_bridgestan_path(str(Path.home() / "bridgestan"))

def bridgestan_log_density_gradient(bsm):
    def bsm_ldg(position, gradient):
        ld = -np.inf
        try:
            ld, _ = bsm.log_density_gradient(position, out = gradient)
        except:
            pass
        return ld
    return bsm_ldg

def test_gaussian():
    model = "gaussian"

    stan_model = f"{str(STAN_FOLDER)}/{model}/{model}.stan"
    stan_data = f"{str(STAN_FOLDER)}/{model}/{model}.data.json"

    bsm = bs.StanModel(stan_model, data = stan_data)
    ldg = bridgestan_log_density_gradient(bsm)
    dims = bsm.param_unc_num()
    stan = ehmc.Stan(dims, ldg, warmup = 5_000)

    omv = ehmc.OnlineMeanVar(dims)

    for m in range(stan.warmup() + 5_000):
        x = stan.sample()
        if m > stan.warmup():
            omv.update(bsm.param_constrain(x))

    assert np.allclose(np.round(omv.location(), 2),
                       np.array([1.01, 0.42]),
                       atol = 1e-2)

    assert np.allclose(np.round(omv.scale(), 2),
                       np.array([0.06, 0.04]),
                       atol = 1e-2)

# def test_sir():
#     model = "sir"

#     stan_model = f"{str(STAN_FOLDER)}/{model}/{model}.stan"
#     stan_data = f"{str(STAN_FOLDER)}/{model}/{model}.data.json"

#     bsm = bs.StanModel(stan_model, data = stan_data)
#     ldg = bridgestan_log_density_gradient(bsm)
#     dims = bsm.param_unc_num()
#     stan = ehmc.Stan(dims, ldg, warmup = 5_000)

#     omv = ehmc.OnlineMeanVar(dims)

#     for m in range(stan.warmup() + 5_000):
#         x = stan.sample()
#         if m > stan.warmup():
#             omv.update(bsm.param_constrain(x))

#     assert np.allclose(np.round(omv.location(), 2),
#                        np.array([1.01, 0.2, 10.54, 0.36]), atol = 1e-2)
#     assert np.allclose(np.round(omv.scale(), 2),
#                        np.array([0.04, 0.01, 0.71, 0.03]), atol = 1e-2)
