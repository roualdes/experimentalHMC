import ctypes

from pathlib import Path

import numpy.ctypeslib as npc
import bridgestan as bs
import numpy as np
import pytest

import experimentalhmc as ehmc

STAN_FOLDER = Path(__file__).parent.parent / "test_models"

bs.set_bridgestan_path(str(Path.home() / "bridgestan"))

def bridgestan_log_density_gradient_c_wrapper(bsm):
    dim = bsm.param_unc_num()
    def bsm_c_wrapper(position, gradient):
        ld = bsm.ldg_ctypes(position, gradient)
        return ld
    return bsm_c_wrapper

# def bridgestan_log_density_gradient_c_wrapper(bsm):
#     dim = bsm.param_unc_num()
#     def bsm_c_wrapper(position, gradient):
#         ld = ctypes.pointer(ctypes.c_double())
#         err = ctypes.pointer(ctypes.c_char_p())
#         bsm._ldg(bsm.model, int(True), int(True), position, ld, gradient, err)
#         return ld.contents.value
#     return bsm_c_wrapper

def test_ldg():
    model = "gaussian"

    stan_model = f"{str(STAN_FOLDER)}/{model}/{model}.stan"
    stan_data = f"{str(STAN_FOLDER)}/{model}/{model}.data.json"

    bsm = bs.StanModel.from_stan_file(stan_file = stan_model, model_data = stan_data)
    ldg = bridgestan_log_density_gradient_c_wrapper(bsm)
    dims = bsm.param_unc_num()
    stan = ehmc.Stan(dims, ldg, warmup = 5_000)

    omv = ehmc.OnlineMeanVar(dims)

    for m in range(stan.warmup() + 5_000):
        x = stan.sample()
        if m > stan.warmup():
            omv.update(bsm.param_constrain(x))

    assert np.allclose(np.round(omv.location(), 2), np.array([1.01, 0.42]), atol = 1e-2)
    assert np.allclose(np.round(omv.scale(), 2), np.array([0.06, 0.04]), atol = 1e-2)
