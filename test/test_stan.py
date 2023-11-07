import ctypes

from pathlib import Path

import numpy.ctypeslib as npc
import bridgestan as bs
import numpy as np
import pytest

import experimentalhmc as ehmc

STAN_FOLDER = Path(__file__).parent.parent / "test_models"

bs.set_bridgestan_path("/Users/edward/bridgestan/") # TODO this shouldn't be hardcoded

def bridgestan_log_density_gradient_c_wrapper(bsm):
    dim = bsm.param_unc_num()
    def bsm_c_wrapper(position, gradient):
        ld, _ = bsm.log_density_gradient(npc.as_array(position, shape = (dim,)),
                                         out = npc.as_array(gradient, shape = (dim,)))
        return ld
    return bsm_c_wrapper

def test_ldg():
    model = "gaussian"

    stan_model = f"{str(STAN_FOLDER)}/{model}/{model}.stan"
    stan_data = f"{str(STAN_FOLDER)}/{model}/{model}.data.json"

    bsm = bs.StanModel.from_stan_file(stan_file = stan_model, model_data = stan_data)
    ldg = bridgestan_log_density_gradient_c_wrapper(bsm)
    dims = bsm.param_unc_num()
    stan = ehmc.Stan(dims, ldg, seed = 204)

    omv = ehmc.OnlineMeanVar(dims)

    for m in range(2000):
        print(f"iteration {m}...")
        x = stan.sample()
        print(f"sampled {x}")
        omv.update(x)
        print(f"running mean = {omv.mean()}")
        print(f"running var = {omv.var()}")
