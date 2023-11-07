import ctypes

from pathlib import Path

import numpy.ctypeslib as npc
import bridgestan as bs
import numpy as np
import pytest

import experimentalhmc as ehmc

STAN_FOLDER = Path(__file__).parent.parent / "test_models"

def test_ldg():
    dims = 2
    model = "gaussian"

    stan_model = f"{str(STAN_FOLDER)}/{model}/{model}.stan"
    stan_data = f"{str(STAN_FOLDER)}/{model}/{model}.data.json"

    bs.set_bridgestan_path("/Users/edward/bridgestan/")

    bsm = bs.StanModel.from_stan_file(stan_file = stan_model, model_data = stan_data)

    def log_density_gradient(position, gradient):
        ld, _ = bsm.log_density_gradient(npc.as_array(position, shape = (bsm.param_unc_num(),)),
                                         out = npc.as_array(gradient, shape = (bsm.param_unc_num(),)))
        return ld

    stan = ehmc.Stan(bsm.param_unc_num(), log_density_gradient, seed = 204)
