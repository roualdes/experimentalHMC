from .ehmc_cpp import _normal_invcdf, _normal_invcdf_broadcast

import ctypes

import numpy as np

def normal_invcdf(p):
    assert np.all( (0 < p) & (p < 1) )
    if type(p) == float:
        z = ctypes.pointer(ctypes.c_double())
        err = _normal_invcdf(p, z)
        if err == -1:
            return np.nan
        return z.contents.value
    else:
        assert type(p) == np.ndarray
        p = np.ravel(p)
        z = np.empty_like(p)
        err = _normal_invcdf_broadcast(p, np.size(p), z)
        if err == -1:
            return np.full_like(p, np.nan)
        return z
