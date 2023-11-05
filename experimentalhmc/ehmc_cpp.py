import ctypes
from pathlib import Path
import numpy as np
from numpy.ctypeslib import ndpointer

double_array = ndpointer(dtype=ctypes.c_double,
                         flags=("C_CONTIGUOUS"))

uint64_array = ndpointer(dtype=ctypes.c_uint64,
                         flags=("C_CONTIGUOUS"))

dir = Path(__file__).parent.absolute().resolve()
lib = ctypes.CDLL(str(dir / "libehmc.so"))


_normal_invcdf = lib.normal_invcdf
_normal_invcdf.restype = ctypes.c_int
_normal_invcdf.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double)]

_normal_invcdf_broadcast = lib.normal_invcdf_broadcast
_normal_invcdf_broadcast.restype = ctypes.c_int
_normal_invcdf_broadcast.argtypes = [double_array, ctypes.c_int, double_array]


_set_seed = lib.xoshiro_set_seed
_set_seed.restype = ctypes.c_void_p
_set_seed.argtypes = [ctypes.POINTER(ctypes.c_uint64), uint64_array]


_uniform_rng = lib.uniform_rng
_uniform_rng.restype = ctypes.c_double
_uniform_rng.argtypes = [uint64_array]

_uniform_rng_broadcast = lib.uniform_rng_broadcast
_uniform_rng_broadcast.restype = ctypes.c_void_p
_uniform_rng_broadcast.argtypes = [uint64_array, ctypes.c_int, double_array]

def _rand_uniform(xoshiro_seed, N: int = None):
    if N is not None:
        assert type(N) == int
        u = np.empty(N)
        _uniform_rng_broadcast(xoshiro_seed, N, u)
        return u
    else:
        return _uniform_rng(xoshiro_seed)


_normal_rng = lib.normal_rng
_normal_rng.restype = ctypes.c_double
_normal_rng.argtypes = [uint64_array]

_normal_rng_broadcast = lib.normal_rng_broadcast
_normal_rng_broadcast.restype = ctypes.c_void_p
_normal_rng_broadcast.argtypes = [uint64_array, ctypes.c_int, double_array]

def _rand_normal(xoshiro_seed, N: int = None):
    if N is not None:
        assert type(N) == int
        z = np.empty(N)
        _normal_rng_broadcast(xoshiro_seed, N, z)
        return z
    else:
        return _normal_rng(xoshiro_seed)
