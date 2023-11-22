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


rand_uniform = lib.uniform_rand
rand_uniform.restype = ctypes.c_double
rand_uniform.argtypes = [uint64_array]

rand_uniform_broadcast = lib.uniform_rand_broadcast
rand_uniform_broadcast.restype = ctypes.c_void_p
rand_uniform_broadcast.argtypes = [uint64_array, ctypes.c_int, double_array]

def uniform_rand(xoshiro_seed, N: int = None):
    if N is not None:
        assert type(N) == int
        u = np.empty(N)
        rand_uniform_broadcast(xoshiro_seed, N, u)
        return u
    else:
        return rand_uniform(xoshiro_seed)


rand_normal = lib.normal_rand
rand_normal.restype = ctypes.c_double
rand_normal.argtypes = [uint64_array]

rand_normal_broadcast = lib.normal_rand_broadcast
rand_normal_broadcast.restype = ctypes.c_void_p
rand_normal_broadcast.argtypes = [uint64_array, ctypes.c_int, double_array]

def normal_rand(xoshiro_seed, N: int = None):
    if N is not None:
        assert type(N) == int
        z = np.empty(N)
        rand_normal_broadcast(xoshiro_seed, N, z)
        return z
    else:
        return rand_normal(xoshiro_seed)


_stan_transition = lib.stan_transition
_stan_transition.restype = ctypes.c_void_p
_stan_transition.argtypes = [double_array,
                             ctypes.CFUNCTYPE(ctypes.c_double,
                                              ctypes.POINTER(ctypes.c_double),
                                              ctypes.POINTER(ctypes.c_double)),
                             uint64_array,
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.POINTER(ctypes.c_bool),
                             ctypes.POINTER(ctypes.c_int),
                             ctypes.POINTER(ctypes.c_int),
                             ctypes.POINTER(ctypes.c_double),
                             ctypes.c_int,
                             double_array,
                             ctypes.c_double,
                             ctypes.c_double,
                             ctypes.c_int]
