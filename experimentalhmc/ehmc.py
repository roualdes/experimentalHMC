from numpy.ctypeslib import ndpointer
from pathlib import Path

import ctypes

double_array = ndpointer(dtype=ctypes.c_double,
                         flags=("C_CONTIGUOUS", "WRITEABLE"))

uint64_array = ndpointer(dtype=ctypes.c_uint64,
                         flags=("C_CONTIGUOUS", "WRITEABLE"))

ehmc_folder = Path(__file__).parent.absolute().resolve()
lib = ctypes.CDLL(str(ehmc_folder / "libehmc.so"))

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


rand_normal = lib.normal_rand
rand_normal.restype = ctypes.c_double
rand_normal.argtypes = [uint64_array]

rand_normal_broadcast = lib.normal_rand_broadcast
rand_normal_broadcast.restype = ctypes.c_void_p
rand_normal_broadcast.argtypes = [uint64_array, ctypes.c_int, double_array]


_stan_transition = lib.stan_transition
_stan_transition.restype = ctypes.c_void_p
_stan_transition.argtypes = [double_array,                     # draws
                             ctypes.CFUNCTYPE(ctypes.c_double, # ldg
                                              ctypes.POINTER(ctypes.c_double),
                                              ctypes.POINTER(ctypes.c_double)),
                             uint64_array, # rngs
                             ctypes.POINTER(ctypes.c_double), # adapt_stat
                             ctypes.POINTER(ctypes.c_bool),   # divergent
                             ctypes.POINTER(ctypes.c_int),    # n_leapfrog
                             ctypes.POINTER(ctypes.c_int),    # tree_depth
                             ctypes.POINTER(ctypes.c_double), # energy
                             ctypes.c_int,                    # dims
                             double_array,                    # metric
                             ctypes.c_double, # step_size
                             ctypes.c_double, # max_delta_H
                             ctypes.c_int]    # max_tree_depth
