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

# f = lib.f
# f.restype = ctypes.c_void_p
# f.argtypes = [double_array, ctypes.c_int, double_array]

_normal_invcdf = lib.normal_invcdf
_normal_invcdf.restype = ctypes.c_int
_normal_invcdf.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double)]

_normal_invcdf_broadcast = lib.normal_invcdf_broadcast
_normal_invcdf_broadcast.restype = ctypes.c_int
_normal_invcdf_broadcast.argtypes = [double_array, ctypes.c_int, double_array]

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


_set_seed = lib.xoshiro_set_seed
_set_seed.restype = ctypes.c_void_p
_set_seed.argtypes = [ctypes.POINTER(ctypes.c_uint64), uint64_array]


_rand = lib.xoshiro_rand_u
_rand.restype = ctypes.c_double
_rand.argtypes = [uint64_array]

_rand_broadcast = lib.xoshiro_rand_u_broadcast
_rand_broadcast.restype = ctypes.c_void_p
_rand_broadcast.argtypes = [uint64_array, ctypes.c_int, double_array]


class UniformRNG():
    def __init__(self, seed: int):
        self._seed = ctypes.c_uint64(seed)
        self._xoshiro_seed = np.empty(4, dtype = np.uint64)
        _set_seed(self._seed, self._xoshiro_seed)

    def rand(self, N: int = None):
        if N is not None:
            assert type(N) == int
            x = np.empty(N)
            _rand_broadcast(self._xoshiro_seed, N, x)
            return x
        else:
            return _rand(self._xoshiro_seed)


_randn = lib.xoshiro_rand_n
_randn.restype = ctypes.c_double
_randn.argtypes = [uint64_array]

_randn_broadcast = lib.xoshiro_rand_n_broadcast
_randn_broadcast.restype = ctypes.c_void_p
_randn_broadcast.argtypes = [uint64_array, ctypes.c_int, double_array]


class NormalRNG():
    def __init__(self, seed: int):
        self._seed = ctypes.c_uint64(seed)
        self._xoshiro_seed = np.empty(4, dtype = np.uint64)
        _set_seed(self._seed, self._xoshiro_seed)

    def rand(self, N: int = None):
        if N is not None:
            assert type(N) == int
            z = np.empty(N)
            _randn_broadcast(self._xoshiro_seed, N, z)
            return z
        else:
            return _randn(self._xoshiro_seed)
