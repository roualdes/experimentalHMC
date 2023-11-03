import ctypes
from pathlib import Path
import numpy as np
from numpy.ctypeslib import ndpointer
double_array = ndpointer(dtype=ctypes.c_double,
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
    elif type(p) == np.ndarray:
        p = np.ravel(p)
        z = np.empty_like(p)
        err = _normal_invcdf_broadcast(p, np.size(p), z)
        if err == -1:
            return np.full_like(p, np.nan)
        return z
