import ctypes
from pathlib import Path
import numpy as np
from numpy.ctypeslib import ndpointer
double_array = ndpointer(dtype=ctypes.c_double,
                         flags=("C_CONTIGUOUS"))

dir = Path(__file__).parent.absolute().resolve()
lib = ctypes.CDLL(str(dir / "libmain.so"))

f = lib.f
f.restype = ctypes.c_void_p
f.argtypes = [double_array, ctypes.c_int, double_array]


_inv_std_normal = lib.inv_std_normal
_inv_std_normal.restype = ctypes.c_int
_inv_std_normal.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double)]

def inv_std_normal(p):
    z = ctypes.pointer(ctypes.c_double())
    err = _inv_std_normal(p, z)
    if err == -1:
        return np.nan
    return z.contents.value
