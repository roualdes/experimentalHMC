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
