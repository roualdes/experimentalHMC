import ctypes

import numpy as np
import numpy.typing as npt
import numpy.ctypeslib as npc

from inspect import isfunction
from typing import Union, Callable

FloatArray = npt.NDArray[np.float64]

def leapfrog(position: FloatArray,
             momentum: FloatArray,
             step_size: Union[float, FloatArray],
             steps: int,
             gradient: FloatArray,
             ldg: Callable[[FloatArray, FloatArray], float]):

    ldg_function = isfunction(ldg) # false if a CFUNCTYPE wrapped function, I think

    ld = 0
    momentum += 0.5 * step_size * gradient

    for step in range(steps):
        position += step_size * momentum
        if ldg_function:
            ld = ldg(position, gradient)
        else:
            ld = ldg(npc.as_ctypes(position), npc.as_ctypes(gradient))
        if step != (steps - 1):
            momentum += step_size * gradient

    momentum += 0.5 * step_size * gradient

    return ld
