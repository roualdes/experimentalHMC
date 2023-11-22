from .ehmc import normal_rand, uniform_rand
from .leapfrog import leapfrog

import numpy as np
import numpy.ctypeslib as npc

def step_size_initializer(position, dims, step_size, metric, seed, ldg):

    q = np.copy(position)
    momentum = normal_rand(seed, dims)
    gradient = np.empty_like(momentum)

    ld = ldg(npc.as_ctypes(q), npc.as_ctypes(gradient))
    H0 = -ld + 0.5 * np.dot(momentum, momentum)

    ld = leapfrog(q, momentum, step_size * np.sqrt(metric), 1, gradient, ldg)
    H = -ld + 0.5 * np.dot(momentum, momentum)
    if np.isnan(H):
        H = np.finfo(np.float64).max

    dH = H0 - H
    direction = 1 if dH > np.log(0.8) else -1

    while True:
        momentum = normal_rand(seed, dims)
        q = np.copy(position)
        ld = ldg(npc.as_ctypes(q), npc.as_ctypes(gradient))
        H0 = -ld + 0.5 * np.dot(momentum, momentum)

        ld = leapfrog(q, momentum, step_size * np.sqrt(metric), 1, gradient, ldg)
        H = -ld + 0.5 * np.dot(momentum, momentum)
        if np.isnan(H):
            H = np.finfo(np.float64).max

        dH = H0 - H
        if np.isnan(dH):
            dH = np.finfo(np.float64).max

        if direction == 1 and not dH > np.log(0.8):
            break
        elif direction == -1 and not dH < np.log(0.8):
            break
        else:
            step_size = 2 * step_size if direction == 1 else 0.5 * step_size

        if step_size > 1e7:
            raise ValueError("Step size too large.  Posterior is improper.  Please check your model.")

        if step_size <= 0.0:
            raise ValueError("Step size too small.  No acceptable step size could be found.  Perhaps the posterior is not continuous.")

    return step_size
