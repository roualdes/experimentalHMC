from .ehmc import uniform_rand

import numpy as np
import numpy.ctypeslib as npc

def generate_draw(seed, dims, radius):
    U = uniform_rand(seed, dims)
    return radius * (2 * U - 1)

def initialize_draws(seed, dims, ldg, initial_draw_radius = 2, initial_draw_attempts = 100):

    attempts = initial_draw_attempts
    attempt = 0

    initialized = False
    radius = initial_draw_radius
    gradient = np.empty(dims)

    while attempt < attempts and not initialized:
        initial_draw = generate_draw(seed, dims, radius)
        # TODO probably need some logic to call
        # ldg(initial_draw, gradient)
        # when appropriate
        ld = ldg(npc.as_ctypes(initial_draw), npc.as_ctypes(gradient))

        if np.isfinite(ld) and not np.isnan(ld):
            initialized = True

        g = np.sum(gradient)
        if not np.isfinite(g) or np.isnan(g):
            initialized = False
            continue

        attempt += 1

    if attempt > attempts:
        print(f"Failed to find initial values in {attempt} attempts")

    return initial_draw
