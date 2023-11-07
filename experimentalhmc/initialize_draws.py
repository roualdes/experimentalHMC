from .ehmc_cpp import _rand_normal, _rand_uniform

import numpy as np
import numpy.ctypeslib as npc

def generate_draw(seed, dims, radius):
    U = _rand_uniform(seed, dims)
    return radius * (2 * U - 1)

def initialize_draws(seed, dims, ldg, initial_draw_radius = 2, initial_draw_attempts = 100):

    attempts = initial_draw_attempts
    attempt = 0
    initialized = False

    radius = initial_draw_radius
    initial_draw = generate_draw(seed, dims, radius)
    print(f"{type(initial_draw)}")

    gradient = np.empty_like(initial_draw)
    momentum = np.empty_like(initial_draw)

    while attempt < attempts and not initialized:
        ld = ldg(npc.as_ctypes(initial_draw), npc.as_ctypes(gradient))
        print("successfully called ldg()")
        print(f"attempt number {attempt}")
        if np.isfinite(ld) and ~np.isnan(ld):
            initialized = True

        g = np.sum(gradient)
        if ~np.isfinite(g) or np.isnan(g):
            initialized = False
            continue

        attempt += 1
        initial_draw = generate_draw(seed, dims, radius)

    if attempt > attempts:
        print(f"Failed to find initial values in {attempt} attempts")

    return initial_draw
