from .rand import uniform_rand

import numpy as np
import numpy.ctypeslib as npc

def generate_draw(xoshiro_key, dims, radius):
    U = uniform_rand(xoshiro_key, dims)
    return radius * (2 * U - 1)

def initialize_draws(xoshiro_key, dims, ldg, initial_draw_radius = 2, initial_draw_attempts = 100):

    attempts = initial_draw_attempts
    attempt = 0

    initialized = False
    radius = initial_draw_radius
    gradient = np.empty(dims)

    while attempt < attempts and not initialized:
        initial_draw = generate_draw(xoshiro_key, dims, radius)
        ld = ldg(initial_draw, gradient)

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
