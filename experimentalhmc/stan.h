#ifndef STAN_H
#define STAN_H

/// \file stan.h

#ifdef __cplusplus
#include "cmath"
#include "cstdint"
extern "C" {
#else
#include <math.h>
#include <stdint.h>
#endif

  extern void stan_transition(const double* position,
                              const double* momentum,
                              void (*log_density_gradient)(double* q, double* p),
                              uint64_t* s,
                              const int dims,
                              const double* metric,
                              const double step_size,
                              const double max_delta_H,
                              const int max_tree_depth);

#ifdef __cplusplus
}
#endif

#endif
