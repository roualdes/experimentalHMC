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

  extern void stan_kernel(const double* q,
                          const double* p,
                          void (*log_density_gradient)(double* q, double* p),
                          uint64_t* rng,
                          const int dims,
                          const double* metric,
                          const double step_size,
                          const double max_delta_H,
                          const int max_tree_depth,
                          double* position_new,
                          double* energy,
                          double* accept_prob);

#ifdef __cplusplus
}
#endif

#endif
