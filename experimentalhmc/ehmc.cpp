#include "phi_inv.cpp"
#include "xoshiro.cpp"
#include "stan.cpp"
#include <Eigen/Dense>

extern "C" {
  int normal_invcdf(const double p, double* z) {
    return phi_inv(p, z);
  }

  int normal_invcdf_broadcast(const double* p, int N, double* z) {
    int err = 0;
    for (int n = 0; n < N; ++n) {
      err = phi_inv(p[n], &z[n]);
      if (err != 0) break;
    }
    return err;
  }

  void xoshiro_set_seed(uint64_t* x, uint64_t* s) {
    set_seed(x, s);
  }

  double uniform_rng(uint64_t* s) {
    return xoshiro_rand(s);
  }

  void uniform_rng_broadcast(uint64_t* s, int N, double* x) {
    for (int n = 0; n < N; ++n) {
      x[n] = xoshiro_rand(s);
    }
  }

  double normal_rng(uint64_t* s) {
    double z;
    int err = -1;
    while (err != 0) {
      err = phi_inv(xoshiro_rand(s), &z);
    };
    return z;
  }

  void normal_rng_broadcast(uint64_t* s, int N, double* z) {
    for (int n = 0; n < N; ++n) {
      z[n] = normal_rng(s);
    }
  }

  void stan_transition(const double* q,
                       double(*ldg)(double* q, double* p),
                       uint64_t* rng,
                       const int dims,
                       const double* metric,
                       const double step_size,
                       const double max_delta_H,
                       const int max_tree_depth,
                       double* position_new,
                       double* energy,
                       double* accept_prob) {
    stan_kernel(q, ldg, rng, dims, metric, step_size,
                max_delta_H, max_tree_depth,
                position_new, energy, accept_prob);
  }
}
