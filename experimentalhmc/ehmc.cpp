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

  double uniform_rand(uint64_t* s) {
    return xoshiro_rand(s);
  }

  void uniform_rand_broadcast(uint64_t* s, int N, double* x) {
    for (int n = 0; n < N; ++n) {
      x[n] = xoshiro_rand(s);
    }
  }

  double normal_rand(uint64_t* s) {
    double z;
    int err = -1;
    while (err != 0) {
      err = phi_inv(xoshiro_rand(s), &z);
    };
    return z;
  }

  void normal_rand_broadcast(uint64_t* s, int N, double* z) {
    for (int n = 0; n < N; ++n) {
      z[n] = normal_rand(s);
    }
  }

  void stan_transition(double* q,
                       double(*ldg)(double* q, double* p),
                       uint64_t* rng,
                       double* accept_prob,
                       bool* divergent,
                       int* n_leapfrog,
                       int* tree_depth,
                       double* energy,
                       const int dims,
                       const double* metric,
                       const double step_size,
                       const double max_delta_H,
                       const int max_tree_depth) {
    stan_kernel(q, ldg, rng, accept_prob, divergent,
                n_leapfrog, tree_depth, energy, dims,
                metric, step_size, max_delta_H, max_tree_depth);
  }
}
