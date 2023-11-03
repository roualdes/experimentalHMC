#include "phi_inv.cpp"
#include "xoshiro.cpp"
#include <Eigen/Dense>

extern "C" {
  void f(const double* theta, const int N, double* out) {
    Eigen::VectorXd q = Eigen::VectorXd::Map(theta, N);

    out[0] = 3;
    out[1] = 2.5;
    out[2] = -1;
    out[3] = q(0) + q(0);
  }

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

  double xoshiro_rand_u(uint64_t* s) {
    return xoshiro_rand(s);
  }

  void xoshiro_rand_u_broadcast(uint64_t* s, int N, double* x) {
    for (int n = 0; n < N; ++n) {
      x[n] = xoshiro_rand(s);
    }
  }

  double xoshiro_rand_n(uint64_t* s) {
    double z;
    int err = -1;
    while (err != 0) {
      err = phi_inv(xoshiro_rand(s), &z);
    };
    return z;
  }

  void xoshiro_rand_n_broadcast(uint64_t* s, int N, double* z) {
    for (int n = 0; n < N; ++n) {
      z[n] = xoshiro_rand_n(s);
    }
  }
}
