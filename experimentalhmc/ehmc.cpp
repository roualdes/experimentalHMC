#include "phi_inv.cpp"
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
}
