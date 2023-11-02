#include "inv_std_normal.cpp"
#include <iostream>
#include <Eigen/Dense>

extern "C" {
  void f(const double* theta, const int N, double* out) {
    Eigen::VectorXd q = Eigen::VectorXd::Map(theta, N);

    out[0] = 3;
    out[1] = 2.5;
    out[2] = -1;
    out[3] = q(0) + q(0);
  }

  double inv_normal(const double p) {
    return inv_std_normal(p);
  }
}
