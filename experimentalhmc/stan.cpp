#include "stan.h"
#include "phi_inv.cpp"
#include "xoshiro.cpp"
#include <Eigen/Dense>
#include <limits>

// Adapted from various files within
// https://github.com/stan-dev/stan/

class ps_point {

public:
  Eigen::VectorXd position;
  Eigen::VectorXd momentum;
  explicit ps_point(Eigen::VectorXd& q, Eigen::VectorXd& p) : position(q), momentum(p) {}
};

static constexpr double INFTY = std::numeric_limits<double>::infinity();

double Hamiltonian(const double log_density, const ps_point z) {
  return -log_density + 0.5 * z.position.dot(z.momentum);
}

double leapfrog() {};

double uniform_rng(uint64_t* rng) {
  return xoshiro_rand(rng);
}

void normal_rng(uint64_t* rng, Eigen::VectorXd& z) {
  for (auto &zn : z) {
    zn = phi_inv(uniform_rng(rng));
  }
}

inline double log1p_exp(const double a) {
  // like log_sum_exp below with b=0.0; prevents underflow
  if (a > 0.0) {
    return a + std::log1p(std::exp(-a));
  }
  return std::log1p(std::exp(a));
}

inline double log_sum_exp(const double a, const double b) {
  if (a == -INFTY) {
    return b;
  }
  if (a == INFTY && b == INFTY) {
    return INFTY;
  }
  if (a > b) {
    return a + log1p_exp(b - a);
  }
  return b + log1p_exp(a - b);
}

bool compute_criterion(Eigen::VectorXd& p_sharp_minus,
                       Eigen::VectorXd& p_sharp_plus,
                       Eigen::VectorXd& rho) {
  return p_sharp_plus.dot(rho) > 0 && p_sharp_minus.dot(rho) > 0;
}

void stan_transition(const double* q,
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
                     double* accept_prob) {

  ps_point z_ = ps_point(Eigen::VectorXd::Map(q, dims), Eigen::VectorXd::Map(p, dims));

  ps_point z_fwd(z_);           // State at forward end of trajectory
  ps_point z_bck(z_fwd);        // State at backward end of trajectory

  ps_point z_sample(z_fwd);
  ps_point z_propose(z_fwd);

  // Momentum and sharp momentum at forward end of forward subtree
  Eigen::VectorXd p_fwd_fwd = z_.momentum;
  Eigen::VectorXd p_sharp_fwd_fwd = z_.momentum;

  // Momentum and sharp momentum at backward end of forward subtree
  Eigen::VectorXd p_fwd_bck = z_.momentum;
  Eigen::VectorXd p_sharp_fwd_bck = p_sharp_fwd_fwd;

  // Momentum and sharp momentum at forward end of backward subtree
  Eigen::VectorXd p_bck_fwd = z_.momentum;
  Eigen::VectorXd p_sharp_bck_fwd = p_sharp_fwd_fwd;

  // Momentum and sharp momentum at backward end of backward subtree
  Eigen::VectorXd p_bck_bck = z_.momentum;
  Eigen::VectorXd p_sharp_bck_bck = p_sharp_fwd_fwd;

  // Integrated momenta along trajectory
  Eigen::VectorXd rho = z_.momentum.transpose();

  // TODO calculate log_density

  double log_sum_weight = 0.0;
  double H0 = Hamiltonian(log_density, position, momentum);
  double sum_metro_prob = 0.0;
  int n_leapfrog = 0;
  int depth = 0;
  bool divergent = false;

  while ( depth < max_tree_depth ) {
    // Build a new subtree in a random direction
    Eigen::VectorXd rho_fwd = Eigen::VectorXd::Zero(rho.size());
    Eigen::VectorXd rho_bck = Eigen::VectorXd::Zero(rho.size());

    bool valid_subtree = false;
    double log_sum_weight_subtree = -INFTY;

    if (uniform_rng(rng) > 0.5) {
      // Extend the current trajectory forward
      z_ = z_fwd;
      rho_bck = rho;
      p_bck_fwd = p_fwd_fwd;
      p_sharp_bck_fwd = p_sharp_fwd_fwd;

      valid_subtree = build_tree(depth, z_propose, p_sharp_fwd_bck, p_sharp_fwd_fwd, rho_fwd,
                                 p_fwd_bck, p_fwd_fwd, H0, 1, // TODO probably stepsize * direction
                                 n_leapfrog, log_sum_weight_subtree,
                                 sum_metro_prob);
      z_fwd = z_;
    } else {
      // Extend the current trajectory backwards
      z_ = z_bck;
      rho_fwd = rho;
      p_fwd_bck = p_bck_bck;
      p_sharp_fwd_bck = p_sharp_bck_bck;

      valid_subtree = build_tree(
                                 this->depth_, z_propose, p_sharp_bck_fwd, p_sharp_bck_bck, rho_bck,
                                 p_bck_fwd, p_bck_bck, H0, -1, // TODO stepsize * direction?
                                 n_leapfrog, log_sum_weight_subtree,
                                 sum_metro_prob);
      z_bck = z_;
    }

    if (!valid_subtree)
      break;

    // Sample from accepted subtree
    ++depth;

    if (log_sum_weight_subtree > log_sum_weight) {
      z_sample = z_propose;
    } else {
      double accept_prob = std::exp(log_sum_weight_subtree - log_sum_weight);
      if (uniform_rng(rng) < accept_prob)
        z_sample = z_propose;
    }

    log_sum_weight
      = log_sum_exp(log_sum_weight, log_sum_weight_subtree);

    // Break when no-u-turn criterion is no longer satisfied
    rho = rho_bck + rho_fwd;

    // Demand satisfaction around merged subtrees
    bool persist_criterion
      = compute_criterion(p_sharp_bck_bck, p_sharp_fwd_fwd, rho);

    // Demand satisfaction between subtrees
    Eigen::VectorXd rho_extended = rho_bck + p_fwd_bck;

    persist_criterion
      &= compute_criterion(p_sharp_bck_bck, p_sharp_fwd_bck, rho_extended);

    rho_extended = rho_fwd + p_bck_fwd;
    persist_criterion
      &= compute_criterion(p_sharp_bck_fwd, p_sharp_fwd_fwd, rho_extended);

    if (!persist_criterion)
      break;
  }

  *accept_prob = sum_metro_prob / static_cast<double>(n_leapfrog);
  Eigen::VectroXd::Map(position_new, dims) = z_sample;
  // TODO double log_density = ...
  *energy = Hamiltonian(log_density, z_sample);
}
