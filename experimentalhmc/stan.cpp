#include "stan.h"
#include "phi_inv.h"
#include "xoshiro.h"
#include <Eigen/Dense>
#include <limits>
#include <iostream>

// Adapted from various files within
// https://github.com/stan-dev/stan/

class ps_point {
public:
  Eigen::VectorXd position;
  Eigen::VectorXd momentum;
  template<typename Derived>
  explicit ps_point(const Eigen::MatrixBase<Derived>& q,
                    const Eigen::MatrixBase<Derived>& p) : position(q), momentum(p) {}
};

static constexpr double INFTY = std::numeric_limits<double>::infinity();

double Hamiltonian(const double log_density, const ps_point z) {
  return -log_density + 0.5 * z.momentum.dot(z.momentum);
}

double leapfrog(ps_point& z,
                const Eigen::VectorXd step_size,
                const int steps,
                Eigen::VectorXd& gradient,
                double(*log_density_gradient)(double* q, double* grad)) {
  double ld;
  z.momentum += 0.5 * step_size * gradient;

  for (int step = 0; step < steps; ++step) {
    z.position += step_size * z.momentum;
    ld = (*log_density_gradient)(z.position.data(), gradient.data());
    if (step != steps - 1) {
      z.momentum += step_size * gradient;
    }
  }

  z.momentum += 0.5 * step_size * gradient;

  return ld;
}


double _uniform_rng(uint64_t* rng) {
  return xoshiro_rand(rng);
}

void _normal_rng(uint64_t* rng, Eigen::VectorXd& z) {
  for (auto &zn : z) {
    phi_inv(_uniform_rng(rng), &zn);
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


bool build_tree(int tree_depth,
                double(*log_density_gradient)(double* q, double* grad),
                Eigen::VectorXd& gradient,
                uint64_t* rng,
                ps_point& z_,
                ps_point& z_propose,
                Eigen::VectorXd& p_sharp_beg,
                Eigen::VectorXd& p_sharp_end,
                Eigen::VectorXd& rho,
                Eigen::VectorXd& p_beg,
                Eigen::VectorXd& p_end,
                Eigen::VectorXd step_size,
                double H0,
                int* n_leapfrog,
                double& log_sum_weight,
                double& sum_metro_prob,
                double max_delta_H) {
  // Base case
  if (tree_depth == 0) {
    bool divergent = false;

    double ld = (*log_density_gradient)(z_.position.data(), gradient.data());

    ld = leapfrog(z_, step_size, 1, gradient, log_density_gradient);
    ++(*n_leapfrog);

    z_propose = z_;

    double h = Hamiltonian(ld, z_);
    if (std::isnan(h)) {
      h = INFTY;
    }

    if ((h - H0) > max_delta_H) {
      divergent = true;
    }

    log_sum_weight = log_sum_exp(log_sum_weight, H0 - h);

    if (H0 - h > 0) {
      sum_metro_prob += 1;
    } else {
      sum_metro_prob += std::exp(H0 - h);
    }

    p_sharp_beg = z_.momentum;
    p_sharp_end = p_sharp_beg;

    rho += z_.momentum;
    p_beg = z_.momentum;
    p_end = p_beg;

    return !divergent;
  }
  // General recursion

  // Build the initial subtree
  double log_sum_weight_init = -INFTY;

  // Momentum and sharp momentum at end of the initial subtree
  Eigen::VectorXd p_init_end(z_.momentum.size());
  Eigen::VectorXd p_sharp_init_end(z_.momentum.size());

  Eigen::VectorXd rho_init = Eigen::VectorXd::Zero(rho.size());

  bool valid_init
    = build_tree(tree_depth - 1, log_density_gradient, gradient, rng, z_, z_propose,
                 p_sharp_beg, p_sharp_init_end,
                 rho_init, p_beg, p_init_end,
                 step_size, H0,
                 n_leapfrog, log_sum_weight_init,
                 sum_metro_prob, max_delta_H);

  if (!valid_init) {
    return false;
  }

  // Build the final subtree
  ps_point z_propose_final(z_);

  double log_sum_weight_final = -INFTY;

  // Momentum and sharp momentum at beginning of the final subtree
  Eigen::VectorXd p_final_beg(z_.momentum.size());
  Eigen::VectorXd p_sharp_final_beg(z_.momentum.size());

  Eigen::VectorXd rho_final = Eigen::VectorXd::Zero(rho.size());

  bool valid_final
    = build_tree(tree_depth - 1, log_density_gradient, gradient, rng, z_, z_propose_final,
                 p_sharp_final_beg, p_sharp_end,
                 rho_final, p_final_beg, p_end,
                 step_size, H0,
                 n_leapfrog, log_sum_weight_final,
                 sum_metro_prob, max_delta_H);

  if (!valid_final) {
    return false;
  }

  // Multinomial sample from right subtree
  double log_sum_weight_subtree
    = log_sum_exp(log_sum_weight_init, log_sum_weight_final);
  log_sum_weight = log_sum_exp(log_sum_weight, log_sum_weight_subtree);

  if (log_sum_weight_final > log_sum_weight_subtree) {
    z_propose = z_propose_final;
  } else {
    double accept_prob
      = std::exp(log_sum_weight_final - log_sum_weight_subtree);
    if (_uniform_rng(rng) < accept_prob) {
      z_propose = z_propose_final;
    }
  }

  Eigen::VectorXd rho_subtree = rho_init + rho_final;
  rho += rho_subtree;

  // Demand satisfaction around merged subtrees
  bool persist_criterion
    = compute_criterion(p_sharp_beg, p_sharp_end, rho_subtree);

  // Demand satisfaction between subtrees
  rho_subtree = rho_init + p_final_beg;
  persist_criterion
    &= compute_criterion(p_sharp_beg, p_sharp_final_beg, rho_subtree);

  rho_subtree = rho_final + p_init_end;
  persist_criterion
    &= compute_criterion(p_sharp_init_end, p_sharp_end, rho_subtree);

  return persist_criterion;
}


void stan_kernel(double* q,
                 double(*log_density_gradient)(double* q, double* p),
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


  Eigen::VectorXd position(dims);
  for (int d = 0; d < dims; ++d) {
    position(d) = q[d];
  }

  Eigen::VectorXd momentum(dims);
  _normal_rng(rng, momentum);

  ps_point z_ = ps_point(position, momentum);

  Eigen::VectorXd M = Eigen::VectorXd::Map(metric, dims);
  Eigen::VectorXd ss = step_size * M.array().sqrt();

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

  Eigen::VectorXd gradient(dims);
  double ld = (*log_density_gradient)(z_.position.data(), gradient.data());
  double H0 = Hamiltonian(ld, z_);

  double log_sum_weight = 0.0;
  double sum_metro_prob = 0.0;
  *n_leapfrog = 0;
  *tree_depth = 0;

  while ( *tree_depth < max_tree_depth ) {
    // Build a new subtree in a random direction
    Eigen::VectorXd rho_fwd = Eigen::VectorXd::Zero(rho.size());
    Eigen::VectorXd rho_bck = Eigen::VectorXd::Zero(rho.size());

    bool valid_subtree = false;
    double log_sum_weight_subtree = -INFTY;

    if (_uniform_rng(rng) > 0.5) {
      // Extend the current trajectory forward
      z_.ps_point::operator=(z_fwd);
      rho_bck = rho;
      p_bck_fwd = p_fwd_fwd;
      p_sharp_bck_fwd = p_sharp_fwd_fwd;

      valid_subtree = build_tree(*tree_depth, log_density_gradient, gradient, rng, z_, z_propose,
                                 p_sharp_fwd_bck, p_sharp_fwd_fwd,
                                 rho_fwd, p_fwd_bck, p_fwd_fwd,
                                 ss, H0,
                                 n_leapfrog, log_sum_weight_subtree,
                                 sum_metro_prob, max_delta_H);
      z_fwd.ps_point::operator=(z_);
    } else {
      // Extend the current trajectory backwards
      z_.ps_point::operator=(z_bck);
      rho_fwd = rho;
      p_fwd_bck = p_bck_bck;
      p_sharp_fwd_bck = p_sharp_bck_bck;

      valid_subtree = build_tree(*tree_depth, log_density_gradient, gradient, rng, z_, z_propose,
                                 p_sharp_bck_fwd, p_sharp_bck_bck,
                                 rho_bck, p_bck_fwd, p_bck_bck,
                                 -1 * ss, H0,
                                 n_leapfrog, log_sum_weight_subtree,
                                 sum_metro_prob, max_delta_H);
      z_bck.ps_point::operator=(z_);
    }

    if (!valid_subtree) {
      *divergent = true;
      break;
    }

    // Sample from accepted subtree
    ++(*tree_depth);

    if (log_sum_weight_subtree > log_sum_weight) {
      z_sample = z_propose;
    } else {
      double accept_prob = std::exp(log_sum_weight_subtree - log_sum_weight);
      if (_uniform_rng(rng) < accept_prob) {
        z_sample = z_propose;
      }
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

    if (!persist_criterion) {
      break;
    }
  }

  *accept_prob = sum_metro_prob / static_cast<double>(*n_leapfrog);
  Eigen::VectorXd::Map(q, dims) = z_sample.position;
  ld = (*log_density_gradient)(z_sample.position.data(), gradient.data());
  *energy = Hamiltonian(ld, z_sample);
}
