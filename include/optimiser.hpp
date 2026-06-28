#pragma once

#include <functional>
#include <span>
#include <types.hpp>

using ObjectiveFunction = std::function<double(const std::vector<double>&)>;

using BoundsArray = std::array<std::array<double, 2>, 2>;

enum class Result {
  Success,
  Failure
};

struct OptimiserResults {
  double f_opt_ll;                      // Objective function log likelihood value at the optimum
  double aic;                           // AIC value
  double number_it;                     // Number of iterations performed
  double number_fev;                    // Number of function evaluations
  Result result;                        // Result status (success or failure)
};

class NelderMeadSimplex {

  ObjectiveFunction objective_f;        // Objective function to minimise
  double tolerance = 1e-8;              // Tolerance for convergence
  double alpha = 1.0;                   // Reflection coefficient
  double gamma = 2.0;                   // Expansion coefficient
  double rho = 0.5;                     // Contraction coefficient
  double sigma = 0.5;                   // Shrinkage coefficient
  bool adaptive;                        // Flag for adaptive parameter settings
  std::array<double, 2> bounds;         // Parameter bounds
  double k;                             // Number of params
  usize max_iter;                       // Max iterations without convergence

  public:

  template <usize N>
  NelderMeadSimplex(
    ObjectiveFunction objective_f,
    std::array<double, N>* params,
    bool adaptive = true,
    usize max_iter = 1000
  ) 
    : objective_f(std::move(objective_f)),
      k(static_case<double>(params.size())),
      max_iter(max_iter)
  {
    if (adaptive) {
      alpha = 1.0;
      gamma = 1.0 + 2.0 / k;
      rho = 0.75 - 1.0 / (2.0 * k);
      sigma = 1.0 - 1.0 / k
    }
  }

  void fit() {

  }
};