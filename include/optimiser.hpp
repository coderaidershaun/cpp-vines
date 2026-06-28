//! Nelder Mead optimiser adapted from existing Cryptoi Wizards Rust codebase.
//! Allows for MLE discovery of optimum Copula parameters. 

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <utility>

#include <types.hpp>


template<usize N>
using Params = std::array<double, N>;

template<usize N>
using Bounds = std::array<std::array<double, 2>, N>;

template<usize N>
using ObjectiveFunction = std::function<double(const Params<N>&)>;

enum class Result {
  Failure,
  Pending,
  Success
};

struct OptimiserResults {
  double f_opt_ll;                        // Objective function log likelihood value at the optimum
  double aic;                             // AIC value
  double number_it;                       // Number of iterations performed
  double number_fev;                      // Number of function evaluations
  Result result = Result::Pending;        // Result status
};

template<usize N>
class NelderMeadSimplex {
  using Simplex = std::array<std::array<double, N>, N + 1>;
  using ObjectiveValues = std::array<double, N + 1>;

  ObjectiveFunction<N> m_objective_f;     // Objective function to minimise
  double m_tolerance = 1e-8;              // Tolerance for convergence
  double m_alpha = 1.0;                   // Reflection coefficient
  double m_gamma = 2.0;                   // Expansion coefficient
  double m_rho = 0.5;                     // Contraction coefficient
  double m_sigma = 0.5;                   // Shrinkage coefficient
  double m_k = static_cast<double>(N);    // Number of params
  bool m_adaptive;                        // Flag for adaptive parameter settings
  usize m_max_iter;                       // Max iterations without convergence

  ParamBounds<N> m_params_with_bounds;
  Params<N> m_optimal_params{};

  OptimiserResults m_results{};

  public:
  NelderMeadSimplex(
    ObjectiveFunction<N> objective_f,
    ParamBounds<N> params,
    bool adaptive = true,
    usize max_iter = 10'000
  ) 
    : m_objective_f(std::move(objective_f)),
      m_adaptive(adaptive),
      m_max_iter(max_iter),
      m_params_with_bounds(params)
  {
    for (usize i=0; i<N; i++) {
      m_optimal_params[i] = m_params_with_bounds[i][0];
    }

    if (adaptive) {
      m_alpha = 1.0;
      m_gamma = 1.0 + 2.0 / m_k;
      m_rho = 0.75 - 1.0 / (2.0 * m_k);
      m_sigma = 1.0 - 1.0 / m_k;
    }
  }

  const OptimiserResults& results() const {
    return m_results;
  }

  const Params<N>& optimal_params() const {
    return m_optimal_params;
  }

  OptimiserResults fit() {
    auto [x_start, x_bounds] = params_bounds_split();

    clamp_params_to_bounds(x_start, x_bounds);
    Simplex simplex = simplex_init(x_start);
    clamp_simplex_to_bounds(simplex, x_bounds);

    ObjectiveValues objective_values = eval_obj_fn_at_simplex_vertices(simplex);
    usize function_calls = simplex.size();
    usize iterations = 0;

    // Performs optimisation via iteration
    while (iterations < m_max_iter) {
      iterations += 1;
      sort_simplex(simplex, objective_values);

      if (simplex_converged(simplex, objective_values)) break;

      const Params<N> centroid = simplex_centroid(simplex);
      Params<N> reflected = reflect_worst_vertex(simplex, centroid);
      clamp_params_to_bounds(reflected, x_bounds);

      const double reflected_value = m_objective_f(reflected);
      function_calls += 1;

      if (
        objective_values[0] <= reflected_value &&
        reflected_value < objective_values[N - 1]
      ) {
        replace_worst_vertex(simplex, objective_values, reflected, reflected_value);
      } else if (reflected_value < objective_values[0]) {
        Params<N> expanded = expand_reflected_vertex(centroid, reflected);
        clamp_params_to_bounds(expanded, x_bounds);

        const double expanded_value = m_objective_f(expanded);
        function_calls += 1;

        if (expanded_value < reflected_value) {
          replace_worst_vertex(simplex, objective_values, expanded, expanded_value);
        } else {
          replace_worst_vertex(simplex, objective_values, reflected, reflected_value);
        }
      } else {
        Params<N> contracted = contract_worst_vertex(simplex, centroid);
        clamp_params_to_bounds(contracted, x_bounds);

        const double contracted_value = m_objective_f(contracted);
        function_calls += 1;

        if (contracted_value < objective_values[N]) {
          replace_worst_vertex(simplex, objective_values, contracted, contracted_value);
        } else {
          shrink_simplex(simplex, objective_values, x_bounds);
          function_calls += N;
        }
      }
    }

    sort_simplex(simplex, objective_values);
    m_optimal_params = simplex[0];
    const double f_opt = objective_values[0];

    m_results.f_opt_ll = -f_opt;
    m_results.aic = 2.0 * m_k - 2.0 * m_results.f_opt_ll;
    m_results.number_it = static_cast<double>(iterations);
    m_results.number_fev = static_cast<double>(function_calls);
    m_results.result = iterations >= m_max_iter ? Result::Failure : Result::Success;

    return m_results;
  }

  private:
  inline std::pair<Params<N>, Bounds<N>> params_bounds_split() const {
    Params<N> params{};
    Bounds<N> bounds{};

    for (usize i=0; i<N; i++) {
      params[i] = m_params_with_bounds[i][0];
      bounds[i][0] = m_params_with_bounds[i][1];
      bounds[i][1] = m_params_with_bounds[i][2];
    }

    return {params, bounds};
  }

  inline void clamp_params_to_bounds(
    Params<N>& params,
    const Bounds<N>& bounds
  ) const {
    for (usize i=0; i<N; i++) {
      params[i] = std::max(
        bounds[i][0],
        std::min(params[i], bounds[i][1])
      );
    }
  }

  inline void clamp_simplex_to_bounds(
    Simplex& simplex,
    const Bounds<N>& bounds
  ) const {
    for (Params<N>& params : simplex) {
      clamp_params_to_bounds(params, bounds);
    }
  }

  inline Simplex simplex_init(const Params<N>& x_start) const {
    Simplex simplex{};
    simplex[0] = x_start;

    for (usize i=0; i<N; i++) {
      auto y = x_start;
      if (y[i] != 0.0) {
        y[i] *= 1.05;
      } else {
        y[i] = 0.00025;
      }
      simplex[i + 1] = y;
    }

    return simplex;
  }

  inline ObjectiveValues eval_obj_fn_at_simplex_vertices(const Simplex& simplex) const {
    ObjectiveValues objective_values{};
    for(usize i=0; i<simplex.size(); i++) {
      objective_values[i] = m_objective_f(simplex[i]);
    }

    return objective_values;
  }

  inline void sort_simplex(Simplex& simplex, ObjectiveValues& objective_values) const {
    std::array<usize, N + 1> indices{};
    for (usize i=0; i<indices.size(); i++) {
      indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(), [&](usize i, usize j) {
      return objective_values[i] < objective_values[j];
    });

    const Simplex unsorted_simplex = simplex;
    const ObjectiveValues unsorted_objective_values = objective_values;

    for (usize i=0; i<indices.size(); i++) {
      simplex[i] = unsorted_simplex[indices[i]];
      objective_values[i] = unsorted_objective_values[indices[i]];
    }
  }

  inline bool objective_values_converged(const ObjectiveValues& objective_values) const {
    for (usize i=1; i<objective_values.size(); i++) {
      if (std::abs(objective_values[i] - objective_values[0]) >= m_tolerance) {
        return false;
      }
    }

    return true;
  }

  inline bool simplex_vertices_converged(const Simplex& simplex) const {
    for (usize i=1; i<simplex.size(); i++) {
      for (usize j=0; j<N; j++) {
        if (std::abs(simplex[i][j] - simplex[0][j]) >= m_tolerance) {
          return false;
        }
      }
    }

    return true;
  }

  inline bool simplex_converged(
    const Simplex& simplex,
    const ObjectiveValues& objective_values
  ) const {
    return
      objective_values_converged(objective_values) &&
      simplex_vertices_converged(simplex);
  }

  inline Params<N> simplex_centroid(const Simplex& simplex) const {
    Params<N> centroid{};
    for (usize param_i=0; param_i<N; param_i++) {
      for (usize vertex_i=0; vertex_i<N; vertex_i++) {
        centroid[param_i] += simplex[vertex_i][param_i];
      }
      centroid[param_i] /= m_k;
    }

    return centroid;
  }

  inline Params<N> reflect_worst_vertex(
    const Simplex& simplex,
    const Params<N>& centroid
  ) const {
    Params<N> reflected{};

    for (usize i=0; i<N; i++) {
      reflected[i] = centroid[i] + m_alpha * (centroid[i] - simplex[N][i]);
    }

    return reflected;
  }

  inline Params<N> expand_reflected_vertex(
    const Params<N>& centroid,
    const Params<N>& reflected
  ) const {
    Params<N> expanded{};

    for (usize i=0; i<N; i++) {
      expanded[i] = centroid[i] + m_gamma * (reflected[i] - centroid[i]);
    }

    return expanded;
  }

  inline Params<N> contract_worst_vertex(
    const Simplex& simplex,
    const Params<N>& centroid
  ) const {
    Params<N> contracted{};

    for (usize i=0; i<N; i++) {
      contracted[i] = centroid[i] + m_rho * (simplex[N][i] - centroid[i]);
    }

    return contracted;
  }

  inline void replace_worst_vertex(
    Simplex& simplex,
    ObjectiveValues& objective_values,
    const Params<N>& replacement,
    double replacement_value
  ) const {
    simplex[N] = replacement;
    objective_values[N] = replacement_value;
  }

  inline void shrink_simplex(
    Simplex& simplex,
    ObjectiveValues& objective_values,
    const Bounds<N>& bounds
  ) const {
    for (usize i=1; i<simplex.size(); i++) {
      for (usize j=0; j<N; j++) {
        simplex[i][j] =
          simplex[0][j] +
          m_sigma * (simplex[i][j] - simplex[0][j]);
      }

      clamp_params_to_bounds(simplex[i], bounds);
      objective_values[i] = m_objective_f(simplex[i]);
    }
  }
};
