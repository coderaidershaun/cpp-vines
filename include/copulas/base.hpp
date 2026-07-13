#pragma once

//! Base class for structuring other copulas within folder.

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <optimiser.hpp>
#include <types.hpp>


struct CondProbsH {
  const double u1_given_u2;
  const double u2_given_u1;

  CondProbsH(const double u1_given_u2, const double u2_given_u1)
    : u1_given_u2(u1_given_u2),
      u2_given_u1(u2_given_u1)
  {}
};

struct CopulaDensityEstimate {
  std::vector<double> densities;
  double ln_likelihood;
};

template <usize ParamsN>
class Copula {

  protected:
  std::span<const double> m_u1;
  std::span<const double> m_u2;
  ParamBounds<ParamsN> m_params;

  public:
  Copula(
    std::span<const double> u1,
    std::span<const double> u2,
    const ParamBounds<ParamsN>& params
  )
    : m_u1(u1),
      m_u2(u2),
      m_params(params)
  {}
  
  virtual ~Copula() = default;

  inline const ParamBounds<ParamsN>& params() const {
    return m_params;
  }

  static constexpr usize parameter_count() {
    return ParamsN;
  }

  /// Runs optimisation algorithm to finds optimum parameter(s) for copula
  OptimiserResults fit() {
    const ParamBounds<ParamsN> initial_params = m_params;

    const auto negative_log_likelihood =
      [this](const std::array<double, ParamsN>& candidate_params) {
        for (usize i=0; i<ParamsN; i++) {
          m_params[i][0] = candidate_params[i];
        }

        return -estimate_copula_prob_densities().ln_likelihood;
      };

    NelderMeadSimplex<ParamsN> optimiser(
      negative_log_likelihood,
      m_params
    );

    OptimiserResults results = optimiser.fit();
    results.bic =
      -2.0 * results.f_opt_ll +
      std::log(static_cast<double>(m_u1.size())) *
      static_cast<double>(parameter_count());

    if (results.result != Result::Success) {
      m_params = initial_params;
      std::cout << "Optimisation failed for: " << name() << std::endl;
      return results;
    }

    const auto& optimal_params = optimiser.optimal_params();
    for (usize i=0; i<ParamsN; i++) {
      m_params[i][0] = optimal_params[i];
    }

    return results;
  }
  
  CopulaDensityEstimate estimate_copula_prob_densities() const {
    double ln_likelihood = 0.0;
    std::vector<double> copula_prob_densities;
    const usize n = this->m_u1.size();
    copula_prob_densities.reserve(n);
    bool has_invalid_density = false;

    for (usize i=0; i<n; i++) {
      const double copula_density = estimate_copula_density(this->m_u1[i], this->m_u2[i]);
      copula_prob_densities.emplace_back(copula_density);

      if (!std::isfinite(copula_density) || copula_density <= 0.0) {
        has_invalid_density = true;
        continue;
      }

      if (!has_invalid_density) {
        ln_likelihood += std::log(copula_density);
      }
    }

    return {
      std::move(copula_prob_densities),
      has_invalid_density ? -std::numeric_limits<double>::infinity() : ln_likelihood
    };
  }

  CondProbsH h_conditional_prob_set(double u1_scalar, double u2_scalar) const {
    return CondProbsH (
      h_conditional_prob(u2_scalar, u1_scalar),
      h_conditional_prob(u1_scalar, u2_scalar)
    );
  }

  virtual double estimate_copula_density(double u1_scalar, double u2_scalar) const = 0;
  virtual double h_conditional_prob(double u1_scalar, double u2_scalar) const = 0;
  virtual std::string name() const = 0;
  virtual double kendalls_tau() const = 0;
};
