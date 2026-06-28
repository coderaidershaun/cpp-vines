#pragma once

#include <array>
#include <tuple>
#include <vector>
#include <types.hpp>

struct CondProbsH {
  const double u1_given_u2;
  const double u2_given_u1;

  CondProbsH(const double u1_given_u2, const double u2_given_u1)
    : u1_given_u2(u1_given_u2),
      u2_given_u1(u2_given_u1)
  {}
};

template <usize ParamsN>
class Copula {

  protected:
  const std::vector<double>& m_u1;
  const std::vector<double>& m_u2;
  ParamBounds<ParamsN> m_params;

  public:
  Copula(
    const std::vector<double>& u1,
    const std::vector<double>& u2,
    ParamBounds<ParamsN> m_params
  )
    : m_u1(u1),
      m_u2(u2),
      m_params(m_params)
  {}
  
  virtual ~Copula() = default;

  inline const ParamBounds<ParamsN>& params() const { return m_params; }
  
  std::tuple<std::vector<double>, double> estimate_copula_prob_densities() const {
    double ln_likelihood = 0.0;
    std::vector<double> copula_prob_densities;
    const usize n = this->m_u1.size();
    copula_prob_densities.reserve(n);

    for (usize i=0; i<n; i++) {
      double copula_density = estimate_copula_density(this->m_u1[i], this->m_u2[i]);
      ln_likelihood += std::log(copula_density);
      copula_prob_densities.emplace_back(copula_density);
    }

    return std::tuple{copula_prob_densities, ln_likelihood};
  }

  CondProbsH h_conditional_prob_set(double u1_scalar, double u2_scalar) const {
    return CondProbsH (
      h_conditional_prob(u2_scalar, u1_scalar),
      h_conditional_prob(u1_scalar, u2_scalar)
    );
  }

  virtual double estimate_copula_density(double u1_scalar, double u2_scalar) const = 0;
  virtual double h_conditional_prob(double u1_scalar, double u2_scalar) const = 0;
};
