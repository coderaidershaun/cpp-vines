#pragma once

//! Implements the zero-parameter bivariate independence copula.

#include <span>
#include <string>

#include <copulas/base.hpp>


class Independence : public Copula<0> {

  public:
  Independence(
    std::span<const double> u1,
    std::span<const double> u2
  )
    : Copula(u1, u2, {})
  {}

  static constexpr double cdf(double u1_scalar, double u2_scalar) {
    return u1_scalar * u2_scalar;
  }

  OptimiserResults fit() const {
    return OptimiserResults{
      .f_opt_ll = 0.0,
      .aic = 0.0,
      .bic = 0.0,
      .result = Result::Success
    };
  }

  double h_conditional_prob(
    double /* u1_scalar */,
    double u2_scalar
  ) const override {
    return u2_scalar;
  }

  double estimate_copula_density(
    double /* u1_scalar */,
    double /* u2_scalar */
  ) const override {
    return 1.0;
  }

  std::string name() const override {
    return "Independence";
  }

  double kendalls_tau() const override {
    return 0.0;
  }
};
