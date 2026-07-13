#pragma once

//! Implements the bivariate Clayton copula for lower-tail dependence.

#include <cmath>
#include <limits>
#include <span>
#include <string>

#include <copulas/base.hpp>


class Clayton : public Copula<1> {

  public:
  Clayton(
    std::span<const double> u1,
    std::span<const double> u2,
    double alpha_init = 0.5
  ) 
    : Copula(u1, u2, {{alpha_init, 0.0001, 50.0}})
  {}

  std::string name() const override {
    return "Clayton";
  }

  double cdf(double u1_scalar, double u2_scalar) const {
    const double alpha = this->params()[0][0];
    if (std::abs(alpha) < 1e-12) return u1_scalar * u2_scalar;

    return std::pow(
      std::pow(u1_scalar, -alpha) + std::pow(u2_scalar, -alpha) - 1.0,
      -1.0 / alpha
    );
  }
  
  /// c(u_1, u_2; \alpha) = (\alpha + 1) \left(u_1^{-\alpha} + u_2^{-\alpha} - 1\right)^{-2 - \frac{1}{\alpha}} u_1^{-\alpha - 1} u_2^{-\alpha - 1}
  inline double estimate_copula_density(double u1_scalar, double u2_scalar) const 
  override {
    const double alpha = this->params()[0][0];
    if (std::abs(alpha) < 1e-12) return 1.0;

    return
      (alpha + 1.0) *
      std::powf(
        (std::powf(u1_scalar, -alpha) + std::powf(u2_scalar, -alpha) - 1.0),
        -2.0 - (1.0 / alpha)
      ) *
      std::powf(u1_scalar, -alpha - 1.0) * std::powf(u2_scalar, -alpha - 1.0);
  }

  /// C_{2|1}(u_2 \mid u_1; \alpha) = u_1^{-(1+\alpha)} \left(u_1^{-\alpha} + u_2^{-\alpha} - 1\right)^{-\frac{1+\alpha}{\alpha}}
  double h_conditional_prob(double u1_scalar, double u2_scalar) const override {
    const double alpha = this->params()[0][0];
    if (std::abs(alpha) < 1e-12) return u2_scalar;

    return 
      std::powf(u1_scalar, -(1.0 + alpha)) *
      std::powf(
        (std::powf(u1_scalar, -alpha) + std::powf(u2_scalar, -alpha) - 1.0),
        (-(1.0 + alpha) / alpha)
      );
  }

  /// \frac{/alpa}{/alpha + 2.0}
  double kendalls_tau() const override {
    const double alpha = this->params()[0][0];
    return alpha / (alpha + 2.0);
  }

  /// \alpha = \frac{2.0 tau}{1.0 - tau}
  static double alpha_from_kendalls_tau(double tau) {
    if (tau >= 1.0) {
      return std::numeric_limits<double>::infinity();
    }
    return (2.0 * tau) / (1.0 - tau);
  }
};
