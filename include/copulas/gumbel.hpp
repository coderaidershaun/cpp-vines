#pragma once

//! Implements the bivariate Gumbel copula for upper-tail dependence.

#include <cmath>
#include <limits>
#include <span>
#include <string>

#include <copulas/base.hpp>


class Gumbel : public Copula<1> {

  public:
  Gumbel(
    std::span<const double> u1,
    std::span<const double> u2,
    double delta_init = 0.5
  ) 
    : Copula(u1, u2, {{delta_init, 1.0001, 50.0}})
  {}

  std::string name() const override {
    return "Gumbel";
  }

  double cdf(double u1_scalar, double u2_scalar) const {
    return std::exp(-term_A(u1_scalar, u2_scalar));
  }

  /// c(u_1, u_2; \delta) = (A + \delta - 1) A^{1 - 2\delta} \exp(-A) (u_1 u_2)^{-1} (-\ln u_1)^{\delta - 1} (-\ln u_2)^{\delta - 1}
  inline double estimate_copula_density(double u1_scalar, double u2_scalar) const 
  override {
    const double delta = this->m_params[0][0];
    const double A = term_A(u1_scalar, u2_scalar);
    return (
      (A + delta - 1.0) * std::pow(A, (1.0 - 2.0 * delta)) *
      std::exp(-A) * std::pow(u1_scalar * u2_scalar, -1.0) *
      std::pow(-std::log(u1_scalar), delta - 1.0) *
      std::pow(-std::log(u2_scalar), delta - 1.0)
    );
  }

  /// C_{2|1}(u_2 \mid u_1; \delta) = u_1^{-1} (-\ln u_1)^{\delta - 1} B \exp(-A)
  double h_conditional_prob(double u1_scalar, double u2_scalar) const override {
    const double delta = this->m_params[0][0];
    return 
      std::pow(u1_scalar, -1.0) *
      std::pow(-std::log(u1_scalar), delta - 1.0) *
      term_B(u1_scalar, u2_scalar) *
      std::exp(-term_A(u1_scalar, u2_scalar));
  }

  /// \tau = 1 - \frac{1}{\delta}
  double kendalls_tau() const override {
    const double delta = this->params()[0][0];
    if (delta == 0.0) {
      return -std::numeric_limits<double>::infinity();
    }
    return 1.0 - (1.0 / delta);
  }

  /// \delta = \frac{1}{1 - \tau}
  static double delta_from_kendalls_tau(double tau) {
    if (tau >= 1.0) {
      return std::numeric_limits<double>::infinity();
    }
    return 1.0 / (1.0 - tau);
  }

  private:

  /// A = \left((- \ln u_1)^{\delta} + (- \ln u_2)^{\delta}\right)^{1/\delta}
  inline double term_A(double u1_scalar, double u2_scalar) const {
    const double delta = this->m_params[0][0];
    return
      std::pow(
        (
          std::pow(-std::log(u1_scalar), delta) +
          std::pow(-std::log(u2_scalar), delta)
        ), 
      1.0 / delta
    );
  }

  /// B = \left((- \ln u_1)^{\delta} + (- \ln u_2)^{\delta}\right)^{\frac{1-\delta}{\delta}}
  inline double term_B(double u1_scalar, double u2_scalar) const {
    const double delta = this->m_params[0][0];
    return
      std::pow(
        (
          std::pow(-std::log(u1_scalar), delta) +
          std::pow(-std::log(u2_scalar), delta)
        ), 
      (1.0 - delta) / delta
    );
  }
};
