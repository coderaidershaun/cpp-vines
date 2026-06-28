#pragma once

#include <cmath>
#include <tuple>
#include <copulas/base.hpp>

class Gumbel : public Copula<1> {

  public:
  Gumbel(
    const std::vector<double>& u1,
    const std::vector<double>& u2,
    double delta_init = 0.5
  ) 
    : Copula(u1, u2, {{delta_init, 1.0001, 50.0}})
  {}

  // c(u_1, u_2; \delta) = (A + \delta - 1) A^{1 - 2\delta} \exp(-A) (u_1 u_2)^{-1} (-\ln u_1)^{\delta - 1} (-\ln u_2)^{\delta - 1}
  inline double estimate_copula_density(double u1_scalar, double u2_scalar) const 
  override {
    const double delta = this->m_params[0][0];
    const double A = term_A(u1_scalar, u2_scalar);
    return (
      (A + delta - 1.0) * std::powf(A, (1.0 - 2.0 * delta)) *
      std::exp(-A) * std::powf(u1_scalar * u2_scalar, -1.0) *
      std::powf(-std::log(u1_scalar), delta - 1.0) * 
      std::powf(-std::log(u2_scalar), delta - 1.0)
    );
  }

  // C_{2|1}(u_2 \mid u_1; \delta) = u_1^{-1} (-\ln u_1)^{\delta - 1} B \exp(-A)
  double h_conditional_prob(double u1_scalar, double u2_scalar) const override {
    const double delta = this->m_params[0][0];
    return 
      std::powf(u1_scalar, -1.0) * 
      std::powf(-std::log(u1_scalar), delta - 1.0) * 
      term_B(u1_scalar, u2_scalar) *
      std::exp(-term_A(u1_scalar, u2_scalar));
  }

  private:

  // A = \left((- \ln u_1)^{\delta} + (- \ln u_2)^{\delta}\right)^{1/\delta}
  inline double term_A(double u1_scalar, double u2_scalar) const {
    const double delta = this->m_params[0][0];
    return
      std::powf(
        (
          std::powf(-std::log(u1_scalar), delta) + 
          std::powf(-std::log(u2_scalar), delta)
        ), 
      1.0 / delta
    );
  }

  // B = \left((- \ln u_1)^{\delta} + (- \ln u_2)^{\delta}\right)^{\frac{1-\delta}{\delta}}
  inline double term_B(double u1_scalar, double u2_scalar) const {
    const double delta = this->m_params[0][0];
    return
      std::powf(
        (
          std::powf(-std::log(u1_scalar), delta) + 
          std::powf(-std::log(u2_scalar), delta)
        ), 
      (1.0 - delta) / delta
    );
  }
};
