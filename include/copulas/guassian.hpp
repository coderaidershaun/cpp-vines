#pragma once

#include <cmath>
#include <numbers>
#include <span>
#include <string>

#include <boost/math/distributions/normal.hpp>

#include <copulas/base.hpp>


class Guassian : public Copula<1> {

  public:
  Guassian(
    std::span<const double> u1,
    std::span<const double> u2,
    double rho_init = 0.5
  ) 
    : Copula(u1, u2, {{rho_init, -0.999, 0.999}})
  {}
  
  std::string name() const override {
    return "Guassian";
  }

  // c(u_1, u_2; \rho) = (1 - \rho^2)^{-1/2} \exp\left(-\frac{\rho^2 \xi_1^2 - 2\rho \xi_1 \xi_2 + \rho^2 \xi_2^2}{2(1 - \rho^2)}\right)
  inline double estimate_copula_density(double u1, double u2) const override {
    const double rho = this->params()[0][0];
    const double r2 = rho * rho, omr2 = 1.0 - r2;
    const double x1 = inverse_cdf(u1), x2 = inverse_cdf(u2);

    return 
      std::exp(-(r2 * x1 * x1 - 2.0 * rho * x1 * x2 + r2 * x2 * x2) 
      / (2.0 * omr2))
      / std::sqrt(omr2);
  }

  // C_{2|1}(u_2 \mid u_1) = \Phi\left(\frac{\Phi^{-1}(u_2) - \rho \Phi^{-1}(u_1)}{\sqrt{1 - \rho^2}}\right)
  double h_conditional_prob(double u1_scalar, double u2_scalar) const override {
    const double rho = this->params()[0][0];
    const double x1 = inverse_cdf(u1_scalar);
    const double x2 = inverse_cdf(u2_scalar);

    return cdf((x2 - rho * x1) / std::sqrt(1.0 - rho * rho));
  }

  // \tau = \frac{2}{\pi}\arcsin(\rho)
  double kendalls_tau() const {
    const double rho = this->params()[0][0];
    return (2.0 / std::numbers::pi) * std::asin(rho);
  }

  private:
  double inverse_cdf(double probability) const {
    boost::math::normal_distribution<double> normal_dist{0.0, 1.0};
    return boost::math::quantile(normal_dist, probability);
  }

  double cdf(double value) const {
    boost::math::normal_distribution<double> normal_dist{0.0, 1.0};
    return boost::math::cdf(normal_dist, value);
  }
};
