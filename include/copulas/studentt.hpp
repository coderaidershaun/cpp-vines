#pragma once

//! Implements the bivariate Student-t copula with symmetric tail dependence.

#include <cmath>
#include <numbers>
#include <span>
#include <string>

#include <boost/math/distributions/students_t.hpp>
#include <copulas/base.hpp>


class StudentT : public Copula<2> {

  public:
  StudentT(
    std::span<const double> u1,
    std::span<const double> u2,
    double rho_init = 0.5,
    double nu_init = 2.50 // degrees of freedom
  ) 
    : Copula(u1, u2, {{{rho_init, -0.999, 0.999}, {nu_init, 2.01, 50.0}}})
  {}
  
  std::string name() const override {
    return "StudentT";
  }

  /// c(u_1, u_2) = K(1 - \rho^2)^{-1/2} \left[1 + \nu^{-1}(1 - \rho^2)^{-1}\left(\xi_1^2 - 2\rho \xi_1 \xi_2 + \xi_2^2\right)\right]^{-(\nu + 2)/2} \left[\left(1 + \nu^{-1}\xi_1^2\right)\left(1 + \nu^{-1}\xi_2^2\right)\right]^{(\nu + 1)/2}
  inline double estimate_copula_density(double u1, double u2) const override {
    const double rho = this->params()[0][0];
    const double nu = this->params()[1][0];
    const double x1 = inverse_cdf(u1);
    const double x2 = inverse_cdf(u2);
    const double r2 = rho * rho, omr2 = 1.0 - r2;
    const double q = x1 * x1 - 2.0 * rho * x1 * x2 + x2 * x2;

    return term_K() / std::sqrt(omr2)
      * std::pow(1.0 + q / (nu * omr2), -(nu + 2.0) / 2.0)
      * std::pow((1.0 + x1 * x1 / nu) * (1.0 + x2 * x2 / nu), (nu + 1.0) / 2.0);
  }

  /// C_{2|1}(u_2, u_1) = t_{\nu+1}\left(\sqrt{\frac{\nu + 1}{\nu + \left(t_{\nu}^{-1}(u_1)\right)^2}} \left[\frac{t_{\nu}^{-1}(u_2) - \rho t_{\nu}^{-1}(u_1)}{\sqrt{1 - \rho^2}}\right]\right)
  double h_conditional_prob(double u1, double u2) const override {
    const double rho = this->params()[0][0];
    const double nu = this->params()[1][0];
    const double x1 = inverse_cdf(u1);
    const double x2 = inverse_cdf(u2);

    const double arg =
      std::sqrt((nu + 1.0) / (nu + x1 * x1))
      * (x2 - rho * x1)
      / std::sqrt(1.0 - rho * rho);

    boost::math::students_t_distribution<double> dist{nu + 1.0};
    return boost::math::cdf(dist, arg);
  }

  /// \tau = \frac{2}{\pi}\arcsin(\rho)
  double kendalls_tau() const override {
    const double rho = this->params()[0][0];
    return (2.0 / std::numbers::pi) * std::asin(rho);
  }

  private:
  inline double inverse_cdf(double probability) const {
    const double nu = this->params()[1][0];
    boost::math::students_t_distribution<double> dist{nu};
    return boost::math::quantile(dist, probability);
  }

  inline double cdf(double value, double nu) const {
    boost::math::students_t_distribution<double> dist{nu};
    return boost::math::cdf(dist, value);
  }
  
  /// K = \exp\left(\log\Gamma\left(\frac{\nu}{2}\right) - 2\log\Gamma\left(\frac{\nu + 1}{2}\right) + \log\Gamma\left(\frac{\nu + 2}{2}\right)\right)
  inline double term_K() const {
    const double nu = this->params()[1][0];
    constexpr double n = 2.0;
    return std::exp(
      (n - 1.0) * std::lgammaf(nu / 2.0) -
      2.0 * std::lgammaf((nu + 1.0) / 2.0) + 
      std::lgammaf((nu + 2.0) / 2.0)
    );
  }
};
