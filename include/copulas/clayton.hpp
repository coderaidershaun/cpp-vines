#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <copulas/base.hpp>


class Clayton : public Copula<1> {

  public:
  Clayton(
    const std::vector<double>& u1,
    const std::vector<double>& u2,
    double alpha_init = 0.5
  ) 
    : Copula(u1, u2, {{alpha_init, -0.99, 50.0}})
  {}

  std::string name() const override {
    return "Clayton";
  }
  
  // c(u_1, u_2; \alpha) = (\alpha + 1) \left(u_1^{-\alpha} + u_2^{-\alpha} - 1\right)^{-2 - \frac{1}{\alpha}} u_1^{-\alpha - 1} u_2^{-\alpha - 1}
  inline double estimate_copula_density(double u1_scalar, double u2_scalar) const 
  override {
    const double alpha = this->params()[0][0];
    return
      (alpha + 1.0) *
      std::powf(
        (std::powf(u1_scalar, -alpha) + std::powf(u2_scalar, -alpha) - 1.0),
        -2.0 - (1.0 / alpha)
      ) *
      std::powf(u1_scalar, -alpha - 1.0) * std::powf(u2_scalar, -alpha - 1.0);
  }

  // C_{2|1}(u_2 \mid u_1; \alpha) = u_1^{-(1+\alpha)} \left(u_1^{-\alpha} + u_2^{-\alpha} - 1\right)^{-\frac{1+\alpha}{\alpha}}
  double h_conditional_prob(double u1_scalar, double u2_scalar) const override {
    const double alpha = this->params()[0][0];
    return 
      std::powf(u1_scalar, -(1.0 + alpha)) *
      std::powf(
        (std::powf(u1_scalar, -alpha) + std::powf(u2_scalar, -alpha) - 1.0),
        (-(1.0 + alpha) / alpha)
      );
  }
};
