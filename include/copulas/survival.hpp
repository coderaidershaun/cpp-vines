#pragma once

#include <algorithm>
#include <string>


template <typename BaseCopula>
class SurvivalCopula : public BaseCopula {

  public:
  using BaseCopula::BaseCopula;

  std::string name() const override {
    return "Survival " + BaseCopula::name();
  }

  double cdf(double u1_scalar, double u2_scalar) const {
    return std::clamp(
      u1_scalar + u2_scalar - 1.0 +
        BaseCopula::cdf(1.0 - u1_scalar, 1.0 - u2_scalar),
      0.0,
      1.0
    );
  }

  double estimate_copula_density(
    double u1_scalar,
    double u2_scalar
  ) const override {
    return BaseCopula::estimate_copula_density(
      1.0 - u1_scalar,
      1.0 - u2_scalar
    );
  }

  double h_conditional_prob(
    double u1_scalar,
    double u2_scalar
  ) const override {
    if (u2_scalar <= 0.0) return 0.0;
    if (u2_scalar >= 1.0) return 1.0;

    return std::clamp(
      1.0 - BaseCopula::h_conditional_prob(
        1.0 - u1_scalar,
        1.0 - u2_scalar
      ),
      0.0,
      1.0
    );
  }
};
