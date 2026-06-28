#pragma once

#include <expected>
#include <vector>

#include <errors.hpp>


const std::expected<double, SmartError> kendals_tau(
  const std::vector<double>& x,
  const std::vector<double>& y
) {
  if (x.size() == 0) {
    return std::unexpected(SmartError::ArrayLengthZero);
  } else if (x.size() != y.size()) {
    return std::unexpected(SmartError::ArraysLengthMismatch);
  }

  int n = x.size();
  int n_f64 = static_cast<double>(n);
  int nc{}; // number of concordant pairs
  int nd{}; // number of disconcordant pairs
  for (int i=0; i<n; i++) {
    for (int j=i + 1; j<n; j++) {
      double concordance = (x[j] - x[i]) * (y[j] - y[i]);
      if (concordance > 0.0) {
        nc += 1;
      } else if (concordance < 0.0) {
        nd += 1;
      }
    }
  }

  if ((nc - nd) == 0) {
    return std::unexpected(SmartError::DivisionByZero);
  }

  double denominator = 0.5 * n_f64 * (n_f64 - 1.0);
  return static_cast<double>(nc - nd) / denominator;
}
