//! cmake --build build --target optimiser
//! ./build/optimiser

#include <array>
#include <cassert>
#include <cmath>

#include <optimiser.hpp>
#include <types.hpp>


double objective_f(const std::array<double, 2>& params) {
  const double x = params[0] - 1.0;
  const double y = params[1] + 2.0;
  return x * x + y * y + 3.0;
}

int main() {
  ParamBounds<2> params = {{{0.5, -5.0, 5.0}, {-0.5, -5.0, 5.0}}};

  NelderMeadSimplex<2> optimiser(objective_f, params);
  const OptimiserResults results = optimiser.fit();
  assert(results.result == Result::Success);
  assert(std::abs(results.f_opt_ll - (-3.0)) < 1e-8);
  assert(std::abs(results.aic - 10.0) < 1e-8);
  assert(results.number_it > 0.0);
  assert(results.number_it < 10'000.0);
  assert(results.number_fev >= 3.0);

  const OptimiserResults& stored_results = optimiser.results();
  assert(stored_results.result == results.result);
  assert(stored_results.f_opt_ll == results.f_opt_ll);

  return 0;
}
