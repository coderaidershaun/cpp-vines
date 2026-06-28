//! cmake --build build --target ecdf
//! ./build/ecdf

#include <cassert>
#include <iostream>
#include <optional>
#include <ecdf.hpp>

int main() {

  Ecdf ecdf = Ecdf(10);

  ecdf.push(0.1); // 0.5 / (5 + 1) = 0.08333
  ecdf.push(0.2); // 1.5 / (5 + 1) = 0.25000
  ecdf.push(0.3); // 2.5 / (5 + 1) = 0.41667
  ecdf.push(0.4); // 3.5 / (5 + 1) = 0.58333
  ecdf.push(0.5); // 4.5 / (5 + 1) = 0.75000

  std::optional<double> lookup = ecdf.lookup_value(0.5);
  assert(ecdf.lookup_value(0.2) == 0.25);
  assert(ecdf.lookup_value(0.5) == 0.75);
  assert(ecdf.lookup_value(0.5) == 0.75);

  return 0;
}
