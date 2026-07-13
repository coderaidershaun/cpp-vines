//! cmake --build build --target stats
//! ./build/stats

#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <vector>

#include <asset.hpp>


int main() {

  Asset test_asset(5);

  const std::size_t n = 5;
  std::array<double, n> prices{1.1, 2.2, 3.3, 4.4, 5.5};

  for (double price : prices) {
    test_asset.push_price(price);
  }

  std::cout << "Ln Returns:" << std::endl;
  for (double ln_ret : test_asset.ln_returns()) {
    std::cout << ln_ret << " ";
  }
  std::cout << std::endl;
  
  std::cout << "\nU values:" << std::endl;
  std::vector<double> u_values = test_asset.u_values();
  for (double value : u_values) {
    std::cout << value << std::endl;
  }

  assert(u_values[0] == 0.7);
  assert(u_values[3] == 0.1);
  assert(u_values.size() == 4);

  return 0;
}
