//! cmake --build build --target csv
//! ./build/csv

#include <cassert>
#include <iostream>

#include <csv.hpp>
#include <errors.hpp>


int main() {

  auto values_res = read_csv("prices.csv", 0);

  if (values_res) {
    assert(values_res.value()[0] == 0.031124561);
    assert(values_res.value()[1] == 0.010075686);
  } else {
    std::cout << error_to_string(values_res.error()) << std::endl;
    assert(false);
  }

  return 0;
}
