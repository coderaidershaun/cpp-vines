//! cmake --build build --target series
//! ./build/series

#include <cassert>
#include <iostream>
#include <series.hpp>

int main() {

  FixedSeries<double> series(4, 2);

  series.push(1.0);
  series.push(2.0);
  series.push(3.0);
  series.push(4.0);
  series.push(5.0);
  series.push(6.0);
  series.push(7.0);
  series.push(8.0);
  series.push(9.0);
  series.push(10.0);

  assert(series.sum() == 34.0);

  std::cout << "Series raw: ";
  for (double item: series.data()) {
    std::cout << item << " ";
  }
  std::cout << std::endl;
  
  std::cout << "Series sum: " << series.sum() << std::endl;
  std::cout << "Series mean: " << series.mean().value() << std::endl;
  
  return 0;
}
