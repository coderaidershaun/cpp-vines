//! Empirical cumulative distribution function construction and lookup

#pragma once

#include <algorithm>
#include <cassert>
#include <iterator>
#include <vector>

#include <types.hpp>


class Ecdf {
  std::vector<double> values_sorted{};
  std::vector<double> cdf{};
  usize capacity;

  public:
  Ecdf(usize capacity=20'000) 
    : capacity(capacity)
  {
    assert(capacity > 1);
    values_sorted.reserve(capacity + 1);
    cdf.reserve(capacity);
  }

  double lookup_value(double value) const {
    auto it = std::upper_bound(
      values_sorted.begin(),
      values_sorted.end(),
      value
    );

    if (it == values_sorted.begin()) {
      return 0.0;
    }

    usize index = static_cast<usize>(
      std::distance(values_sorted.begin(), it) - 1
    );

    return cdf[index];
  }

  void push(double value) {
    insert_value(value);
    trim_values_to_capacity();
    cdf.resize(values_sorted.size());
    update_cdf();
  }

  private:
  inline void insert_value(double value) {

    // uses binary search algorithm
    auto it = std::lower_bound(
      values_sorted.begin(),
      values_sorted.end(),
      value
    );

    values_sorted.insert(it, value);
  }

  inline void trim_values_to_capacity() {
    if (values_sorted.size() > capacity) {
      usize midpoint_i = values_sorted.size() / 2;
      values_sorted.erase(values_sorted.begin() + midpoint_i);
    }
  }

  inline void update_cdf() {
    usize n = values_sorted.size();
    for (usize i=0; i<n; i++) {
      cdf[i] = (0.5 + static_cast<double>(i) * 1.0) / (static_cast<double>(n) + 1.0);
    }
  }
};
