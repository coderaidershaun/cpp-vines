#pragma once

#include <cstddef>

using usize = std::size_t;

/// [
///   [p1, lb, ub] Param1, Lower Bound, Upper Bound
///   [p2, lb, ub] Param2, Lower Bound, Upper Bound
///   ... etc
/// ]
template <usize N>
using ParamBounds = std::array<std::array<double, 3>, N>;
