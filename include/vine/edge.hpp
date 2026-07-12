#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <expected>
#include <future>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <utility>
#include <variant>

#include <copulas/base.hpp>
#include <copulas/clayton.hpp>
#include <copulas/guassian.hpp>
#include <copulas/gumbel.hpp>
#include <copulas/studentt.hpp>
#include <errors.hpp>
#include <types.hpp>


constexpr Mask bitmask(int i) {
  return 1ULL << i;
}

constexpr Mask bitmask_empty() {
  return 0ULL;
}

constexpr usize asset_index_from_bitmask(Mask mask) {
  return std::countr_zero(mask); 
}

using FittedCopula = std::variant<Clayton, Guassian, Gumbel, StudentT>;

using CopulaFitResult = std::expected<FittedCopula, SmartError>;

/// Using bitwise operations to manage conditioned and
/// conditioning sets to help with low latency set comparisons.
///
/// For example {2, 3}|{1} = 23|1 = {00100, 03000}|{00010}.
/// So A_edge intersection of B_edge = A.union() & B.union().
struct Edge {
  Mask m_conditioned_left;
  Mask m_conditioned_right;
  Mask m_conditioning_set;
  std::optional<double> m_k_tau;
  std::optional<CopulaFitResult> m_copula = std::nullopt;

  public:
  Edge(
    Mask conditioned_left,
    Mask conditioned_right,
    Mask conditioning_set,
    std::optional<double> k_tau = std::nullopt
  )
    : m_conditioned_left(conditioned_left),
      m_conditioned_right(conditioned_right),
      m_conditioning_set(conditioning_set),
      m_k_tau(k_tau)
  {}

  std::expected<void, SmartError> fit(
    std::span<const double> u1,
    std::span<const double> u2,
    double k_tau_init
  ) {
    auto copula = fit_optimal_copula(u1, u2, k_tau_init);
    if (!copula) {
      m_k_tau = 0.0;
      m_copula = std::unexpected(copula.error());
      return std::unexpected(copula.error());
    }

    m_k_tau = std::visit(
      [](const auto& fitted_copula) {
        return fitted_copula.kendalls_tau();
      },
      *copula
    );
    m_copula = std::move(copula);
    return {};
  }

  bool copula_available() const {
    return m_copula != std::nullopt;
  }

  Mask conditioned_left() {
    return m_conditioned_left;
  }

  Mask conditioned_right() {
    return m_conditioned_right;
  }

  Mask conditioning_set() {
    return m_conditioning_set;
  }

  Mask union_set() const {
    return m_conditioned_left | m_conditioned_right | m_conditioning_set;
  }

  void update_k_tau(double k_tau) {
    m_k_tau = k_tau;
  }

  std::optional<double> k_tau() const {
    if (!m_k_tau) return m_k_tau;
    return std::abs(*m_k_tau);
  }

  static CopulaFitResult fit_optimal_copula(
    std::span<const double> u1,
    std::span<const double> u2,
    double k_tau_init
  ) {
    if (u1.empty()) {
      return std::unexpected(SmartError::ArrayLengthZero);
    }

    if (u1.size() != u2.size()) {
      return std::unexpected(SmartError::ArraysLengthMismatch);
    }

    try {
      const double rho_init = std::sin(std::numbers::pi * k_tau_init / 2.0);

      Clayton clayton(u1, u2, Clayton::alpha_from_kendalls_tau(k_tau_init));
      Guassian gaussian(u1, u2, rho_init);
      Gumbel gumbel(u1, u2, Gumbel::delta_from_kendalls_tau(k_tau_init));
      StudentT student_t(u1, u2, rho_init);

      std::array<std::future<OptimiserResults>, 4> futures{
        std::async(std::launch::async, [&] { return clayton.fit(); }),
        std::async(std::launch::async, [&] { return gaussian.fit(); }),
        std::async(std::launch::async, [&] { return gumbel.fit(); }),
        std::async(std::launch::async, [&] { return student_t.fit(); })
      };

      std::array<OptimiserResults, 4> results{};

      for (usize i = 0; i < futures.size(); i++) {
        try {
          results[i] = futures[i].get();
        } catch (...) {
          results[i].result = Result::Failure;
        }
      }

      usize best_index = results.size();

      double best_aic = std::numeric_limits<double>::infinity();

      for (usize i = 0; i < results.size(); ++i) {
        if (
          results[i].result == Result::Success &&
          std::isfinite(results[i].aic) &&
          results[i].aic < best_aic
        ) {
          best_index = i;
          best_aic = results[i].aic;
        }
      }

      switch (best_index) {
        case 0:
          return clayton;
        case 1:
          return gaussian;
        case 2:
          return gumbel;
        case 3:
          return student_t;
        default:
          return std::unexpected(SmartError::CopulaFitsFailed);
      }
    } catch (...) {
      return std::unexpected(SmartError::CopulaFitsFailed);
    }
  }
};
