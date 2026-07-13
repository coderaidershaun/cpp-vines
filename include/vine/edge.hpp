#pragma once

//! Fits pair-copula edges and caches both conditional probability directions.
//! Bitmask sets let higher vine trees reuse conditionals by variable identity.

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
#include <vector>

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
  private:
  Mask m_conditioned_left;
  Mask m_conditioned_right;
  Mask m_conditioning_set;
  std::optional<double> m_k_tau; // used for ranking
  std::optional<CopulaFitResult> m_copula;
  std::vector<double> m_left_given_right;
  std::vector<double> m_right_given_left;

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
    std::span<const double> u_left,
    std::span<const double> u_right,
    double empirical_tau
  ) {
    if (u_left.empty()) {
      return std::unexpected(SmartError::ArrayLengthZero);
    }

    if (u_left.size() != u_right.size()) {
      return std::unexpected(SmartError::ArraysLengthMismatch);
    }

    auto copula_result = fit_optimal_copula(u_left, u_right, empirical_tau);

    if (!copula_result) {
      m_copula = std::unexpected(copula_result.error());
      return std::unexpected(copula_result.error());
    }

    m_k_tau = empirical_tau;
    m_copula = std::move(copula_result);

    m_left_given_right.resize(u_left.size());
    m_right_given_left.resize(u_left.size());

    std::visit(
      [&](const auto& copula) {
        for (usize i = 0; i < u_left.size(); i++) {
          const CondProbsH h = copula.h_conditional_prob_set(u_left[i], u_right[i]);
          m_left_given_right[i] = h.u1_given_u2;
          m_right_given_left[i] = h.u2_given_u1;
        }
      },
      m_copula->value()
    );

    return {};
  }

  bool copula_available() const {
    return m_copula.has_value() && m_copula->has_value();
  }

  Mask conditioned_left() const {
    return m_conditioned_left;
  }

  Mask conditioned_right() const {
    return m_conditioned_right;
  }

  Mask conditioning_set() const {
    return m_conditioning_set;
  }

  Mask union_set() const {
    return m_conditioned_left | m_conditioned_right | m_conditioning_set;
  }

  void update_k_tau(double k_tau) {
    m_k_tau = k_tau;
  }

  std::optional<double> k_tau() const {
    return m_k_tau;
  }

  std::expected<std::span<const double>, SmartError> left_given_right() const {
    if (!copula_available()) {
      return std::unexpected(SmartError::MissingConditionalValues);
    }

    return std::span<const double>(m_left_given_right);
  }

  std::expected<std::span<const double>, SmartError> right_given_left() const {
    if (!copula_available()) {
      return std::unexpected(SmartError::MissingConditionalValues);
    }

    return std::span<const double>(m_right_given_left);
  }

  /// For left, right | conditioning
  std::expected<std::span<const double>, SmartError> conditional_values(
    Mask target,
    Mask given
  ) const {
    if (!copula_available()) {
      return std::unexpected(SmartError::MissingConditionalValues);
    }

    const Mask left_given = m_conditioned_right | m_conditioning_set;
    const Mask right_given = m_conditioned_left | m_conditioning_set;

    if (target == m_conditioned_left && given == left_given) {
      return left_given_right();
    }

    if (target == m_conditioned_right && given == right_given) {
      return right_given_left();
    }

    return std::unexpected(SmartError::MissingConditionalValues);
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

      for (usize i = 0; i < results.size(); i++) {
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
