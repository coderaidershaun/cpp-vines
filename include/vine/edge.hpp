#pragma once

//! Fits pair-copula edges and caches both conditional probability directions.
//! Bitmask sets let higher vine trees reuse conditionals by variable identity.

#include <bit>
#include <cmath>
#include <expected>
#include <future>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include <copulas/base.hpp>
#include <copulas/clayton.hpp>
#include <copulas/guassian.hpp>
#include <copulas/gumbel.hpp>
#include <copulas/independence.hpp>
#include <copulas/survival.hpp>
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

enum class SelectionCriterion {
  Aic,
  Bic
};

using FittedCopula = std::variant<
  Independence,
  Clayton,
  SurvivalCopula<Clayton>,
  Guassian,
  Gumbel,
  SurvivalCopula<Gumbel>,
  StudentT
>;

using CopulaFitResult = std::expected<FittedCopula, SmartError>;

/// Bitmasks keep higher-tree set comparisons allocation-free.
struct Edge {
  private:
  Mask m_conditioned_left;
  Mask m_conditioned_right;
  Mask m_conditioning_set;
  std::optional<double> m_k_tau; // used for ranking
  std::optional<FittedCopula> m_copula;
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
    double empirical_tau,
    SelectionCriterion criterion = SelectionCriterion::Bic
  ) {
    if (u_left.empty()) {
      return std::unexpected(SmartError::ArrayLengthZero);
    }

    if (u_left.size() != u_right.size()) {
      return std::unexpected(SmartError::ArraysLengthMismatch);
    }

    auto copula_result = fit_optimal_copula(
      u_left,
      u_right,
      empirical_tau,
      criterion
    );

    if (!copula_result) {
      m_copula.reset();
      return std::unexpected(copula_result.error());
    }

    m_k_tau = empirical_tau;
    m_copula = std::move(*copula_result);

    m_left_given_right.resize(u_left.size());
    m_right_given_left.resize(u_left.size());

    std::visit(
      [&](const auto& copula) {
        for (usize i = 0; i < u_left.size(); i++) {
          const CondProbsH h = copula.h_conditional_prob_set(
            u_left[i],
            u_right[i]
          );
          m_left_given_right[i] = h.u1_given_u2;
          m_right_given_left[i] = h.u2_given_u1;
        }
      },
      *m_copula
    );

    return {};
  }

  bool copula_available() const {
    return m_copula.has_value();
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

  std::optional<double> k_tau() const {
    return m_k_tau;
  }

  std::optional<std::string> copula_name() const {
    if (!copula_available()) return std::nullopt;

    return std::visit(
      [](const auto& copula) { return copula.name(); },
      *m_copula
    );
  }

  std::optional<usize> parameter_count() const {
    if (!copula_available()) return std::nullopt;

    return std::visit(
      [](const auto& copula) { return copula.parameter_count(); },
      *m_copula
    );
  }

  std::optional<std::vector<double>> parameters() const {
    if (!copula_available()) return std::nullopt;

    return std::visit(
      [](const auto& copula) {
        std::vector<double> values;
        values.reserve(copula.parameter_count());

        for (const auto& parameter : copula.params()) {
          values.emplace_back(parameter[0]);
        }

        return values;
      },
      *m_copula
    );
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
    double k_tau_init,
    SelectionCriterion criterion = SelectionCriterion::Bic
  ) {
    if (u1.empty()) {
      return std::unexpected(SmartError::ArrayLengthZero);
    }

    if (u1.size() != u2.size()) {
      return std::unexpected(SmartError::ArraysLengthMismatch);
    }

    struct Candidate {
      FittedCopula copula;
      OptimiserResults result;
    };

    constexpr double criterion_tolerance = 1e-10;
    std::vector<Candidate> candidates;
    candidates.reserve(7);

    Independence independence(u1, u2);
    const OptimiserResults independence_result = independence.fit();
    candidates.emplace_back(Candidate{
      .copula = std::move(independence),
      .result = independence_result
    });

    const double rho_init = std::sin(std::numbers::pi * k_tau_init / 2.0);

    std::vector<std::future<std::optional<Candidate>>> fits;
    fits.reserve(6);

    const auto start_fit = [&fits](auto make_copula) {
      try {
        fits.emplace_back(std::async(
          std::launch::async,
          [make_copula = std::move(make_copula)]() mutable
            -> std::optional<Candidate>
          {
            try {
              auto copula = make_copula();
              const OptimiserResults result = copula.fit();
              return Candidate{
                .copula = std::move(copula),
                .result = result
              };
            } catch (...) {
              return std::nullopt;
            }
          }
        ));
      } catch (const std::system_error&) {}
    };

    if (k_tau_init > 0.0) {
      const double clayton_init = Clayton::alpha_from_kendalls_tau(k_tau_init);
      const double gumbel_init = Gumbel::delta_from_kendalls_tau(k_tau_init);
      start_fit([=] { return Clayton(u1, u2, clayton_init); });
      start_fit([=] { return SurvivalCopula<Clayton>(u1, u2, clayton_init); });
      start_fit([=] { return Gumbel(u1, u2, gumbel_init); });
      start_fit([=] { return SurvivalCopula<Gumbel>(u1, u2, gumbel_init); });
    }

    start_fit([=] { return Guassian(u1, u2, rho_init); });
    start_fit([=] { return StudentT(u1, u2, rho_init); });

    for (auto& fit : fits) {
      if (auto candidate = fit.get()) {
        candidates.emplace_back(std::move(*candidate));
      }
    }

    std::optional<usize> best_index;
    double best_score = std::numeric_limits<double>::infinity();
    usize best_parameter_count = std::numeric_limits<usize>::max();

    for (usize i = 0; i < candidates.size(); i++) {
      const Candidate& candidate = candidates[i];
      const double score = criterion == SelectionCriterion::Aic
        ? candidate.result.aic
        : candidate.result.bic;

      if (
        candidate.result.result != Result::Success ||
        !std::isfinite(score)
      ) {
        continue;
      }

      const bool has_lower_score = score < best_score - criterion_tolerance;
      const bool scores_tied = std::abs(score - best_score) <= criterion_tolerance;
      const usize parameter_count = std::visit(
        [](const auto& copula) { return copula.parameter_count(); },
        candidate.copula
      );
      const bool has_lower_complexity =
        scores_tied && parameter_count < best_parameter_count;

      if (!best_index || has_lower_score || has_lower_complexity) {
        best_index = i;
        best_score = score;
        best_parameter_count = parameter_count;
      }
    }

    if (!best_index) {
      return std::unexpected(SmartError::CopulaFitsFailed);
    }

    return std::move(candidates[*best_index].copula);
  }
};
