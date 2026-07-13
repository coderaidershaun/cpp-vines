#pragma once

//! Builds a regular vine one tree at a time from fitted conditional probabilities.
//! Prior edges become nodes; bitmasks enforce proximity and |tau| ranks connections.

#include <algorithm>
#include <bit>
#include <cmath>
#include <expected>
#include <limits>
#include <optional>
#include <ostream>
#include <set>
#include <span>
#include <utility>
#include <vector>

#include <concordance.hpp>
#include <types.hpp>
#include <vine/edge.hpp>

using Marginals = std::vector<std::span<const double>>;

using Edges = std::vector<Edge>;

enum class Method {
  Cmpi,
  CmpiZscore,
  Standard
};

/// Views remain valid until the fitted vine is refitted or destroyed.
struct FinalProbabilities {
  std::span<const double> left_given_right;
  std::span<const double> right_given_left;
};

/// Values contain probabilities, cumulative CMPI, or normalized CMPI according to method.
struct FinalResults {
  Method method;
  std::vector<double> left_given_right;
  std::vector<double> right_given_left;
};

struct Vine {
  private:
  struct Candidate {
    usize node_a;
    usize node_b;
    std::span<const double> u_left;
    std::span<const double> u_right;
    double tau;
    Edge edge;
  };

  struct CandidateSets {
    Mask conditioned_left;
    Mask conditioned_right;
    Mask conditioning;
  };

  using Candidates = std::vector<Candidate>;

  Marginals m_marginals;
  usize m_max_nodes;
  Method m_method;
  SelectionCriterion m_selection_criterion;

  /// Higher-tree copulas retain spans into conditional values owned by earlier trees.
  std::vector<Edges> m_trees;

  public:
  Vine(
    Marginals marginals,
    usize max_nodes = 6,
    Method method = Method::Cmpi,
    SelectionCriterion selection_criterion = SelectionCriterion::Bic
  )
    : m_marginals(std::move(marginals)),
      m_max_nodes(std::min(m_marginals.size(), max_nodes)),
      m_method(method),
      m_selection_criterion(selection_criterion)
  {}

  /// Builds lower trees first because each higher tree consumes their conditionals
  std::expected<void, SmartError> fit() {
    if (m_max_nodes < 2) {
      return std::unexpected(SmartError::ArrayLengthZero);
    }

    auto initial_candidates = build_initial_candidates();
    if (!initial_candidates) return std::unexpected(initial_candidates.error());

    auto initial_tree = build_spanning_tree(
      std::move(*initial_candidates),
      m_max_nodes - 1,
      SmartError::ArrayLengthZero
    );

    if (!initial_tree) return std::unexpected(initial_tree.error());

    m_trees.clear();
    m_trees.reserve(m_max_nodes - 1);
    m_trees.emplace_back(std::move(*initial_tree));

    for (usize tree_number = 2; tree_number < m_max_nodes; tree_number++) {
      auto next_tree = build_next_tree(m_trees.back(), tree_number);
      if (!next_tree) return std::unexpected(next_tree.error());

      m_trees.emplace_back(std::move(*next_tree));
    }

    return {};
  }

  const std::vector<Edges>& trees() const {
    return m_trees;
  }

  SelectionCriterion selection_criterion() const {
    return m_selection_criterion;
  }

  /// Exposes final-edge views to avoid copying fitted conditional probabilities
  std::expected<FinalProbabilities, SmartError> final_probabilities() const {
    if (
      m_max_nodes < 2 ||
      m_trees.size() != m_max_nodes - 1 ||
      m_trees.back().size() != 1
    ) {
      return std::unexpected(SmartError::CopulaNotFitted);
    }

    const Edge& final_edge = m_trees.back().front();
    auto left_given_right = final_edge.left_given_right();
    if (!left_given_right) {
      return std::unexpected(left_given_right.error());
    }

    auto right_given_left = final_edge.right_given_left();
    if (!right_given_left) {
      return std::unexpected(right_given_left.error());
    }

    return FinalProbabilities{
      .left_given_right = *left_given_right,
      .right_given_left = *right_given_left
    };
  }

  /// Owns results so CMPI transforms cannot mutate the fitted edge state
  std::expected<FinalResults, SmartError> final_results() const {
    auto probabilities = final_probabilities();
    if (!probabilities) {
      return std::unexpected(probabilities.error());
    }

    FinalResults results{
      .method = m_method,
      .left_given_right = {
        probabilities->left_given_right.begin(),
        probabilities->left_given_right.end()
      },
      .right_given_left = {
        probabilities->right_given_left.begin(),
        probabilities->right_given_left.end()
      }
    };

    if (m_method != Method::Standard) {
      accumulate_cmpi(results.left_given_right);
      accumulate_cmpi(results.right_given_left);
    }

    if (m_method == Method::CmpiZscore) {
      auto left_normalized = normalize(results.left_given_right);
      if (!left_normalized) {
        return std::unexpected(left_normalized.error());
      }

      auto right_normalized = normalize(results.right_given_left);
      if (!right_normalized) {
        return std::unexpected(right_normalized.error());
      }
    }

    return results;
  }

  private:

  static void accumulate_cmpi(std::vector<double>& values) {
    double cumulative = 0.0;

    for (double& value : values) {
      cumulative += value - 0.5;
      value = cumulative;
    }
  }

  static std::expected<void, SmartError> normalize(
    std::vector<double>& values
  ) {
    double sum = 0.0;
    for (double value : values) sum += value;
    const double mean = sum / static_cast<double>(values.size());

    double squared_deviations = 0.0;
    for (double value : values) {
      const double deviation = value - mean;
      squared_deviations += deviation * deviation;
    }

    const double standard_deviation = std::sqrt(
      squared_deviations / static_cast<double>(values.size())
    );

    if (standard_deviation == 0.0) {
      return std::unexpected(SmartError::DivisionByZero);
    }

    for (double& value : values) {
      value = (value - mean) / standard_deviation;
    }

    return {};
  }

  /// Considers every marginal pair so dependence can determine the first topology
  std::expected<Candidates, SmartError> build_initial_candidates() const {
    Candidates candidates{};
    const usize marginal_count = m_marginals.size();
    candidates.reserve(marginal_count * (marginal_count - 1) / 2);

    for (usize left = 0; left < marginal_count; left++) {
      for (usize right = left + 1; right < marginal_count; right++) {
        auto tau = kendalls_tau(m_marginals[left], m_marginals[right]);
        if (!tau) return std::unexpected(tau.error());

        candidates.emplace_back(
          Candidate{
            .node_a = left,
            .node_b = right,
            .u_left = m_marginals[left],
            .u_right = m_marginals[right],
            .tau = *tau,
            .edge = Edge(
              bitmask(static_cast<int>(left)),
              bitmask(static_cast<int>(right)),
              bitmask_empty(),
              *tau
            )
          }
        );
      }
    }

    return candidates;
  }

  /// Removes one edge per level to preserve the regular-vine tree shape
  std::expected<Edges, SmartError> build_next_tree(
    const Edges& prior_tree,
    usize tree_number
  ) const {
    if (prior_tree.size() < 2) {
      return std::unexpected(SmartError::DisconnectedGraph);
    }

    auto candidates = build_next_candidates(prior_tree, tree_number);
    if (!candidates) return std::unexpected(candidates.error());

    return build_spanning_tree(
      std::move(*candidates),
      prior_tree.size() - 1,
      SmartError::DisconnectedGraph
    );
  }

  /// Tests every prior-edge pair because proximity, not order, decides eligibility
  std::expected<Candidates, SmartError> build_next_candidates(
    const Edges& prior_tree,
    usize tree_number
  ) const {
    Candidates candidates{};
    candidates.reserve(prior_tree.size() * (prior_tree.size() - 1) / 2);

    for (usize node_a = 0; node_a < prior_tree.size(); node_a++) {
      for (usize node_b = node_a + 1; node_b < prior_tree.size(); node_b++) {
        auto candidate = make_next_candidate(
          prior_tree[node_a],
          prior_tree[node_b],
          node_a,
          node_b,
          tree_number
        );

        if (!candidate) return std::unexpected(candidate.error());
        if (*candidate) candidates.emplace_back(std::move(**candidate));
      }
    }

    return candidates;
  }

  /// Reuses prior conditionals to model dependence given the shared variables
  static std::expected<std::optional<Candidate>, SmartError> make_next_candidate(
    const Edge& edge_a,
    const Edge& edge_b,
    usize node_a,
    usize node_b,
    usize tree_number
  ) {
    const auto sets = candidate_sets(edge_a, edge_b, tree_number);
    if (!sets) return std::optional<Candidate>{};

    auto u_left = conditional_input(
      edge_a,
      edge_b,
      sets->conditioned_left,
      sets->conditioning
    );

    if (!u_left) return std::unexpected(u_left.error());

    auto u_right = conditional_input(
      edge_a,
      edge_b,
      sets->conditioned_right,
      sets->conditioning
    );

    if (!u_right) return std::unexpected(u_right.error());

    auto tau = kendalls_tau(*u_left, *u_right);
    if (!tau) return std::unexpected(tau.error());

    return Candidate{
      .node_a = node_a,
      .node_b = node_b,
      .u_left = *u_left,
      .u_right = *u_right,
      .tau = *tau,
      .edge = Edge(
        sets->conditioned_left,
        sets->conditioned_right,
        sets->conditioning,
        *tau
      )
    };
  }

  /// Enforces proximity: parents must differ by one variable around a shared set
  static std::optional<CandidateSets> candidate_sets(
    const Edge& edge_a,
    const Edge& edge_b,
    usize tree_number
  ) {
    const Mask union_a = edge_a.union_set();
    const Mask union_b = edge_b.union_set();
    const Mask conditioning = union_a & union_b;
    Mask conditioned_left = union_a & ~union_b;
    Mask conditioned_right = union_b & ~union_a;

    if (count_bits(conditioning) != tree_number - 1) return std::nullopt;
    if (!is_singleton(conditioned_left)) return std::nullopt;
    if (!is_singleton(conditioned_right)) return std::nullopt;

    if (conditioned_left > conditioned_right) {
      std::swap(conditioned_left, conditioned_right);
    }

    return CandidateSets{
      .conditioned_left = conditioned_left,
      .conditioned_right = conditioned_right,
      .conditioning = conditioning
    };
  }

  /// Selects by variable set because left/right orientation may change by level
  static std::expected<std::span<const double>, SmartError> conditional_input(
    const Edge& edge_a,
    const Edge& edge_b,
    Mask target,
    Mask conditioning
  ) {
    const Mask required_union = target | conditioning;

    if (edge_a.union_set() == required_union) {
      return edge_a.conditional_values(target, conditioning);
    }

    if (edge_b.union_set() == required_union) {
      return edge_b.conditional_values(target, conditioning);
    }

    return std::unexpected(SmartError::MissingConditionalValues);
  }

  /// Uses Prim-style growth to favor strong dependence without forming cycles
  std::expected<Edges, SmartError> build_spanning_tree(
    Candidates candidates,
    usize target_edges,
    SmartError empty_candidates_error
  ) const {
    if (candidates.empty()) {
      return std::unexpected(empty_candidates_error);
    }

    sort_candidates_by_tau(candidates);

    Edges tree{};
    tree.reserve(target_edges);

    std::set<usize> connected_nodes{};

    auto first_edge = fit_candidate(candidates.front(), connected_nodes, tree);
    if (!first_edge) return std::unexpected(first_edge.error());

    while (tree.size() < target_edges) {
      auto candidate = std::find_if(
        candidates.begin(),
        candidates.end(),
        [&](const Candidate& current) {
          return connects_to_tree(current, connected_nodes);
        }
      );

      if (candidate == candidates.end()) {
        return std::unexpected(SmartError::DisconnectedGraph);
      }

      auto next_edge = fit_candidate(*candidate, connected_nodes, tree);
      if (!next_edge) return std::unexpected(next_edge.error());
    }

    sort_edges_by_tau(tree);
    return tree;
  }

  /// Fits only selected links because fitting every candidate copula is costly
  std::expected<void, SmartError> fit_candidate(
    Candidate& candidate,
    std::set<usize>& connected_nodes,
    Edges& tree
  ) const {
    auto fit = candidate.edge.fit(
      candidate.u_left,
      candidate.u_right,
      candidate.tau,
      m_selection_criterion
    );

    if (!fit) return std::unexpected(fit.error());

    connected_nodes.insert(candidate.node_a);
    connected_nodes.insert(candidate.node_b);
    tree.emplace_back(std::move(candidate.edge));

    return {};
  }

  /// Requires one new endpoint so each step connects without closing a cycle
  static bool connects_to_tree(
    const Candidate& candidate,
    const std::set<usize>& connected_nodes
  ) {
    const bool a_connected = connected_nodes.contains(candidate.node_a);
    const bool b_connected = connected_nodes.contains(candidate.node_b);
    return a_connected != b_connected;
  }

  /// Uses magnitude because either dependence direction is equally informative
  static void sort_candidates_by_tau(Candidates& candidates) {
    std::sort(
      candidates.begin(),
      candidates.end(),
      [](const Candidate& x, const Candidate& y) {
        return std::abs(x.tau) > std::abs(y.tau);
      }
    );
  }

  static void sort_edges_by_tau(Edges& edges) {
    std::sort(edges.begin(), edges.end(), [](const Edge& x, const Edge& y) {
      const auto left_tau = x.k_tau();
      const auto right_tau = y.k_tau();

      if (!left_tau) return false;
      if (!right_tau) return true;

      return std::abs(*left_tau) > std::abs(*right_tau);
    });
  }

  static constexpr usize count_bits(Mask mask) {
    return static_cast<usize>(std::popcount(mask));
  }

  static constexpr bool is_singleton(Mask mask) {
    return count_bits(mask) == 1;
  }
};

/// Decodes bitmasks to compact, one-based regular-vine notation
inline std::ostream& operator<<(std::ostream& output, const Vine& vine) {
  const auto write_nodes = [&output](Mask nodes) {
    for (usize node = 0; node < std::numeric_limits<Mask>::digits; node++) {
      if ((nodes & bitmask(static_cast<int>(node))) != 0) {
        output << node + 1;
      }
    }
  };

  const auto write_edges = [&output](
    const Edges& edges,
    const auto& write_edge
  ) {
    for (usize edge_index = 0; edge_index < edges.size(); edge_index++) {
      if (edge_index > 0) output << ", ";
      write_edge(edges[edge_index]);
    }
  };

  const auto& trees = vine.trees();

  for (usize tree_index = 0; tree_index < trees.size(); tree_index++) {
    if (tree_index > 0) output << '\n';

    output << "T" << tree_index + 1 << ": ";

    const auto& tree = trees[tree_index];

    write_edges(tree, [&](const Edge& edge) {
      write_nodes(edge.conditioned_left());
      write_nodes(edge.conditioned_right());

      if (edge.conditioning_set() != bitmask_empty()) {
        output << '|';
        write_nodes(edge.conditioning_set());
      }
    });

    output << "\n  copulas: ";
    write_edges(tree, [&](const Edge& edge) {
      output << edge.copula_name().value_or("Unavailable");
    });

    output << "\n  params: [";
    write_edges(tree, [&](const Edge& edge) {
      output << '[';
      const auto parameters = edge.parameters();
      if (parameters) {
        for (
          usize parameter_index = 0;
          parameter_index < parameters->size();
          parameter_index++
        ) {
          if (parameter_index > 0) output << ", ";
          output << (*parameters)[parameter_index];
        }
      }
      output << ']';
    });
    output << ']';

    output << "\n  tau: [";
    write_edges(tree, [&](const Edge& edge) {
      const auto tau = edge.k_tau();
      if (tau) {
        output << *tau;
      } else {
        output << "Unavailable";
      }
    });
    output << ']';
  }

  return output;
}
