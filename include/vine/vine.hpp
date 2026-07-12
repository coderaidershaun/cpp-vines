#pragma once

#include <algorithm>
#include <cmath>
#include <expected>
#include <numeric>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <concordance.hpp>
#include <stats.hpp>
#include <types.hpp>
#include <vine/edge.hpp>

using Marginals = std::vector<std::span<const double>>;

using Edges = std::vector<Edge>;

enum class Method {
  Cmpi,
  Standard
};

struct Vine {
  Marginals m_marginals;
  usize m_max_nodes;
  Method m_method;
  usize m_tree_number{};

  Vine(
    Marginals marginals,
    usize max_nodes = 6,
    Method method = Method::Cmpi
  )
    : m_max_nodes(std::min(marginals.size(), max_nodes)),
      m_marginals(std::move(marginals)),
      m_method(method)
  {}

  std::expected<void, SmartError> fit() {
    auto tau_rankings_res = build_initial_rankings();
    if (!tau_rankings_res) return std::unexpected(tau_rankings_res.error());

    auto init_tree = build_tree(std::move(*tau_rankings_res));

    return {};
  }

  private:

  /// Builds initial rankings of all provided values. For example
  /// you might wish to provide the ln returns for 10 assets. This
  /// function will determine the concordance for them all and rank
  /// them as such in descending order of absolute kendalls tau.
  std::expected<Edges, SmartError> build_initial_rankings() {
    Edges rankings{};
    rankings.reserve(static_cast<usize>((m_max_nodes * (m_max_nodes - 1.0)) / 2.0));

    for (usize i=0; i<m_marginals.size(); i++) {
      for (usize j=i+1; j<m_marginals.size(); j++) {
        auto k_tau_res = kendals_tau(m_marginals[i], m_marginals[j]);
        if (!k_tau_res) return std::unexpected(k_tau_res.error());
        Edge edge(bitmask(i), bitmask(j), bitmask_empty(), *k_tau_res);
        rankings.emplace_back(edge);
      }
    }

    std::sort(rankings.begin(), rankings.end(), [](const Edge& x, const Edge& y) {
      return abs(*x.k_tau()) > abs(*y.k_tau());
    });

    return rankings;
  }

  /// Builds new tree based on prior tree
  std::expected<Edges, SmartError> build_tree(Edges edges_prior) {
    m_tree_number += 1;
    int edges_new_size_allowance = m_max_nodes - m_tree_number;

    Edges edges_new{};
    edges_new.reserve(edges_new_size_allowance);

    // Preparation for cycle closed loop prevention
    std::vector<int> parent(edges_prior.size());
    std::iota(parent.begin(), parent.end(), 0);
    auto find_root = [&](this auto&& self, int node) -> int {
      if (parent[node] != node) parent[node] = self(parent[node]);
      return parent[node];
    };

    for (int i=0; i<edges_prior.size(); i++) {

      // Guard: Ensure size matches allowance for this tree
      if (edges_new.size() == edges_new_size_allowance) break;

      // Initialise
      Mask a_conditioned_left = edges_prior[i].conditioned_left();
      Mask a_conditioned_right = edges_prior[i].conditioned_right();
      Mask a_conditioning_set = edges_prior[i].conditioning_set();
      Mask a_union = edges_prior[i].union_set();
      std::optional<double> a_weighting_opt = edges_prior[i].k_tau();
      if (!a_weighting_opt) return std::unexpected(SmartError::MissingWeighting);

      // Determine tree connections
      for (int j=i+1; j<edges_prior.size(); j++) {

        // Proximity check: conditioning intercept
        Mask b_union = edges_prior[j].union_set();
        Mask conditioning_set = a_union & b_union;
        int conditioning_count = count_bits(conditioning_set);
        if (!(conditioning_count == m_tree_number)) continue;

        // Proximity check: singletons on differences
        Mask conditioned_left = a_union & ~b_union;
        Mask conditioned_right = b_union & ~a_union;
        if (!(is_singleton(conditioned_left))) continue;
        if (!(is_singleton(conditioned_right))) continue;

        // Cycle check: ensure no loops closed
        int root_i = find_root(i);
        int root_j = find_root(j);
        if (root_i == root_j) continue;

        // Ordering check: ensure left conditioned less than right
        if (conditioned_left > conditioned_right) {
          std::swap(conditioned_left, conditioned_right);
        }

        // Create new edge
        Edge edge_new(
          conditioned_left,
          conditioned_right,
          conditioning_set
        );
        
        // The first tree will not yet have copulas attached. So u1 and u2 marginals are not
        // based on conditional h functions but rather the u1 and u2 marginals directly.
        std::span<const double> u1;
        std::span<const double> u2;
        if (!edges_prior[i].copula_available()) {
          u1 = m_marginals[asset_index_from_bitmask(conditioned_left)];
          u2 = m_marginals[asset_index_from_bitmask(conditioned_right)];
        } else {
          // TODO! Need some kind of solution for looking up the h function here.
        }

        // Fit copula and append edge
        auto new_k_tau_res = kendals_tau(u1, u2);
        if (!new_k_tau_res) return std::unexpected(new_k_tau_res.error());
        edge_new.fit_optimal_copula(u1, u2, *new_k_tau_res);
        edges_new.emplace_back(std::move(edge_new));
        parent[root_j] = root_i;
      }
    }

    std::sort(edges_new.begin(), edges_new.end(), [](const Edge& x, const Edge& y) {
      return abs(*x.k_tau()) > abs(*y.k_tau());
    });

    return edges_new;
  }

  static constexpr int count_bits(Mask mask) {
    return std::popcount(mask);
  }

  static constexpr bool is_singleton(Mask mask_check) {
    return count_bits(mask_check) == 1;
  }
};
