//! Calculates Regular (R) Vine structure using Prim's algorithm.
//! Note: 
//!   This implementation steals parts from both Kruskals and Prims
//!   algorithms. Essentially, Prim is used, but where the starting node
//!   uses the highest correlation and there for is selected along with its edge
//!   supported by generating tau rankings.

#pragma once

#include <algorithm>
#include <iostream>
#include <expected>
#include <set>
#include <span>
#include <vector>

#include <concordance.hpp>

using Data = std::vector<std::span<const double>>;

using Edge = std::array<int, 2>;

using GraphEdges = std::vector<Edge>;

struct KTauResult {
  int a;
  int b;
  double tau;
};

class Prims {
  Data m_data;

  public:
  Prims(Data data) : m_data(std::move(data)) {}


  /// Build a connected graph without any closed loops and
  /// following Kruskalls and Prims methods. The output will be a set
  /// of edges (somewhat in order of concordance).
  std::expected<GraphEdges, SmartError> build_graph() {
    GraphEdges graph{};
    int target_edges = m_data.size() - 1;

    auto tau_results = generate_tau_rankings();
    if (!tau_results) {
      return std::unexpected(tau_results.error());
    } else if (tau_results.value().empty()) {
      return std::unexpected(SmartError::ArrayLengthZero);
    }

    std::set<int> connected{};

    // First item has the highest correlation so this is our
    // starting connection.
    auto& taus = tau_results.value();
    graph.emplace_back(Edge{taus[0].a, taus[0].b});
    connected.insert(taus[0].a);
    connected.insert(taus[0].b);

    // Connect nodes with edges
    while (graph.size() < target_edges) {
      bool added_edge = false;

      for (const auto& tau : taus) {
        const bool a_connected = connected.contains(tau.a);
        const bool b_connected = connected.contains(tau.b);

        if (a_connected != b_connected) {
          graph.emplace_back(Edge{tau.a, tau.b});
          connected.insert(tau.a);
          connected.insert(tau.b);
          added_edge = true;
          break;
        }
      }
    }

    return graph;
  }

  private:

  /// Generates individual pair correlations ranked in order of highest 
  /// correlated to lowest. Useful for Kruskalls or Prims algorith to 
  /// identify a starting connected pair of nodes.
  std::expected<std::vector<KTauResult>, SmartError> generate_tau_rankings() {
    
    std::set<int> connected{};
    std::vector<KTauResult> tau_results;

    for (int i=0; i<m_data.size(); i++) {
      for (int j=i+1; j<m_data.size(); j++) {
        auto tau_result = kendals_tau(m_data[i], m_data[j]);
        if (!tau_result.has_value()) {
          return std::unexpected(tau_result.error());
        }

        tau_results.emplace_back(KTauResult {i, j, *tau_result});
      }
    }

    std::sort(tau_results.begin(), tau_results.end(), [](const KTauResult&x, const KTauResult&y) {
      return x.tau > y.tau;
    });

    return tau_results;
  }
};
