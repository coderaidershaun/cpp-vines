//! cmake --build build --target vine
//! ./build/vine

#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <asset.hpp>
#include <csv.hpp>
#include <errors.hpp>
#include <types.hpp>
#include <vine/edge.hpp>
#include <vine/vine.hpp>

std::vector<Asset> load_assets(const std::string& filename) {
  const std::vector<int> csv_asset_col_indexes{3, 4, 5, 6, 7, 8};
  std::vector<Asset> assets;
  assets.reserve(csv_asset_col_indexes.size());

  usize series_size = 0;
  for (usize asset_i=0; asset_i<csv_asset_col_indexes.size(); asset_i++) {
    int csv_col_i = csv_asset_col_indexes[asset_i];
    auto prices_res = read_csv(filename, csv_col_i);
    
    if (!prices_res.has_value()) {
      throw std::runtime_error(
        "Price series failed to load. CSV Column Index: " +
        std::to_string(csv_col_i)
      );
    } else if (prices_res.value().size() == 0) {
      throw std::runtime_error(
        "Price series must be greater than zero in size. CSV Column Index: " +
        std::to_string(csv_col_i)
      );
    }
    
    if (asset_i == 0) {
      series_size = prices_res.value().size();
    } else if (prices_res.value().size() != series_size) {
      throw std::runtime_error(
        "Expected price series to have matching sizes. CSV Column Index: " +
        std::to_string(csv_col_i)
      );
    }

    auto& asset = assets.emplace_back(series_size);
    for (double price : prices_res.value()) {
      auto res = asset.push_price(price);
      if (!res) {
        throw std::runtime_error(std::string(error_to_string(res.error())));
      }
    }
  }

  return assets;
}

void verify_tree(
  const Edges& tree,
  usize expected_edges,
  usize expected_conditioning_size
) {
  if (tree.size() != expected_edges) {
    throw std::runtime_error("Unexpected fitted vine tree size");
  }

  for (usize i = 0; i < tree.size(); i++) {
    const Edge& edge = tree[i];
    const auto tau = edge.k_tau();

    if (!edge.copula_available() || !tau) {
      throw std::runtime_error("Expected every vine edge to be fitted");
    }

    if (std::popcount(edge.conditioning_set()) != expected_conditioning_size) {
      throw std::runtime_error("Unexpected edge conditioning set size");
    }

    if (i > 0 && std::abs(*tree[i - 1].k_tau()) < std::abs(*tau)) {
      throw std::runtime_error("Expected edges to be ranked by absolute tau");
    }
  }
}

void verify_vine(const Vine& vine) {
  if (vine.trees().size() != 3) {
    throw std::runtime_error("Expected three fitted vine trees");
  }

  verify_tree(vine.trees()[0], 3, 0);
  verify_tree(vine.trees()[1], 2, 1);
  verify_tree(vine.trees()[2], 1, 2);
}

int main() {
  auto assets = load_assets("prices.csv");

  std::vector<std::vector<double>> marginal_values{};
  marginal_values.reserve(assets.size());

  for (const Asset& asset : assets) {
    marginal_values.emplace_back(asset.u_values());
  }

  Marginals marginals{};
  marginals.reserve(marginal_values.size());

  for (const std::vector<double>& marginal : marginal_values) {
    marginals.emplace_back(marginal);
  }

  Vine too_small({marginals.front()}, 1);
  const auto unfitted_probabilities = too_small.final_probabilities();
  const auto unfitted_results = too_small.final_results();

  if (
    unfitted_probabilities ||
    unfitted_results ||
    unfitted_probabilities.error() != SmartError::CopulaNotFitted ||
    unfitted_results.error() != SmartError::CopulaNotFitted
  ) {
    throw std::runtime_error("Expected final inference to require a fitted vine");
  }

  const auto too_small_result = too_small.fit();

  if (too_small_result || too_small_result.error() != SmartError::ArrayLengthZero) {
    throw std::runtime_error("Expected a vine with fewer than two nodes to fail");
  }

  std::ostringstream empty_output;
  empty_output << too_small;

  if (!empty_output.str().empty()) {
    throw std::runtime_error("Expected an unfitted vine to print nothing");
  }

  Vine vine(std::move(marginals), 4);
  const auto fit_result = vine.fit();

  if (!fit_result) {
    throw std::runtime_error(std::string(error_to_string(fit_result.error())));
  }

  verify_vine(vine);

  const auto final_probabilities = vine.final_probabilities();
  if (!final_probabilities) {
    throw std::runtime_error(std::string(error_to_string(final_probabilities.error())));
  }

  const Edge& final_edge = vine.trees().back().front();
  const auto edge_left_given_right = final_edge.left_given_right();
  const auto edge_right_given_left = final_edge.right_given_left();

  if (!edge_left_given_right || !edge_right_given_left) {
    throw std::runtime_error("Expected the final edge probabilities to be available");
  }

  if (
    final_probabilities->left_given_right.data() != edge_left_given_right->data() ||
    final_probabilities->left_given_right.size() != edge_left_given_right->size() ||
    final_probabilities->right_given_left.data() != edge_right_given_left->data() ||
    final_probabilities->right_given_left.size() != edge_right_given_left->size()
  ) {
    throw std::runtime_error("Expected final probabilities from both edge directions");
  }

  const auto cmpi_results = vine.final_results();
  if (!cmpi_results || cmpi_results->method != Method::Cmpi) {
    throw std::runtime_error("Expected CMPI final results");
  }

  double expected_left_cmpi = 0.0;
  double expected_right_cmpi = 0.0;

  for (usize i = 0; i < final_probabilities->left_given_right.size(); i++) {
    expected_left_cmpi += final_probabilities->left_given_right[i] - 0.5;
    expected_right_cmpi += final_probabilities->right_given_left[i] - 0.5;

    if (
      std::abs(cmpi_results->left_given_right[i] - expected_left_cmpi) > 1e-12 ||
      std::abs(cmpi_results->right_given_left[i] - expected_right_cmpi) > 1e-12
    ) {
      throw std::runtime_error("Unexpected cumulative CMPI result");
    }
  }

  Marginals standard_marginals{
    std::span<const double>(marginal_values[0]),
    std::span<const double>(marginal_values[1])
  };
  Vine standard_vine(std::move(standard_marginals), 2, Method::Standard);
  const auto standard_fit = standard_vine.fit();
  const auto standard_probabilities = standard_vine.final_probabilities();
  const auto standard_results = standard_vine.final_results();

  if (!standard_fit || !standard_probabilities || !standard_results) {
    throw std::runtime_error("Expected Standard final results");
  }

  if (
    standard_results->method != Method::Standard ||
    !std::equal(
      standard_results->left_given_right.begin(),
      standard_results->left_given_right.end(),
      standard_probabilities->left_given_right.begin()
    ) ||
    !std::equal(
      standard_results->right_given_left.begin(),
      standard_results->right_given_left.end(),
      standard_probabilities->right_given_left.begin()
    )
  ) {
    throw std::runtime_error("Expected Standard results to contain raw probabilities");
  }

  Marginals zscore_marginals{
    std::span<const double>(marginal_values[0]),
    std::span<const double>(marginal_values[1])
  };
  Vine zscore_vine(std::move(zscore_marginals), 2, Method::CmpiZscore);
  const auto zscore_fit = zscore_vine.fit();
  const auto zscore_probabilities = zscore_vine.final_probabilities();
  const auto zscore_results = zscore_vine.final_results();

  if (!zscore_fit || !zscore_probabilities || !zscore_results) {
    throw std::runtime_error("Expected CMPI z-score final results");
  }

  const auto verify_zscores = [](
    std::span<const double> probabilities,
    const std::vector<double>& actual
  ) {
    std::vector<double> expected;
    expected.reserve(probabilities.size());

    double cumulative = 0.0;
    double sum = 0.0;

    for (double probability : probabilities) {
      cumulative += probability - 0.5;
      expected.emplace_back(cumulative);
      sum += cumulative;
    }

    const double mean = sum / static_cast<double>(expected.size());
    double squared_deviations = 0.0;

    for (double value : expected) {
      const double deviation = value - mean;
      squared_deviations += deviation * deviation;
    }

    const double standard_deviation = std::sqrt(
      squared_deviations / static_cast<double>(expected.size())
    );

    for (usize i = 0; i < expected.size(); i++) {
      expected[i] = (expected[i] - mean) / standard_deviation;

      if (std::abs(actual[i] - expected[i]) > 1e-12) {
        return false;
      }
    }

    return true;
  };

  if (
    zscore_results->method != Method::CmpiZscore ||
    !verify_zscores(
      zscore_probabilities->left_given_right,
      zscore_results->left_given_right
    ) ||
    !verify_zscores(
      zscore_probabilities->right_given_left,
      zscore_results->right_given_left
    )
  ) {
    throw std::runtime_error("Unexpected CMPI z-score result");
  }

  const auto refit_result = vine.fit();
  if (!refit_result) {
    throw std::runtime_error(std::string(error_to_string(refit_result.error())));
  }

  verify_vine(vine);

  std::ostringstream output;

  if (&(output << vine) != &output) {
    throw std::runtime_error("Expected vine output to return its stream");
  }

  const std::string expected_output =
    "T1: 12, 23, 25\n"
    "T2: 35|2, 15|2\n"
    "T3: 13|25";

  if (output.str() != expected_output) {
    throw std::runtime_error("Unexpected vine output");
  }

  return 0;
}
