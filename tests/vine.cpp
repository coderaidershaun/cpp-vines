//! cmake --build build --target vine
//! ./build/vine

#include <bit>
#include <cmath>
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
  if (vine.trees().size() != 2) {
    throw std::runtime_error("Expected two fitted vine trees");
  }

  verify_tree(vine.trees()[0], 2, 0);
  verify_tree(vine.trees()[1], 1, 1);
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

  Vine vine(std::move(marginals), 3);
  const auto fit_result = vine.fit();

  if (!fit_result) {
    throw std::runtime_error(std::string(error_to_string(fit_result.error())));
  }

  verify_vine(vine);

  const auto cmpi_results = vine.final_results();
  if (
    !cmpi_results ||
    cmpi_results->method != Method::Cmpi ||
    cmpi_results->left_given_right.empty() ||
    cmpi_results->right_given_left.empty()
  ) {
    throw std::runtime_error("Expected CMPI final results");
  }

  std::ostringstream output;
  output << vine;

  const std::string rendered_vine = output.str();
  if (
    !rendered_vine.starts_with("T1: 12, 23\n  copulas: ") ||
    rendered_vine.find("T2: 13|2\n  copulas: ") == std::string::npos ||
    rendered_vine.find("\n  params: [[") == std::string::npos ||
    rendered_vine.find("\n  tau: [") == std::string::npos
  ) {
    throw std::runtime_error("Unexpected vine output");
  }

  return 0;
}
