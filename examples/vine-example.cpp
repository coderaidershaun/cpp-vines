//! cmake --build build --target example-vine
//! ./build/example-vine

#include <array>
#include <fstream>
#include <iostream>
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

std::vector<AssetStats> load_assets(const std::string& filename) {
  constexpr std::array columns{3, 4, 5, 6, 7, 8};
  std::vector<AssetStats> assets;
  assets.reserve(columns.size());

  usize price_count = 0;

  for (int column : columns) {
    auto prices = read_csv(filename, column);

    if (!prices || prices->empty()) {
      throw std::runtime_error("Failed to load prices from column " + std::to_string(column));
    }

    if (price_count == 0) {
      price_count = prices->size();
    } else if (prices->size() != price_count) {
      throw std::runtime_error("Price columns have different lengths");
    }

    auto& asset = assets.emplace_back(price_count);

    for (double price : *prices) {
      auto result = asset.push_price(price);

      if (!result) {
        throw std::runtime_error(std::string(error_to_string(result.error())));
      }
    }
  }

  return assets;
}

int main() {
  auto assets = load_assets("prices.csv");
  std::vector<std::vector<double>> marginal_values;
  marginal_values.reserve(assets.size());

  for (const auto& asset : assets) {
    marginal_values.emplace_back(asset.u_values());
  }

  Marginals marginals;
  marginals.reserve(marginal_values.size());

  for (const auto& values : marginal_values) {
    marginals.emplace_back(values);
  }

  Vine vine(std::move(marginals), 6, Method::Standard);
  const auto fit_result = vine.fit();

  if (!fit_result) {
    throw std::runtime_error(std::string(error_to_string(fit_result.error())));
  }

  std::cout << vine << '\n';

  const auto results_res = vine.final_results();

  if (!results_res) {
    throw std::runtime_error(std::string(error_to_string(results_res.error())));
  }

  const Edge& final_edge = vine.trees().back().front();
  const usize left_asset = asset_index_from_bitmask(final_edge.conditioned_left()) + 1;
  const usize right_asset = asset_index_from_bitmask(final_edge.conditioned_right()) + 1;

  const char* result_name = "conditional probabilities";

  if (results_res->method == Method::Cmpi) {
    result_name = "CMPI values";
  } else if (results_res->method == Method::CmpiZscore) {
    result_name = "CMPI z-scores";
  }

  constexpr const char* results_filename = "vine-results.txt";
  std::ofstream results_file(results_filename);

  if (!results_file) {
    throw std::runtime_error("Failed to open results file");
  }

  results_file << "observation,U" << left_asset << " <= u" << left_asset
               << ",U" << right_asset << " <= u" << right_asset << '\n';

  for (usize i = 0; i < results_res->left_given_right.size(); i++) {
    results_file << i + 1 << ','
                 << results_res->left_given_right[i] << ','
                 << results_res->right_given_left[i] << '\n';
  }

  if (!results_file) {
    throw std::runtime_error("Failed to write results file");
  }

  std::cout << "\nWrote all final " << result_name << " to "
            << results_filename << '\n';

  std::cout << "\nLatest final " << result_name << ":\n"
            << "left_given_right (U" << left_asset << " <= u" << left_asset
            << "): " << results_res->left_given_right.back() << '\n'
            << "right_given_left (U" << right_asset << " <= u" << right_asset
            << "): " << results_res->right_given_left.back() << '\n';
}
