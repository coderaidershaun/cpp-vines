//! cmake --build build --target graph
//! ./build/graph

#include <iostream>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <series.hpp>
#include <csv.hpp>
#include <stats.hpp>

#include <vine/graph.hpp>

std::vector<AssetStats> load_asset_stats(const std::string& filename) {
  std::array<int, 6> csv_asset_col_indexes{3, 4, 5, 6, 7, 8};
  std::vector<AssetStats> assets;

  int series_size = 0;
  for (int i : csv_asset_col_indexes) {
    const auto prices_res = read_csv(filename, i);
    
    if (!prices_res.has_value()) {
      throw std::runtime_error(
        "Price series failed to load. CSV Column Index: " + std::to_string(i)
      );
    } else if (prices_res.value().size() == 0) {
      throw std::runtime_error(
        "Price series must be greater than zero in size. CSV Column Index: " +
        std::to_string(i)
      );
    }
    
    if (assets.empty()) {
      series_size = prices_res.value().size();
    } else if (prices_res.value().size() != series_size) {
      throw std::runtime_error(
        "Expected price series to have matching sizes. CSV Column Index: " +
        std::to_string(i)
      );
    }

    AssetStats asset(prices_res.value().size());

    for (double price : prices_res.value()) {
      auto res = asset.push_price(price);
      if (!res) {
        throw std::runtime_error(std::string(error_to_string(res.error())));
      }
    }

    assets.emplace_back(asset);
  }

  return assets;
}

int main() {
  std::vector<std::span<const double>> data;
  auto assets = load_asset_stats("prices.csv");

  for (const auto& asset : assets) {
    data.emplace_back(asset.ln_returns());
  }

  Prims prims(std::move(data));
  auto graph_res = prims.build_graph();
  assert(graph_res.has_value());

  for (auto& connection : graph_res.value()) {
    std::cout << connection[0] << ", " << connection[1] << std::endl;
  }

  return 0;
}
