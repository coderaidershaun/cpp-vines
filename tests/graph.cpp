//! cmake --build build --target graph
//! ./build/graph

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <series.hpp>
#include <csv.hpp>
#include <stats.hpp>

#include <vine/graph.hpp>

std::vector<AssetStats> load_asset_stats(const std::string& filename) {
  const std::vector<int> csv_asset_col_indexes{3, 4, 5, 6, 7, 8};
  std::vector<AssetStats> assets;
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

int main() {
  auto assets = load_asset_stats("prices.csv");
  Vine vine(assets);
  vine.build();

  return 0;
}
