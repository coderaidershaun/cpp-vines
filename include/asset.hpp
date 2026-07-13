#pragma once

//! Stores an asset's prices, log returns, and empirical marginal values.

#include <cmath>
#include <expected>
#include <optional>
#include <span>
#include <vector>

#include <ecdf.hpp>
#include <errors.hpp>
#include <series.hpp>
#include <types.hpp>


class Asset {

  FixedSeries<double> m_prices;
  FixedSeries<double> m_ln_returns;
  Ecdf m_ecdf;

  public:
  Asset(
    usize series_size,
    usize price_track_multiple=5,
    usize ecdf_size=10'000
  )
    : m_prices(series_size, price_track_multiple),
      m_ln_returns(series_size, price_track_multiple),
      m_ecdf(ecdf_size)
  {}

  std::expected<void, SmartError> push_price(double price) {
    const std::optional<double> last_price_opt = m_prices.last();

    const std::expected<void, SmartError> price_push_res = add_price(price);
    if (!price_push_res) return std::unexpected(price_push_res.error());

    if (!last_price_opt) return {};
    push_ln_return(std::log(price / *last_price_opt));

    return {};
  }

  void push_ln_return(double ln_return, bool update_marginals=true) {
    m_ln_returns.push(ln_return);
    m_ecdf.push(ln_return);
  }

  std::span<const double> ln_returns() const {
    return std::span(m_ln_returns.data().begin(), m_ln_returns.data().end());
  }

  std::span<const double> prices() const {
    return std::span(m_prices.data().begin(), m_prices.data().end());
  }

  std::vector<double> u_values() const {
    const std::span<const double> ln_returns_values = m_ln_returns.data();
    const usize n = ln_returns_values.size();

    std::vector<double> u_values{};
    u_values.reserve(n);
    
    for (usize i=0; i<n; i++) {
      u_values.emplace_back(m_ecdf.lookup_value(ln_returns_values[i]));
    }

    return u_values;
  }

  private:
  static std::expected<void, SmartError> check_price(double price) {
    if (price == 0.0) {
      return std::unexpected(SmartError::PriceZero);
    } else if (price < 0.0) {
      return std::unexpected(SmartError::PriceNegative);
    } else if (!std::isfinite(price)) {
      return std::unexpected(SmartError::PriceInfinite);
    } else if (std::isnan(price)) {
      return std::unexpected(SmartError::PriceNan);
    }

    return {};
  }

  inline std::expected<void, SmartError> add_price(double price) {
    const std::expected<void, SmartError> price_ok = check_price(price);
    if (!price_ok) { return std::unexpected(price_ok.error()); }
    m_prices.push(price);
    return {};
  }
};
