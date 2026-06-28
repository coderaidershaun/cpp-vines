#pragma once

#include <algorithm>    // std::move algorithm
#include <vector>       // std::vector
#include <cassert>      // assert
#include <span>         // std::span, non-owning view
#include <utility>      // std::move cast
#include <concepts>     // std::integral, std::floating_point, std::same_as, std::copyable
#include <type_traits>  // type traits utilities
#include <optional>     // std::optional, std::nullopt
#include <stdexcept>    // std::invalid_argument

#include <types.hpp>    // usize


template <typename T>
concept Number =
  (std::integral<T> || std::floating_point<T>) &&
  !std::same_as<T, bool>;

template <typename T>
class SeriesMean {};

template <Number T>
class SeriesMean<T> {
  protected:
  T m_sum{};
};

template <typename T>
requires std::copyable<T> && std::default_initializable<T>
class FixedSeries : public SeriesMean<T> {

  std::vector<T> m_data_track{};
  usize m_data_track_capacity;
  usize m_series_capacity;
  usize m_head_i = 0;
  usize m_start_i = 0;

  public:
  FixedSeries(usize series_capacity, usize track_multiple)
    : m_data_track_capacity(series_capacity * track_multiple),
      m_series_capacity(series_capacity)
  {
    if (series_capacity == 0) {
      throw std::invalid_argument("series_capacity must be greater than 0");
    }

    if (track_multiple <= 1) {
      throw std::invalid_argument("track_multiple must be greater than 1");
    }

    m_data_track.reserve(m_data_track_capacity);
  }

  void push(T item) {
    prepare_for_push();
    m_data_track[m_head_i] = std::move(item);
    finish_push();
  }

  std::span<const T> data() const {
    return std::span<const T>(
      m_data_track.data() + m_start_i,
      m_head_i - m_start_i
    );
  }

  std::optional<double> mean() const requires Number<T> {
    if (size() == 0) return std::nullopt;
    return static_cast<double>(sum()) / static_cast<double>(size());
  }

  T sum() const requires Number<T> {
    return this->m_sum;
  }

  usize size() const {
    return m_head_i - m_start_i;
  }

  std::optional<T> last() const {
    if (m_head_i == m_start_i) return std::nullopt;
    return m_data_track[m_head_i - 1];
  }

  private:
  bool end_of_track() const {
    return m_head_i == m_data_track_capacity;
  }

  bool size_filled() const {
    return m_head_i > m_series_capacity;
  }

  inline void move_series_to_start() {
    const usize n = size();

    std::move(
      m_data_track.begin() + m_start_i,
      m_data_track.begin() + m_head_i,
      m_data_track.begin()
    );

    m_start_i = 0;
    m_head_i = n;
  }

  inline void update_sum_and_size() requires Number<T> {
    this->m_sum += m_data_track[m_head_i];
    if (size() == m_series_capacity) {
      this->m_sum -= m_data_track[m_start_i];
    }
  }

  inline void prepare_for_push() {
    if (end_of_track()) move_series_to_start();
  }

  inline void finish_push() {
    if constexpr (Number<T>) update_sum_and_size();
    m_head_i += 1;
    if (size_filled()) m_start_i += 1;
  }
};
