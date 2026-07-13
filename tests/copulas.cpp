//! cmake --build build --target copulas
//! ./build/copulas

#include <cassert>
#include <cmath>
#include <expected>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <asset.hpp>
#include <concordance.hpp>
#include <copulas/clayton.hpp>
#include <copulas/guassian.hpp>
#include <copulas/gumbel.hpp>
#include <copulas/studentt.hpp>
#include <csv.hpp>
#include <errors.hpp>
#include <types.hpp>
#include <vine/edge.hpp>


std::pair<Asset, Asset> load_assets(const std::string& filename) {
  auto ln_returns_asset_1_res = read_csv(filename, 0);
  auto ln_returns_asset_2_res = read_csv(filename, 1);

  if (!ln_returns_asset_1_res ||  !ln_returns_asset_2_res) {
    throw std::runtime_error("Failed to load from filename: " + filename);
  }

  if (ln_returns_asset_1_res->size() != ln_returns_asset_2_res->size()) {
    throw std::runtime_error("Loaded series have different lengths");
  }
  
  std::vector<double> ln_returns_asset_1 = *ln_returns_asset_1_res; 
  std::vector<double> ln_returns_asset_2 = *ln_returns_asset_2_res; 
  const usize n = ln_returns_asset_1.size();

  Asset asset_1(n);
  Asset asset_2(n);

  for (usize i=0; i<n; i++) {
    asset_1.push_ln_return(ln_returns_asset_1[i]);
    asset_2.push_ln_return(ln_returns_asset_2[i]);
  }

  return {std::move(asset_1), std::move(asset_2)};
}


int main() {
  auto [asset_1, asset_2] = load_assets("prices.csv");

  const std::vector<double> u1 = asset_1.u_values();
  const std::vector<double> u2 = asset_2.u_values();
  const std::span<const double> u1_view = u1;
  const std::span<const double> u2_view = u2;

  std::expected<double, SmartError> k_tau_res = kendalls_tau(u1, u2);
  assert(k_tau_res.has_value());
  double k_tau = *k_tau_res;
  
  double alpha_init = Clayton::alpha_from_kendalls_tau(k_tau);
  Clayton clayton(u1_view, u2_view, alpha_init);
  clayton.fit();
  
  Guassian guassian(u1_view, u2_view, k_tau);
  guassian.fit();
  
  double delta_init = Gumbel::delta_from_kendalls_tau(k_tau);
  Gumbel gumbel(u1_view, u2_view, delta_init);
  gumbel.fit();

  StudentT student_t(u1_view, u2_view, k_tau);
  student_t.fit();

  Edge edge(bitmask(0), bitmask(1), bitmask_empty());
  assert(!edge.copula_available());
  assert(!edge.left_given_right().has_value());
  assert(!edge.right_given_left().has_value());
  const auto missing_conditionals = edge.conditional_values(bitmask(0), bitmask(1));
  assert(!missing_conditionals.has_value());
  assert(missing_conditionals.error() == SmartError::MissingConditionalValues);

  const auto edge_fit = edge.fit(u1_view, u2_view, k_tau);
  assert(edge_fit.has_value());
  assert(edge.copula_available());
  assert(edge.k_tau().has_value());
  assert(std::fabs(*edge.k_tau() - k_tau) < 1e-12);

  const auto left_given_right = edge.conditional_values(bitmask(0), bitmask(1));
  assert(left_given_right.has_value());
  assert(left_given_right->size() == u1.size());
  assert(edge.left_given_right().has_value());
  assert(edge.left_given_right()->data() == left_given_right->data());

  const auto right_given_left = edge.conditional_values(bitmask(1), bitmask(0));
  assert(right_given_left.has_value());
  assert(right_given_left->size() == u2.size());
  assert(edge.right_given_left().has_value());
  assert(edge.right_given_left()->data() == right_given_left->data());

  const auto unavailable_conditionals = edge.conditional_values(bitmask(0), bitmask_empty());
  assert(!unavailable_conditionals.has_value());
  assert(unavailable_conditionals.error() == SmartError::MissingConditionalValues);

  const std::span<const double> empty;
  const auto failed_fit = Edge::fit_optimal_copula(empty, empty, 0.0);
  assert(!failed_fit.has_value());
  assert(failed_fit.error() == SmartError::ArrayLengthZero);

  Edge failed_edge(bitmask(0), bitmask(1), bitmask_empty());
  const auto failed_edge_fit = failed_edge.fit(empty, empty, 0.0);
  assert(!failed_edge_fit.has_value());
  assert(failed_edge_fit.error() == SmartError::ArrayLengthZero);
  assert(!failed_edge.copula_available());
  assert(!failed_edge.k_tau().has_value());

  CondProbsH h_clayton = clayton.h_conditional_prob_set(u1[0], u2[0]);
  assert(std::fabs(clayton.params()[0][0] - 2.75616) < 1e-3);
  assert(std::fabs(h_clayton.u1_given_u2 - 0.609068) < 1e-3);
  assert(std::fabs(h_clayton.u2_given_u1 - 0.879606) < 1e-3);

  CondProbsH h_guassian = guassian.h_conditional_prob_set(u1[0], u2[0]);
  assert(std::fabs(guassian.params()[0][0] - 0.823819) < 1e-3);
  assert(std::fabs(h_guassian.u1_given_u2 - 0.315475) < 1e-3);
  assert(std::fabs(h_guassian.u2_given_u1 - 0.908858) < 1e-3);

  CondProbsH h_gumbel = gumbel.h_conditional_prob_set(u1[0], u2[0]);
  assert(std::fabs(gumbel.params()[0][0] - 2.43675) < 1e-3);
  assert(std::fabs(h_gumbel.u1_given_u2 - 0.183306) < 1e-3);
  assert(std::fabs(h_gumbel.u2_given_u1 - 0.955891) < 1e-3);

  CondProbsH h_student_t = student_t.h_conditional_prob_set(u1[0], u2[0]);
  assert(std::fabs(student_t.params()[0][0] - 0.836974) < 1e-3);
  assert(std::fabs(student_t.params()[1][0] - 3.61791) < 1e-3);
  assert(std::fabs(h_student_t.u1_given_u2 - 0.239538) < 1e-3);
  assert(std::fabs(h_student_t.u2_given_u1 - 0.943748) < 1e-3);

  return 0;
}
