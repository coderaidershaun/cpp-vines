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
#include <copulas/independence.hpp>
#include <copulas/survival.hpp>
#include <copulas/studentt.hpp>
#include <csv.hpp>
#include <errors.hpp>
#include <types.hpp>


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

template <typename BaseCopula>
void verify_survival_copula(
  const BaseCopula& base,
  const SurvivalCopula<BaseCopula>& survival
) {
  constexpr double u1 = 0.37;
  constexpr double u2 = 0.68;

  const double expected_cdf =
    u1 + u2 - 1.0 + base.cdf(1.0 - u1, 1.0 - u2);
  assert(std::fabs(survival.cdf(u1, u2) - expected_cdf) < 1e-12);
  assert(
    std::fabs(
      survival.estimate_copula_density(u1, u2) -
        base.estimate_copula_density(1.0 - u1, 1.0 - u2)
    ) < 1e-12
  );

  const CondProbsH base_h = base.h_conditional_prob_set(1.0 - u1, 1.0 - u2);
  const CondProbsH survival_h = survival.h_conditional_prob_set(u1, u2);
  assert(std::fabs(survival_h.u1_given_u2 - (1.0 - base_h.u1_given_u2)) < 1e-12);
  assert(std::fabs(survival_h.u2_given_u1 - (1.0 - base_h.u2_given_u1)) < 1e-12);
  assert(std::fabs(survival.kendalls_tau() - base.kendalls_tau()) < 1e-12);
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

  const std::vector<double> rotation_samples{0.2, 0.4, 0.6, 0.8};
  Clayton rotation_clayton(rotation_samples, rotation_samples, 3.0);
  SurvivalCopula<Clayton> survival_clayton(
    rotation_samples,
    rotation_samples,
    3.0
  );
  Gumbel rotation_gumbel(rotation_samples, rotation_samples, 2.5);
  SurvivalCopula<Gumbel> survival_gumbel(
    rotation_samples,
    rotation_samples,
    2.5
  );

  verify_survival_copula(rotation_clayton, survival_clayton);
  verify_survival_copula(rotation_gumbel, survival_gumbel);

  Independence independence(rotation_samples, rotation_samples);
  const CondProbsH independence_h = independence.h_conditional_prob_set(
    0.37,
    0.68
  );
  assert(std::fabs(independence.cdf(0.37, 0.68) - 0.2516) < 1e-12);
  assert(independence.estimate_copula_density(0.37, 0.68) == 1.0);
  assert(independence_h.u1_given_u2 == 0.37);
  assert(independence_h.u2_given_u1 == 0.68);

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
