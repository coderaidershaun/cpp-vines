//! cmake --build build --target copulas
//! ./build/copulas

#include <cassert>
#include <iostream>
#include <optional>
#include <string>

#include <copulas/clayton.hpp>
#include <copulas/guassian.hpp>
#include <copulas/gumbel.hpp>
#include <copulas/studentt.hpp>
#include <csv.hpp>
#include <types.hpp>
#include <stats.hpp>

int main() {
  const std::string filename = "prices.csv";
  auto ln_returns_asset_1_res = read_csv(filename, 0);
  auto ln_returns_asset_2_res = read_csv(filename, 1);

  if (!ln_returns_asset_1_res ||  !ln_returns_asset_2_res) {
    throw ("Failed to load from filename: " + filename);
  }
  
  std::vector<double> ln_returns_asset_1 = *ln_returns_asset_1_res; 
  std::vector<double> ln_returns_asset_2 = *ln_returns_asset_2_res; 
  assert(ln_returns_asset_1.size() == ln_returns_asset_2.size());
  const usize n = ln_returns_asset_1.size();

  AssetStats asset_1(n);
  AssetStats asset_2(n);

  for (usize i=0; i<n; i++) {
    asset_1.push_ln_return(ln_returns_asset_1[i]);
    asset_2.push_ln_return(ln_returns_asset_2[i]);
  }

  std::vector<double> u1 = asset_1.u_values();
  std::vector<double> u2 = asset_2.u_values();
  const double default_param = 0.5;
  
  Clayton clayton(u1, u2, default_param);
  clayton.fit();

  Guassian guassian(u1, u2, default_param);
  guassian.fit();

  Gumbel gumbel(u1, u2, default_param);
  gumbel.fit();

  StudentT student_t(u1, u2, default_param);
  student_t.fit();

  CondProbsH h_clayton = clayton.h_conditional_prob_set(u1[0], u2[0]);
  assert(std::fabs(clayton.params()[0][0] - 2.75524) < 1e-4);
  assert(std::fabs(h_clayton.u1_given_u2 - 0.609068) < 1e-4);
  assert(std::fabs(h_clayton.u2_given_u1 - 0.879606) < 1e-4);

  CondProbsH h_guassian = guassian.h_conditional_prob_set(u1[0], u2[0]);
  assert(std::fabs(guassian.params()[0][0] - 0.823819) < 1e-4);
  assert(std::fabs(h_guassian.u1_given_u2 - 0.315475) < 1e-4);
  assert(std::fabs(h_guassian.u2_given_u1 - 0.908858) < 1e-4);

  CondProbsH h_gumbel = gumbel.h_conditional_prob_set(u1[0], u2[0]);
  assert(std::fabs(gumbel.params()[0][0] - 2.43675) < 1e-4);
  assert(std::fabs(h_gumbel.u1_given_u2 - 0.183306) < 1e-4);
  assert(std::fabs(h_gumbel.u2_given_u1 - 0.955891) < 1e-4);

  CondProbsH h_student_t = student_t.h_conditional_prob_set(u1[0], u2[0]);
  assert(std::fabs(student_t.params()[0][0] - 0.81097) < 1e-4);
  assert(std::fabs(student_t.params()[1][0] - 2.01) < 1e-4);
  assert(std::fabs(h_student_t.u1_given_u2 - 0.236129) < 1e-4);
  assert(std::fabs(h_student_t.u2_given_u1 - 0.956844) < 1e-4);

  return 0;
}
