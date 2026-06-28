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

  const double clayton_param_init = 2.75683214734828;
  const double gumbel_param_init = 2.43657044645699;
  const double guassian_param_init = 0.823818841558402;
  const double student_param_rho_init = 0.836763621513228;
  const double student_param_nu_init = 3.62129898479099;
  
  Clayton clayton(u1, u2, clayton_param_init);
  CondProbsH h_clayton = clayton.h_conditional_prob_set(u1[0], u2[0]);
  std::cout << "Clayton H Function Conditional Probabilities:" << std::endl;
  std::cout << "u1_given_u2: " << h_clayton.u1_given_u2 << std::endl;
  std::cout << "u2_given_u1: " << h_clayton.u2_given_u1 << std::endl;

  Guassian guassian(u1, u2, guassian_param_init);
  CondProbsH h_guassian = guassian.h_conditional_prob_set(u1[0], u2[0]);
  std::cout << "\nGuassian H Function Conditional Probabilities:" << std::endl;
  std::cout << "u1_given_u2: " << h_guassian.u1_given_u2 << std::endl;
  std::cout << "u2_given_u1: " << h_guassian.u2_given_u1 << std::endl;
  
  Gumbel gumbel(u1, u2, gumbel_param_init);
  CondProbsH h_gumbel = gumbel.h_conditional_prob_set(u1[0], u2[0]);
  std::cout << "\nGumbel H Function Conditional Probabilities:" << std::endl;
  std::cout << "u1_given_u2: " << h_gumbel.u1_given_u2 << std::endl;
  std::cout << "u2_given_u1: " << h_gumbel.u2_given_u1 << std::endl;

  StudentT student_t(u1, u2, student_param_rho_init, student_param_nu_init);
  CondProbsH h_student_t = student_t.h_conditional_prob_set(u1[0], u2[0]);
  std::cout << "\nStudentT H Function Conditional Probabilities:" << std::endl;
  std::cout << "u1_given_u2: " << h_student_t.u1_given_u2 << std::endl;
  std::cout << "u2_given_u1: " << h_student_t.u2_given_u1 << std::endl;

  return 0;
}
