//! Python bindings and Python-owned lifetime adapters for cpp-vines.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/native_enum.h>
#include <pybind11/stl.h>

#include <asset.hpp>
#include <copulas/clayton.hpp>
#include <copulas/guassian.hpp>
#include <copulas/gumbel.hpp>
#include <copulas/independence.hpp>
#include <copulas/studentt.hpp>
#include <errors.hpp>
#include <series.hpp>
#include <vine/edge.hpp>
#include <vine/vine.hpp>

namespace py = pybind11;

namespace {

class VinesError : public std::runtime_error {
  public:
  explicit VinesError(SmartError error)
    : std::runtime_error(std::string(error_to_string(error)))
  {}
};

template <typename T>
T expected_value_or_throw(std::expected<T, SmartError> result) {
  if (!result) throw VinesError(result.error());
  return std::move(*result);
}

void expected_value_or_throw(std::expected<void, SmartError> result) {
  if (!result) throw VinesError(result.error());
}

void validate_samples(
  const std::vector<double>& u1,
  const std::vector<double>& u2
) {
  if (u1.empty()) {
    throw std::invalid_argument("u1 and u2 must not be empty");
  }

  if (u1.size() != u2.size()) {
    throw std::invalid_argument("u1 and u2 must have the same length");
  }
}

template <typename CopulaType>
class CopulaBridge {
  std::vector<double> m_u1;
  std::vector<double> m_u2;
  CopulaType m_copula;

  public:
  template <typename... Args>
  CopulaBridge(
    std::vector<double> u1,
    std::vector<double> u2,
    Args... args
  )
    : m_u1(std::move(u1)),
      m_u2(std::move(u2)),
      m_copula(m_u1, m_u2, args...)
  {
    validate_samples(m_u1, m_u2);
  }

  CopulaBridge(const CopulaBridge&) = delete;
  CopulaBridge& operator=(const CopulaBridge&) = delete;
  CopulaBridge(CopulaBridge&&) = delete;
  CopulaBridge& operator=(CopulaBridge&&) = delete;

  OptimiserResults fit() {
    return m_copula.fit();
  }

  std::vector<std::array<double, 3>> params() const {
    const auto& cpp_params = m_copula.params();
    return {cpp_params.begin(), cpp_params.end()};
  }

  usize parameter_count() const {
    return m_copula.parameter_count();
  }

  CopulaDensityEstimate estimate_copula_prob_densities() const {
    return m_copula.estimate_copula_prob_densities();
  }

  CondProbsH h_conditional_prob_set(double u1, double u2) const {
    return m_copula.h_conditional_prob_set(u1, u2);
  }

  double estimate_copula_density(double u1, double u2) const {
    return m_copula.estimate_copula_density(u1, u2);
  }

  double h_conditional_prob(double u1, double u2) const {
    return m_copula.h_conditional_prob(u1, u2);
  }

  std::string name() const {
    return m_copula.name();
  }

  double kendalls_tau() const {
    return m_copula.kendalls_tau();
  }
};

using ClaytonBridge = CopulaBridge<Clayton>;
using GaussianBridge = CopulaBridge<Guassian>;
using GumbelBridge = CopulaBridge<Gumbel>;
using IndependenceBridge = CopulaBridge<Independence>;
using StudentTBridge = CopulaBridge<StudentT>;

struct FinalProbabilitiesBridge {
  std::vector<double> left_given_right;
  std::vector<double> right_given_left;
};

Marginals marginal_views(
  const std::vector<std::vector<double>>& marginal_storage
) {
  Marginals views;
  views.reserve(marginal_storage.size());

  for (const auto& marginal : marginal_storage) {
    views.emplace_back(marginal);
  }

  return views;
}

class VineBridge {
  std::vector<std::vector<double>> m_marginal_storage;
  Marginals m_marginal_views;
  Vine m_vine;

  public:
  VineBridge(
    std::vector<std::vector<double>> marginals,
    usize max_nodes,
    Method method,
    SelectionCriterion selection_criterion
  )
    : m_marginal_storage(std::move(marginals)),
      m_marginal_views(marginal_views(m_marginal_storage)),
      m_vine(m_marginal_views, max_nodes, method, selection_criterion)
  {}

  VineBridge(const VineBridge&) = delete;
  VineBridge& operator=(const VineBridge&) = delete;
  VineBridge(VineBridge&&) = delete;
  VineBridge& operator=(VineBridge&&) = delete;

  void fit() {
    expected_value_or_throw(m_vine.fit());
  }

  std::vector<Edges> trees() const {
    return m_vine.trees();
  }

  SelectionCriterion selection_criterion() const {
    return m_vine.selection_criterion();
  }

  FinalProbabilitiesBridge final_probabilities() const {
    const auto probabilities = expected_value_or_throw(
      m_vine.final_probabilities()
    );

    return {
      .left_given_right = {
        probabilities.left_given_right.begin(),
        probabilities.left_given_right.end()
      },
      .right_given_left = {
        probabilities.right_given_left.begin(),
        probabilities.right_given_left.end()
      }
    };
  }

  FinalResults final_results() const {
    return expected_value_or_throw(m_vine.final_results());
  }

  std::string to_string() const {
    std::ostringstream output;
    output << m_vine;
    return output.str();
  }
};

template <typename Bridge>
void bind_copula_methods(py::class_<Bridge>& cls) {
  cls
    .def("fit", &Bridge::fit, py::call_guard<py::gil_scoped_release>())
    .def_property_readonly("params", &Bridge::params)
    .def("parameter_count", &Bridge::parameter_count)
    .def(
      "estimate_copula_prob_densities",
      &Bridge::estimate_copula_prob_densities
    )
    .def("h_conditional_prob_set", &Bridge::h_conditional_prob_set)
    .def("estimate_copula_density", &Bridge::estimate_copula_density)
    .def("h_conditional_prob", &Bridge::h_conditional_prob)
    .def("name", &Bridge::name)
    .def("kendalls_tau", &Bridge::kendalls_tau);
}

std::vector<double> edge_values(
  std::expected<std::span<const double>, SmartError> result
) {
  const auto values = expected_value_or_throw(std::move(result));
  return {values.begin(), values.end()};
}

} // namespace

PYBIND11_MODULE(_vines, module) {
  module.doc() = "Python bindings for the cpp-vines C++ library";
  module.attr("__version__") = "0.1.0";

  py::register_exception<VinesError>(module, "VinesError");

  py::native_enum<Result>(module, "Result", "enum.Enum")
    .value("FAILURE", Result::Failure)
    .value("PENDING", Result::Pending)
    .value("SUCCESS", Result::Success)
    .finalize();

  py::class_<OptimiserResults>(module, "OptimiserResults")
    .def_readonly("f_opt_ll", &OptimiserResults::f_opt_ll)
    .def_readonly("aic", &OptimiserResults::aic)
    .def_readonly("bic", &OptimiserResults::bic)
    .def_readonly("number_it", &OptimiserResults::number_it)
    .def_readonly("number_fev", &OptimiserResults::number_fev)
    .def_readonly("result", &OptimiserResults::result);

  py::class_<CondProbsH>(module, "CondProbsH")
    .def_readonly("u1_given_u2", &CondProbsH::u1_given_u2)
    .def_readonly("u2_given_u1", &CondProbsH::u2_given_u1);

  py::class_<CopulaDensityEstimate>(module, "CopulaDensityEstimate")
    .def_readonly("densities", &CopulaDensityEstimate::densities)
    .def_readonly("ln_likelihood", &CopulaDensityEstimate::ln_likelihood);

  py::class_<FixedSeries<double>>(module, "FixedSeries")
    .def(
      py::init<usize, usize>(),
      py::arg("series_capacity"),
      py::arg("track_multiple") = 5
    )
    .def("push", &FixedSeries<double>::push, py::arg("item"))
    .def_property_readonly(
      "data",
      [](const FixedSeries<double>& series) {
        const auto values = series.data();
        return std::vector<double>(values.begin(), values.end());
      }
    )
    .def_property_readonly("mean", &FixedSeries<double>::mean)
    .def_property_readonly("sum", &FixedSeries<double>::sum)
    .def_property_readonly("last", &FixedSeries<double>::last)
    .def("__len__", &FixedSeries<double>::size);

  py::class_<Asset>(module, "Asset")
    .def(
      py::init<usize, usize, usize>(),
      py::arg("series_size"),
      py::arg("price_track_multiple") = 5,
      py::arg("ecdf_size") = 10'000
    )
    .def(
      "push_price",
      [](Asset& asset, double price) {
        expected_value_or_throw(asset.push_price(price));
      },
      py::arg("price")
    )
    .def(
      "push_ln_return",
      &Asset::push_ln_return,
      py::arg("ln_return"),
      py::arg("update_marginals") = true
    )
    .def(
      "ln_returns",
      [](const Asset& asset) {
        const auto values = asset.ln_returns();
        return std::vector<double>(values.begin(), values.end());
      }
    )
    .def(
      "prices",
      [](const Asset& asset) {
        const auto values = asset.prices();
        return std::vector<double>(values.begin(), values.end());
      }
    )
    .def("u_values", &Asset::u_values);

  auto clayton = py::class_<ClaytonBridge>(module, "Clayton")
    .def(
      py::init<std::vector<double>, std::vector<double>, double>(),
      py::arg("u1"),
      py::arg("u2"),
      py::arg("alpha_init") = 0.5
    )
    .def_static(
      "alpha_from_kendalls_tau",
      &Clayton::alpha_from_kendalls_tau,
      py::arg("tau")
    );
  bind_copula_methods(clayton);

  auto gaussian = py::class_<GaussianBridge>(module, "Gaussian")
    .def(
      py::init<std::vector<double>, std::vector<double>, double>(),
      py::arg("u1"),
      py::arg("u2"),
      py::arg("rho_init") = 0.5
    );
  bind_copula_methods(gaussian);
  module.attr("Guassian") = module.attr("Gaussian");

  auto gumbel = py::class_<GumbelBridge>(module, "Gumbel")
    .def(
      py::init<std::vector<double>, std::vector<double>, double>(),
      py::arg("u1"),
      py::arg("u2"),
      py::arg("delta_init") = 0.5
    )
    .def_static(
      "delta_from_kendalls_tau",
      &Gumbel::delta_from_kendalls_tau,
      py::arg("tau")
    );
  bind_copula_methods(gumbel);

  auto independence = py::class_<IndependenceBridge>(module, "Independence")
    .def(
      py::init<std::vector<double>, std::vector<double>>(),
      py::arg("u1"),
      py::arg("u2")
    )
    .def("cdf", [](const IndependenceBridge&, double u1, double u2) {
      return Independence::cdf(u1, u2);
    });
  bind_copula_methods(independence);

  auto student_t = py::class_<StudentTBridge>(module, "StudentT")
    .def(
      py::init<std::vector<double>, std::vector<double>, double, double>(),
      py::arg("u1"),
      py::arg("u2"),
      py::arg("rho_init") = 0.5,
      py::arg("nu_init") = 2.5
    );
  bind_copula_methods(student_t);

  py::native_enum<Method>(module, "Method", "enum.Enum")
    .value("CMPI", Method::Cmpi)
    .value("CMPI_ZSCORE", Method::CmpiZscore)
    .value("STANDARD", Method::Standard)
    .finalize();

  py::native_enum<SelectionCriterion>(
    module,
    "SelectionCriterion",
    "enum.Enum"
  )
    .value("AIC", SelectionCriterion::Aic)
    .value("BIC", SelectionCriterion::Bic)
    .finalize();

  py::class_<FinalProbabilitiesBridge>(module, "FinalProbabilities")
    .def_readonly(
      "left_given_right",
      &FinalProbabilitiesBridge::left_given_right
    )
    .def_readonly(
      "right_given_left",
      &FinalProbabilitiesBridge::right_given_left
    );

  py::class_<FinalResults>(module, "FinalResults")
    .def_readonly("method", &FinalResults::method)
    .def_readonly("left_given_right", &FinalResults::left_given_right)
    .def_readonly("right_given_left", &FinalResults::right_given_left);

  py::class_<Edge>(module, "Edge")
    .def("copula_available", &Edge::copula_available)
    .def("conditioned_left", &Edge::conditioned_left)
    .def("conditioned_right", &Edge::conditioned_right)
    .def("conditioning_set", &Edge::conditioning_set)
    .def("union_set", &Edge::union_set)
    .def("k_tau", &Edge::k_tau)
    .def("copula_name", &Edge::copula_name)
    .def("parameter_count", &Edge::parameter_count)
    .def(
      "left_given_right",
      [](const Edge& edge) { return edge_values(edge.left_given_right()); }
    )
    .def(
      "right_given_left",
      [](const Edge& edge) { return edge_values(edge.right_given_left()); }
    )
    .def(
      "conditional_values",
      [](const Edge& edge, Mask target, Mask given) {
        return edge_values(edge.conditional_values(target, given));
      },
      py::arg("target"),
      py::arg("given")
    );

  py::class_<VineBridge>(module, "Vine")
    .def(
      py::init<
        std::vector<std::vector<double>>,
        usize,
        Method,
        SelectionCriterion
      >(),
      py::arg("marginals"),
      py::arg("max_nodes") = 6,
      py::arg("method") = Method::Cmpi,
      py::arg("selection_criterion") = SelectionCriterion::Bic
    )
    .def("fit", &VineBridge::fit, py::call_guard<py::gil_scoped_release>())
    .def("trees", &VineBridge::trees)
    .def("final_probabilities", &VineBridge::final_probabilities)
    .def("final_results", &VineBridge::final_results)
    .def_property_readonly(
      "selection_criterion",
      &VineBridge::selection_criterion
    )
    .def("__str__", &VineBridge::to_string)
    .def("__repr__", [](const VineBridge& vine) {
      return "Vine(" + vine.to_string() + ")";
    });
}
