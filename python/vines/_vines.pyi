from enum import Enum
from typing import Sequence

__version__: str


class VinesError(RuntimeError): ...


class Result(Enum):
    FAILURE = 0
    PENDING = 1
    SUCCESS = 2


class OptimiserResults:
    @property
    def f_opt_ll(self) -> float: ...

    @property
    def aic(self) -> float: ...

    @property
    def bic(self) -> float: ...

    @property
    def number_it(self) -> float: ...

    @property
    def number_fev(self) -> float: ...

    @property
    def result(self) -> Result: ...


class CondProbsH:
    @property
    def u1_given_u2(self) -> float: ...

    @property
    def u2_given_u1(self) -> float: ...


class CopulaDensityEstimate:
    @property
    def densities(self) -> list[float]: ...

    @property
    def ln_likelihood(self) -> float: ...


class FixedSeries:
    def __init__(
        self,
        series_capacity: int,
        track_multiple: int = 5,
    ) -> None: ...

    def push(self, item: float) -> None: ...

    @property
    def data(self) -> list[float]: ...

    @property
    def mean(self) -> float | None: ...

    @property
    def sum(self) -> float: ...

    @property
    def last(self) -> float | None: ...

    def __len__(self) -> int: ...


class Asset:
    def __init__(
        self,
        series_size: int,
        price_track_multiple: int = 5,
        ecdf_size: int = 10_000,
    ) -> None: ...

    def push_price(self, price: float) -> None: ...

    def push_ln_return(
        self,
        ln_return: float,
        update_marginals: bool = True,
    ) -> None: ...

    def ln_returns(self) -> list[float]: ...

    def prices(self) -> list[float]: ...

    def u_values(self) -> list[float]: ...


class _Copula:
    @property
    def params(self) -> list[list[float]]: ...

    def fit(self) -> OptimiserResults: ...

    def parameter_count(self) -> int: ...

    def estimate_copula_prob_densities(self) -> CopulaDensityEstimate: ...

    def h_conditional_prob_set(self, u1: float, u2: float) -> CondProbsH: ...

    def estimate_copula_density(self, u1: float, u2: float) -> float: ...

    def h_conditional_prob(self, u1: float, u2: float) -> float: ...

    def name(self) -> str: ...

    def kendalls_tau(self) -> float: ...


class Clayton(_Copula):
    def __init__(
        self,
        u1: Sequence[float],
        u2: Sequence[float],
        alpha_init: float = 0.5,
    ) -> None: ...

    @staticmethod
    def alpha_from_kendalls_tau(tau: float) -> float: ...


class Gaussian(_Copula):
    def __init__(
        self,
        u1: Sequence[float],
        u2: Sequence[float],
        rho_init: float = 0.5,
    ) -> None: ...


Guassian = Gaussian


class Gumbel(_Copula):
    def __init__(
        self,
        u1: Sequence[float],
        u2: Sequence[float],
        delta_init: float = 0.5,
    ) -> None: ...

    @staticmethod
    def delta_from_kendalls_tau(tau: float) -> float: ...


class Independence(_Copula):
    def __init__(
        self,
        u1: Sequence[float],
        u2: Sequence[float],
    ) -> None: ...

    def cdf(self, u1: float, u2: float) -> float: ...


class StudentT(_Copula):
    def __init__(
        self,
        u1: Sequence[float],
        u2: Sequence[float],
        rho_init: float = 0.5,
        nu_init: float = 2.5,
    ) -> None: ...


class Method(Enum):
    CMPI = 0
    CMPI_ZSCORE = 1
    STANDARD = 2


class SelectionCriterion(Enum):
    AIC = 0
    BIC = 1


class FinalProbabilities:
    @property
    def left_given_right(self) -> list[float]: ...

    @property
    def right_given_left(self) -> list[float]: ...


class FinalResults:
    @property
    def method(self) -> Method: ...

    @property
    def left_given_right(self) -> list[float]: ...

    @property
    def right_given_left(self) -> list[float]: ...


class Edge:
    def copula_available(self) -> bool: ...

    def conditioned_left(self) -> int: ...

    def conditioned_right(self) -> int: ...

    def conditioning_set(self) -> int: ...

    def union_set(self) -> int: ...

    def k_tau(self) -> float | None: ...

    def copula_name(self) -> str | None: ...

    def parameter_count(self) -> int | None: ...

    def left_given_right(self) -> list[float]: ...

    def right_given_left(self) -> list[float]: ...

    def conditional_values(self, target: int, given: int) -> list[float]: ...


class Vine:
    def __init__(
        self,
        marginals: Sequence[Sequence[float]],
        max_nodes: int = 6,
        method: Method = Method.CMPI,
        selection_criterion: SelectionCriterion = SelectionCriterion.BIC,
    ) -> None: ...

    def fit(self) -> None: ...

    def trees(self) -> list[list[Edge]]: ...

    def final_probabilities(self) -> FinalProbabilities: ...

    def final_results(self) -> FinalResults: ...

    @property
    def selection_criterion(self) -> SelectionCriterion: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...
