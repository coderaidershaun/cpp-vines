# CPP-VINES

The Boost Math library is required for the Student-t and Gaussian copulas because
they use inverse cumulative distribution functions.

```shell
brew install boost
```

## Python interface

The Python package exposes the numeric (`double`) `FixedSeries`, `Vine`, and
every concrete copula in `include/copulas`. It is built with pybind11 and
scikit-build-core; use
[uv](https://docs.astral.sh/uv/) to create and manage the local environment:

```shell
uv sync
```

The environment is created at `.venv/`, which is ignored by Git.
The package includes type information for editors and static type checkers.

```python
from vines import Method, Vine

u1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
u2 = [0.2, 0.1, 0.4, 0.3, 0.6, 0.5]

vine = Vine([u1, u2], max_nodes=2, method=Method.STANDARD)
vine.fit()
results = vine.final_results()

print(results.left_given_right)
print(results.right_given_left)
```

Available copulas are `Clayton`, `Gaussian`, `Gumbel`, and `StudentT`.
`Guassian` is retained as an alias for the existing C++ class spelling.

Runnable notebooks for `Vine` and `Clayton` live in `python/examples`. Launch
them in a temporary uv-provided Jupyter environment with:

```shell
uv run --with jupyter jupyter lab python/examples
```
